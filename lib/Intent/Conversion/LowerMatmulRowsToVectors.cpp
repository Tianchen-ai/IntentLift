// Row-blocked mixed-precision contractions with an explicitly packed RHS.
// Match only the scalar linalg body and indexing maps, never a kernel name.
#include "Intent/Conversion/Common/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include <functional>

namespace intent {
#define GEN_PASS_DEF_LOWERMATMULROWSTOVECTORS
#include "Intent/Conversion/Common/Passes.h.inc"
} // namespace intent
using namespace mlir;
namespace {
static bool matches(linalg::GenericOp op, unsigned rows, unsigned lanes) {
  if (!op.hasPureBufferSemantics() || op.getNumLoops() != 3 ||
      op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
    return false;
  auto it = op.getIteratorTypesArray();
  if (it != ArrayRef<utils::IteratorType>({utils::IteratorType::parallel,
          utils::IteratorType::parallel, utils::IteratorType::reduction}))
    return false;
  auto ctx = op.getContext();
  auto i = getAffineDimExpr(0, ctx), j = getAffineDimExpr(1, ctx),
       k = getAffineDimExpr(2, ctx);
  auto maps = op.getIndexingMapsArray();
  if (maps != SmallVector<AffineMap>{AffineMap::get(3, 0, {i, k}, ctx),
          AffineMap::get(3, 0, {j, k}, ctx),
          AffineMap::get(3, 0, {i, j}, ctx)})
    return false;
  auto a = dyn_cast<MemRefType>(op.getDpsInputs()[0].getType());
  auto b = dyn_cast<MemRefType>(op.getDpsInputs()[1].getType());
  auto c = dyn_cast<MemRefType>(op.getDpsInits()[0].getType());
  if (!a || !b || !c || a.getRank() != 2 || b.getRank() != 2 ||
      c.getRank() != 2 || !a.hasStaticShape() || !b.hasStaticShape() ||
      !c.hasStaticShape() || !a.getLayout().isIdentity() ||
      !b.getLayout().isIdentity() || !c.getLayout().isIdentity() ||
      !a.getElementType().isF16() || !b.getElementType().isF16() ||
      !c.getElementType().isF32() || a.getDimSize(1) != b.getDimSize(1) ||
      c.getDimSize(0) != a.getDimSize(0) || c.getDimSize(1) != b.getDimSize(0) ||
      a.getDimSize(0) % rows || b.getDimSize(0) % lanes ||
      !isa<BlockArgument>(op.getDpsInputs()[0]) ||
      !isa<BlockArgument>(op.getDpsInputs()[1]) ||
      !op.getDpsInits()[0].getDefiningOp<memref::AllocOp>())
    return false;
  Block &body = op.getRegion().front();
  if (body.getNumArguments() != 3 ||
      !llvm::hasNItems(body.without_terminator(), 4))
    return false;
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  auto add = yield && yield.getNumOperands() == 1
      ? yield.getOperand(0).getDefiningOp<arith::AddFOp>() : arith::AddFOp{};
  if (!add || add.getLhs() != body.getArgument(2))
    return false;
  auto mul = add.getRhs().getDefiningOp<arith::MulFOp>();
  if (!mul) return false;
  auto left = mul.getLhs().getDefiningOp<arith::ExtFOp>();
  auto right = mul.getRhs().getDefiningOp<arith::ExtFOp>();
  return left && right && left.getIn() == body.getArgument(0) &&
      right.getIn() == body.getArgument(1) &&
      left.getType().isF32() && right.getType().isF32();
}

static void lower(linalg::GenericOp op, unsigned rows, unsigned lanes) {
  OpBuilder builder(op);
  Location loc = op.getLoc();
  Value a = op.getDpsInputs()[0], b = op.getDpsInputs()[1],
        c = op.getDpsInits()[0];
  auto at = cast<MemRefType>(a.getType());
  auto bt = cast<MemRefType>(b.getType());
  int64_t m = at.getDimSize(0), n = bt.getDimSize(0), k = at.getDimSize(1);
  auto halfVector = VectorType::get({lanes}, builder.getF16Type());
  auto floatVector = VectorType::get({lanes}, builder.getF32Type());
  auto packedType = MemRefType::get({n / lanes, k, lanes}, builder.getF16Type());
  Value packed = builder.create<memref::AllocOp>(loc, packedType,
      builder.getI64IntegerAttr(64));
  auto constant = [&](OpBuilder &bb, int64_t value) -> Value {
    return bb.create<arith::ConstantIndexOp>(loc, value);
  };
  auto loop = [&](OpBuilder &bb, int64_t end, int64_t step,
                  std::function<void(OpBuilder &, Value)> emit) {
    auto f = bb.create<scf::ForOp>(loc, constant(bb, 0), constant(bb, end),
                                  constant(bb, step));
    OpBuilder nested = OpBuilder::atBlockBegin(f.getBody());
    emit(nested, f.getInductionVar());
  };
  // Transpose only the RHS layout; each source row is loaded contiguously.
  loop(builder, n / lanes, 1, [&](OpBuilder &nb, Value block) {
    loop(nb, lanes, 1, [&](OpBuilder &lb, Value lane) {
      Value column = lb.create<arith::AddIOp>(loc, lane,
          lb.create<arith::MulIOp>(loc, block, constant(lb, lanes)));
      loop(lb, k, 1, [&](OpBuilder &kb, Value reduction) {
        Value value = kb.create<memref::LoadOp>(loc, b,
                                               ValueRange{column, reduction});
        kb.create<memref::StoreOp>(loc, value, packed,
                                  ValueRange{block, reduction, lane});
      });
    });
  });
  loop(builder, m, rows, [&](OpBuilder &mb, Value row) {
    loop(mb, n / lanes, 1, [&](OpBuilder &nb, Value block) {
      Value column = nb.create<arith::MulIOp>(loc, block, constant(nb, lanes));
      SmallVector<Value> rowIndices, accumulators;
      for (unsigned r = 0; r < rows; ++r) {
        Value ri = nb.create<arith::AddIOp>(loc, row, constant(nb, r));
        rowIndices.push_back(ri);
        accumulators.push_back(nb.create<vector::LoadOp>(loc, floatVector, c,
                                                         ValueRange{ri, column}));
      }
      // Each lane retains the original increasing-k accumulation order.
      auto reduction = nb.create<scf::ForOp>(loc, constant(nb, 0), constant(nb, k),
          constant(nb, 1), accumulators,
          [&](OpBuilder &kb, Location location, Value ki, ValueRange current) {
            Value packet = kb.create<vector::LoadOp>(location, halfVector, packed,
                ValueRange{block, ki, constant(kb, 0)});
            Value wide = kb.create<arith::ExtFOp>(location, floatVector, packet);
            SmallVector<Value> next;
            for (unsigned r = 0; r < rows; ++r) {
              Value scalar = kb.create<memref::LoadOp>(location, a,
                                                     ValueRange{rowIndices[r], ki});
              Value extended = kb.create<arith::ExtFOp>(location,
                                                       kb.getF32Type(), scalar);
              Value broadcast = kb.create<vector::SplatOp>(location,
                                                           floatVector, extended);
              // A finite f16*f16 product is exactly representable in f32
              // (at most 22 significand bits, exponents within f32 range).
              // FMA therefore retains the same per-k rounded f32 sum while
              // allowing RVV's widening multiply-accumulate instruction.
              next.push_back(kb.create<vector::FMAOp>(location, broadcast, wide,
                                                      current[r]));
            }
            kb.create<scf::YieldOp>(location, next);
          });
      for (unsigned r = 0; r < rows; ++r)
        nb.create<vector::StoreOp>(loc, reduction.getResult(r), c,
                                  ValueRange{rowIndices[r], column});
    });
  });
  // The common ownership-based buffer-deallocation pipeline frees the packed
  // workspace, together with the other function-local allocations.
  op.erase();
}

struct LowerMatmulRowsToVectorsPass
    : intent::impl::LowerMatmulRowsToVectorsBase<LowerMatmulRowsToVectorsPass> {
  using LowerMatmulRowsToVectorsBase::LowerMatmulRowsToVectorsBase;
  void runOnOperation() override {
    if (rowBlock < 1 || rowBlock > 8 || vectorWidth < 1 || vectorWidth > 16) {
      getOperation().emitError("row block must be 1..8 and vector width 1..16");
      return signalPassFailure();
    }
    SmallVector<linalg::GenericOp> matched;
    getOperation().walk([&](linalg::GenericOp op) {
      if (matches(op, rowBlock, vectorWidth)) matched.push_back(op);
    });
    if (requireLowering && matched.empty()) {
      getOperation().emitError("no supported row-dot matrix contraction");
      return signalPassFailure();
    }
    for (auto op : matched) lower(op, rowBlock, vectorWidth);
    getOperation()->setAttr("intent.rvv.row_blocked_matmul_count",
        IntegerAttr::get(IntegerType::get(getOperation().getContext(), 64),
                         matched.size()));
  }
};
} // namespace
