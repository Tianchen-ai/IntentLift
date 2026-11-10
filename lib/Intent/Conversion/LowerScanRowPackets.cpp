// Interleave independent scan rows without changing any row's addition order.
#include "Intent/Conversion/Common/Passes.h"
#include "Intent/IntentOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

namespace intent {
#define GEN_PASS_DEF_LOWERSCANROWPACKETS
#include "Intent/Conversion/Common/Passes.h.inc"
}
using namespace mlir;

namespace {
struct LowerScanRowPacketsPass
    : intent::impl::LowerScanRowPacketsBase<LowerScanRowPacketsPass> {
  using LowerScanRowPacketsBase::LowerScanRowPacketsBase;
  void runOnOperation() override {
    if (packetRows < 1 || packetRows > 32) {
      getOperation().emitError("scan row packet must be between 1 and 32");
      signalPassFailure();
      return;
    }
    SmallVector<intent::ScanOp> scans;
    getOperation().walk([&](intent::ScanOp op) { scans.push_back(op); });
    IRRewriter rewriter(&getContext());
    unsigned lowered = 0;
    for (intent::ScanOp op : scans) {
      auto type = dyn_cast<RankedTensorType>(op.getInput().getType());
      if (!type || !type.hasStaticShape() || type.getRank() != 2 ||
          !type.getElementType().isF32() || op.getAxis() != 1 ||
          op.getFn() != "sum" || type != op.getResult().getType())
        continue;
      int64_t pack = std::min<int64_t>(packetRows, std::max<int64_t>(1, type.getDimSize(0)));
      while (pack > 1 && type.getDimSize(0) % pack)
        --pack;
      rewriter.setInsertionPoint(op);
      Location loc = op.getLoc();
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      Value step = rewriter.create<arith::ConstantIndexOp>(loc, pack);
      Value rows = rewriter.create<arith::ConstantIndexOp>(loc, type.getDimSize(0));
      Value cols = rewriter.create<arith::ConstantIndexOp>(loc, type.getDimSize(1));
      Value empty = rewriter.create<tensor::EmptyOp>(loc, type.getShape(), type.getElementType());
      auto outer = rewriter.create<scf::ForOp>(
          loc, zero, rows, step, ValueRange{empty},
          [&](OpBuilder &ob, Location ol, Value row, ValueRange oa) {
            Value initial = ob.create<arith::ConstantOp>(ol, ob.getF32FloatAttr(0.0));
            SmallVector<Value> args(pack, initial);
            args.push_back(oa[0]);
            SmallVector<Value> rowIndices;
            for (int64_t lane = 0; lane < pack; ++lane) {
              Value offset = ob.create<arith::ConstantIndexOp>(ol, lane);
              rowIndices.push_back(ob.create<arith::AddIOp>(ol, row, offset));
            }
            auto inner = ob.create<scf::ForOp>(
                ol, zero, cols, one, args,
                [&](OpBuilder &ib, Location il, Value col, ValueRange ia) {
                  Value first = ib.create<arith::CmpIOp>(il, arith::CmpIPredicate::eq, col, zero);
                  Value output = ia[pack];
                  SmallVector<Value> yields;
                  for (int64_t lane = 0; lane < pack; ++lane) {
                    Value current = ib.create<tensor::ExtractOp>(il, op.getInput(), ValueRange{rowIndices[lane], col});
                    Value sum = ib.create<arith::AddFOp>(il, ia[lane], current);
                    Value inclusive = ib.create<arith::SelectOp>(il, first, current, sum);
                    output = ib.create<tensor::InsertOp>(il, inclusive, output, ValueRange{rowIndices[lane], col});
                    yields.push_back(inclusive);
                  }
                  yields.push_back(output);
                  ib.create<scf::YieldOp>(il, yields);
                });
            ob.create<scf::YieldOp>(ol, inner.getResult(pack));
          });
      rewriter.replaceOp(op, outer.getResult(0));
      ++lowered;
    }
    getOperation()->setAttr("intent.rvv.exact_scan_packet_count", rewriter.getI64IntegerAttr(lowered));
    getOperation()->setAttr("intent.rvv.exact_scan_packet_rows", rewriter.getI64IntegerAttr(packetRows));
    if (requireLowering && lowered == 0) {
      getOperation().emitError("no static rank-two f32 inclusive sum scan matched");
      signalPassFailure();
    }
  }
};
}
