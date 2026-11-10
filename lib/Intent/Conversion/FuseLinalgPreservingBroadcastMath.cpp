// Preserve lower-domain math when fusing elementwise tensor operations.
#include "Intent/Conversion/Common/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace intent {
#define GEN_PASS_DEF_FUSELINALGPRESERVINGBROADCASTMATH
#include "Intent/Conversion/Common/Passes.h.inc"
} // namespace intent

using namespace mlir;

namespace {
struct FuseLinalgPreservingBroadcastMathPass
    : intent::impl::FuseLinalgPreservingBroadcastMathBase<
          FuseLinalgPreservingBroadcastMathPass> {
  using FuseLinalgPreservingBroadcastMathBase::FuseLinalgPreservingBroadcastMathBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    linalg::populateElementwiseOpsFusionPatterns(
        patterns, [](OpOperand *operand) {
          auto producer = operand->get().getDefiningOp<linalg::GenericOp>();
          auto consumer = dyn_cast<linalg::GenericOp>(operand->getOwner());
          if (!producer || !consumer)
            return true;
          bool expensive = false;
          producer.getRegion().walk([&](Operation *op) {
            expensive |= isa<arith::DivFOp>(op) ||
                         op->getName().getDialectNamespace() == "math";
          });
          if (!expensive)
            return true;
          // Projected/broadcast input maps introduce consumer iterations in
          // which this producer value is invariant. Keep the value computed
          // once in its existing domain instead of cloning sqrt/div/exp work.
          AffineMap map = consumer.getMatchingIndexingMap(operand);
          llvm::SmallBitVector used(map.getNumDims());
          for (AffineExpr expression : map.getResults())
            expression.walk([&](AffineExpr subexpression) {
              if (auto dimension = dyn_cast<AffineDimExpr>(subexpression))
                used.set(dimension.getPosition());
            });
          return used.all();
        });
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
