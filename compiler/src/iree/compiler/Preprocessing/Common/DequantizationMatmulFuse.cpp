

#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Preprocessing/Common/PassDetail.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {

namespace {

//-----------------------------------------------------------//
//                        Utility
//-----------------------------------------------------------//

static LogicalResult fuseDequantAndMatmul(RewriterBase &rewriter,
                                          Operation *dequant,
                                          Operation *matmul) {
  
  Flow::DispatchRegionOp regionOp =
      matmul->getParentOfType<Flow::DispatchRegionOp>();
  if (!regionOp) {
    FailureOr<Flow::DispatchRegionOp> maybeRegionOp =
        Flow::wrapOpInDispatchRegion(rewriter, matmul);
    if (failed(maybeRegionOp))
      return failure();
    regionOp = maybeRegionOp.value();
  }

  FailureOr<Flow::DispatchRegionOp> maybeFusedRegionOp =
      movePrecedingOpsIntoDispatchRegion(rewriter, dequant, regionOp);
  if (failed(maybeFusedRegionOp))
    return failure();

  return success();
}

static LogicalResult isGroupedContractionOp(linalg::GenericOp genericOp) {
  unsigned numLoops = genericOp.getNumLoops();
  linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(genericOp.getOperation());
  if (numLoops == 0)
    return failure();
  if (!linalg::isaContractionOpInterface(linalgOp))
    return failure();
  if (genericOp.getNumReductionLoops() != 2)
    return failure();
  return success();
}

static LogicalResult isGroupedDequantizationOp(linalg::GenericOp genericOp) {
  // Check for 1 result, and 2 (input, scales) or 3 (input, scales, zero points) inputs
  if (genericOp.getNumDpsInits() != 1)
    return failure();
  if (genericOp.getNumDpsInputs() != 2 && genericOp.getNumDpsInputs() != 3)
    return failure();

  // Check that the rank is at least 3 and all loops are parallel
  unsigned numLoops = genericOp.getNumLoops();
  unsigned numParallelLoops = genericOp.getNumParallelLoops();
  if (numLoops < 3)
    return failure();
  if (numLoops != numParallelLoops)
    return failure();

  return success();
}

//-----------------------------------------------------------//
//                        Patterns
//-----------------------------------------------------------//

class DequantizationMatmulFusePattern final
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    // Match first generic op as matmul
    if (failed(isGroupedContractionOp(genericOp)))
      return failure();

    // Fail if matmul has already been fused
    Value genericOpResult = genericOp->getResult(0);
    Operation *matmulOp = genericOpResult.getDefiningOp();
    if (matmulOp->getParentOfType<Flow::DispatchRegionOp>())
      return failure();

    // Match operands to dequantizations and fuse if matched
    Value lhs = genericOp->getOperand(0);
    Value rhs = genericOp->getOperand(1);
    auto lhsOp = lhs.getDefiningOp<linalg::GenericOp>();
    auto rhsOp = rhs.getDefiningOp<linalg::GenericOp>();

    LogicalResult succeeded = failure();
    if (lhsOp && !failed(isGroupedDequantizationOp(
                     llvm::dyn_cast<linalg::GenericOp>(*lhsOp)))) {
      if (!failed(fuseDequantAndMatmul(rewriter, lhsOp, matmulOp)))
        succeeded = success();
    }

    if (rhsOp && !failed(isGroupedDequantizationOp(
                     llvm::dyn_cast<linalg::GenericOp>(*rhsOp)))) {
      if (!failed(fuseDequantAndMatmul(rewriter, rhsOp, matmulOp)))
        succeeded = success();
    }

    return succeeded;
  }
};

struct DequantizationMatmulFusePass
    : public DequantizationMatmulFuseBase<DequantizationMatmulFusePass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, Flow::FlowDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    // Main pattern.
    {
      RewritePatternSet patterns(&getContext());
      patterns.insert<DequantizationMatmulFusePattern>(context);
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createDequantizationMatmulFusePass() {
  return std::make_unique<DequantizationMatmulFusePass>();
}

} // namespace IREE
} // namespace iree_compiler
} // namespace mlir