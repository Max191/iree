


#include "iree/compiler/Preprocessing/Common/PassDetail.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler{
namespace IREE{

namespace {

//-----------------------------------------------------------//
//                        Patterns
//-----------------------------------------------------------//

// struct DequantizationMatmulFusePass
//     : public DequantizationMatmulFuseBase<DequantizationMatmulFusePass> {

//   void getDependentDialects(DialectRegistry &registry) const override {
//     registry.insert<linalg::LinalgDialect>();
//   }

class DequantizationMatmulFusePattern final
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genOpMatmul,
                                PatternRewriter &rewriter) const override {        

    // Match first generic op as matmul

    if (genOpMatmul.getNumResults() != 1) return failure();
    if (genOpMatmul.getNumOperands() != 2) return failure();

    // auto mmIndexingMaps = genOpMatmul.getIndexingMaps();

    // auto mmResultType =
    //     llvm::cast<RankedTensorType>(genOpMatmul.getOutputs()[0].getType());
    // auto mmLhsType =
    //     llvm::cast<RankedTensorType>(genOpMatmul.getInputs()[0].getType());
    // auto mmResultShape = mmResultType.getShape();
    // auto mmLhsShape = mmLhsType.getShape();

    auto parallel = utils::IteratorType::parallel;
    auto reduction = utils::IteratorType::reduction;
    SmallVector<utils::IteratorType> mmIteratorTypes =
        genOpMatmul.getIteratorTypesArray();
    SmallVector<utils::IteratorType> mmExpectedIteratorTypes = {parallel, parallel,
                                                                parallel, reduction,
                                                                reduction};
    unsigned mmNLoops = mmIteratorTypes.size();
    unsigned mmExpectedNLoops = 5;

    if (mmNLoops != mmExpectedNLoops) return failure();

    for (int i = 0; i < mmNLoops; i++){
      if (mmIteratorTypes[i] != mmExpectedIteratorTypes[i]) return failure();
    }

    Value mmLhs = genOpMatmul->getOperand(1);
    auto mmLhsDefiningOp = mmLhs.getDefiningOp();
    linalg::GenericOp genOpDequant = dyn_cast<linalg::GenericOp>(mmLhsDefiningOp);
    if (!genOpDequant) return failure();

    // Match defining op as dequantization

    if (genOpDequant.getNumResults() != 1) return failure();
    if (genOpDequant.getNumOperands() != 3) return failure();

    SmallVector<utils::IteratorType> dqIteratorTypes =
        genOpDequant.getIteratorTypesArray();
    SmallVector<utils::IteratorType> dqExpectedIteratorTypes = {parallel, parallel,
                                                                parallel};
    
    unsigned dqNLoops = dqIteratorTypes.size();
    unsigned dqExpectedNLoops = 3;

    if (dqNLoops != dqExpectedNLoops) return failure();

    for (int i = 0; i < dqNLoops; i++){
      if (dqIteratorTypes[i] != dqExpectedIteratorTypes[i]) return failure();
    }

    // Form dispatch region with dequantization and matmul

    Flow::DispatchRegionOp regionOp;
    auto maybeRegionOp = Flow::wrapOpInDispatchRegion(rewriter, genOpMatmul);
    if (failed(maybeRegionOp))
      return failure();
    regionOp = *maybeRegionOp;

    auto combinedRegionOp = Flow::movePrecedingOpsIntoDispatchRegion(rewriter, mmLhsDefiningOp, regionOp);
    if (failed(combinedRegionOp))
      return failure();
    
    return success();

  }
};

struct DequantizationMatmulFusePass
    : public DequantizationMatmulFuseBase<DequantizationMatmulFusePass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
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