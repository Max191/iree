


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
#include <iostream>

namespace mlir {
namespace iree_compiler{
namespace IREE{

namespace {


void fuseDequantAndMatmul(RewriterBase &rewriter,
                          Operation* dequant,
                          Operation* matmul){
  Value mmResultValue = matmul->getResult(0);
  auto mmResultType =
      llvm::cast<RankedTensorType>(matmul->getResult(0).getType());

  SmallVector<Value> workload;
  // Flow::DispatchRegionOp Flow::makeEmptyDispatchRegion(rewriter, genOpMatmul.getLoc(), workload)
  rewriter.setInsertionPointAfter(matmul);

  OpBuilder::InsertionGuard guard(rewriter);

  // Create RegionOp.
  auto regionOp = rewriter.create<Flow::DispatchRegionOp>(
      matmul->getLoc(), /*resultTypes=*/mmResultType, /*dynamicDims=*/ValueRange(), workload);
  Block &body = regionOp.getBody().emplaceBlock();
  rewriter.setInsertionPointToStart(&body);
  rewriter.create<Flow::ReturnOp>(matmul->getLoc(), mmResultValue);
  
  // Values replaced by moving the `targets` into the dispatch region.
  Value replacedOutputValue;

  // New value that is yielded from dispatch.
  Value yieldedResult;
  
  Value regionOpResult = regionOp->getResult(0);
  // std::cout << "regionOp result: " << &regionOpResult << std::endl;
  // std::cout << "matmul result: " << &mmResultValue << std::endl;
  rewriter.replaceAllUsesWith(mmResultValue, regionOpResult);
  // std::cout << "regionOp use num: " << regionOp->hasOneUse() << std::endl;
  // std::cout << "matmul use num: " << matmul->hasOneUse() << std::endl;

  Block &body2 = regionOp.getBody().front();
  rewriter.setInsertionPointToStart(&body2);
  Operation *clonedMatmul = rewriter.clone(*matmul);
  // replacedOutputValue = matmul->getResult(0);
  yieldedResult = clonedMatmul->getResult(0);

  // rewriter.replaceAllUsesWith(replacedOutputValue, yieldedResult);

  auto regionReturnOp =
      cast<Flow::ReturnOp>(regionOp.getBody().front().getTerminator());
  rewriter.setInsertionPoint(regionReturnOp);
  rewriter.replaceOpWithNewOp<Flow::ReturnOp>(regionReturnOp,
                                              yieldedResult);

  // auto returnOp = regionOp.getBody().front().getTerminator();
  // rewriter.replaceOp(returnOp, yieldedResult);

  Block &body3 = regionOp.getBody().front();
  rewriter.setInsertionPointToStart(&body3);
  Operation *clonedDequant = rewriter.clone(*dequant);
  replacedOutputValue = dequant->getResult(0);
  yieldedResult = clonedDequant->getResult(0);

  rewriter.replaceAllUsesWith(replacedOutputValue, yieldedResult);

  rewriter.eraseOp(matmul);
  rewriter.eraseOp(dequant);

  // std::cout << "mm parent: " << clonedMatmul->getParentOp() << std::endl;
  // std::cout << "dq parent: " << clonedDequant->getParentOp() << std::endl;
  // std::cout << "mm isAncestor: " << regionOp->isAncestor(clonedMatmul) << std::endl;
  // std::cout << "dq isAncestor: " << regionOp->isAncestor(clonedDequant) << std::endl;

  return;
}


//-----------------------------------------------------------//
//                        Patterns
//-----------------------------------------------------------//

class DequantizationMatmulFusePattern final
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genOpMatmul,
                                PatternRewriter &rewriter) const override {        

    // Stop after op has been placed in a dispatch
    // std::cout << "Starting..." << std::endl;

    // Match first generic op as matmul

    if (genOpMatmul.getNumResults() != 1) return failure();
    if (genOpMatmul.getNumOperands() != 3) return failure();

    Value mmRes = genOpMatmul->getResult(0);
    auto matmulOp = mmRes.getDefiningOp();

    // std::cout << "checking parent: " << matmulOp->getParentOp() << std::endl;
    if (matmulOp->getParentOfType<Flow::DispatchRegionOp>()) return failure();
    // std::cout << "no parent: " << matmulOp->getParentOp() << std::endl;

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
    auto dequantOp = mmLhs.getDefiningOp();

    linalg::GenericOp genOpDequant = dyn_cast<linalg::GenericOp>(dequantOp);
    if (!genOpDequant) return failure();

    // Match defining op as dequantization

    if (genOpDequant.getNumResults() != 1) return failure();
    if (genOpDequant.getNumOperands() != 4) return failure();

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

    fuseDequantAndMatmul(rewriter, dequantOp, matmulOp);
    // std::cout << "regionOp: " << regionOp << std::endl;

    // Value mmResultValue = genOpMatmul.getResult(0);
    // SmallVector<Value> workload;
    // // Flow::DispatchRegionOp Flow::makeEmptyDispatchRegion(rewriter, genOpMatmul.getLoc(), workload)
    // rewriter.setInsertionPointAfter(genOpMatmul);

    // OpBuilder::InsertionGuard guard(rewriter);

    // // Create RegionOp.
    // auto regionOp = rewriter.create<Flow::DispatchRegionOp>(
    //     genOpMatmul.getLoc(), /*resultTypes=*/mmResultType, /*dynamicDims=*/ValueRange(), workload);
    // Block &body = regionOp.getBody().emplaceBlock();
    // rewriter.setInsertionPointToStart(&body);
    // rewriter.create<Flow::ReturnOp>(genOpMatmul.getLoc(), mmResultValue);
    
    // // Values replaced by moving the `targets` into the dispatch region.
    // // Value replacedOutputValue;

    // // New value that is yielded from dispatch.
    // // Value yieldedResult;

    // // Operation *clonedTarget
    // // Block &body
    
    // Block &body2 = regionOp.getBody().front();
    // rewriter.setInsertionPointToStart(&body2);
    // Operation *clonedMatmul = rewriter.clone(genOpMatmul);
    // // replacedOutputValue = genOpMatmul.getResult(0);
    // // yieldedResult = clonedTarget->getResult(0);
    // // auto returnOp =
    // //       cast<Flow::ReturnOp>(regionOp.getBody().front().getTerminator());
    // // rewriter.replaceOp(clonedTarget, mmResultValue);

    // Block &body3 = regionOp.getBody().front();
    // rewriter.setInsertionPointToStart(&body3);
    // Operation *clonedDequant = rewriter.clone(genOpDequant);
    // // replacedOutputValue = genOpDequant.getResult(0);
    // // yieldedResult = clonedTarget->getResult(0);

    // rewriter.eraseOp(genOpMatmul);
    // rewriter.eraseOp(genOpDequant);



    // SmallVector<Value> workload;
    // auto regionOp = rewriter.create<Flow::DispatchRegionOp>(
    //   genOpMatmul.getLoc(), /*resultTypes=*/TypeRange(), /*dynamicDims=*/ValueRange(), workload);
    // Flow::DispatchRegionOp regionOp;
    // auto maybeRegionOp = Flow::wrapOpInDispatchRegion(rewriter, genOpMatmul);
    // if (failed(maybeRegionOp))
    //   return failure();
    // regionOp = *maybeRegionOp;

    // auto combinedRegionOp = Flow::movePrecedingOpsIntoDispatchRegion(rewriter, mmLhsDefiningOp, regionOp);
    // if (failed(combinedRegionOp))
    //   return failure();

    // Block &body = combinedRegionOp->getBody().front();
    // auto terminator = body.getTerminator();
    // auto terminatorLoc = terminator->getLoc();
    // rewriter.setInsertionPointAfter(combinedRegionOp);
    // rewriter.create<Flow::ReturnOp>();
    // std::cout << "mm parent: " << clonedMatmul->getParentOp() << std::endl;
    // std::cout << "dq parent: " << clonedDequant->getParentOp() << std::endl;
    // std::cout << "mm isAncestor: " << regionOp->isAncestor(clonedMatmul) << std::endl;
    // std::cout << "dq isAncestor: " << regionOp->isAncestor(clonedDequant) << std::endl;
    return success();

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