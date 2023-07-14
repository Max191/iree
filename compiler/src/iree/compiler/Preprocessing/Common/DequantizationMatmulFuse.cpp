


#include "iree/compiler/Preprocessing/Common/PassDetail.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler{
namespace IREE{

namespace {

//-----------------------------------------------------------//
//                        Patterns
//-----------------------------------------------------------//

class FuseDequantizationMatmul final
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genOpMatmul,
                                PatternRewriter &rewriter) const override {        

    // Match first generic op as matmul

    if (genOpMatmul.getNumResults() != 1) return failure();
    if (genOpMatmul.getNumOperands() != 2) return failure();

    auto mmIndexingMaps = genOpMatmul.getIndexingMaps();

    auto mmResultType =
        llvm::cast<RankedTensorType>(genOpMatmul.getOutputs()[0].getType());
    auto mmLhsType =
        llvm::cast<RankedTensorType>(genOpMatmul.getInputs()[0].getType());
    auto mmResultShape = mmResultType.getShape();
    auto mmLhsShape = mmLhsType.getShape();

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

    for (int i; i < mmNLoops; i++){
      if (mmIteratorTypes[i] != mmExpectedIteratorTypes[i]) return failure();
    }

    Value mmLhs = genOpMatmul->getOperand(1);
    auto mmLhsDefiningOp = result.getDefiningOp();

    if (mmLhsDefiningOp == NULL) return failure();

    // Match defining op as dequantization

    if (mmLhsDefiningOp.getNumResults() != 1) return failure();
    if (mmLhsDefiningOp.getNumOperands() != 3) return failure();

    SmallVector<utils::IteratorType> dqIteratorTypes =
        mmLhsDefiningOp.getIteratorTypesArray();
    SmallVector<utils::IteratorType> dqExpectedIteratorTypes = {parallel, parallel,
                                                                parallel};
    
    unsigned dqNLoops = dqIteratorTypes.size();
    unsigned dqExpectedNLoops = 5;

    for (int i; i < dqNLoops; i++){
      if (dqIteratorTypes[i] != dqExpectedIteratorTypes[i]) return failure();
    }

    

  }
}




} // namespace
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir