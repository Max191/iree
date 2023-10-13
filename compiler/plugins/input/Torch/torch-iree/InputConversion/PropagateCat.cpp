// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <numeric>

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-iree/InputConversion/PassDetail.h"
#include "torch-iree/InputConversion/Passes.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

namespace mlir {
namespace iree_compiler {
namespace TorchInput {

// Takes a dim and a tensor, and returns true if the dim is the outermost
// dimension, ignoring all unit dims
static bool isOuterDimension(Value tensor, Value dim) {
  auto dimOp = dim.getDefiningOp<torch::Torch::ConstantIntOp>();
  if (!dimOp) {
    return false;
  }
  auto tensorType =
      llvm::dyn_cast<torch::Torch::ValueTensorType>(tensor.getType());
  if (!tensorType) {
    return false;
  }
  auto tensorShape = tensorType.getSizes();
  int64_t dimVal = dimOp.getValueAttr().getInt();
  for (int i = 0; i < dimVal; i++) {
    if (tensorShape[i] != 1) {
      return false;
    }
  }
  return true;
}

static torch::Torch::ValueTensorType
getTransposedVTensorType(PatternRewriter &rewriter,
                         torch::Torch::ValueTensorType inputType, int64_t tDim0,
                         int64_t tDim1) {
  SmallVector<int64_t> newShape(inputType.getSizes());
  int64_t tDimSize = newShape[tDim0];
  newShape[tDim0] = newShape[tDim1];
  newShape[tDim1] = tDimSize;
  return torch::Torch::ValueTensorType::get(rewriter.getContext(), newShape,
                                            inputType.getOptionalDtype());
}

namespace {
class PropagateAtenCatOp : public OpRewritePattern<torch::Torch::AtenCatOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(torch::Torch::AtenCatOp catOp,
                                PatternRewriter &rewriter) const override {
    // Inputs are torch.list<vtensor>
    Value tensorList = catOp.getTensors();
    SmallVector<Value> tensors;
    if (!torch::Torch::getListConstructElements(tensorList, tensors)) {
      return catOp.emitError(
          "unimplemented: the tensor list is not from list construct");
    }
    auto catResultType = llvm::dyn_cast<torch::Torch::ValueTensorType>(
        catOp.getResult().getType());
    if (!catResultType) {
      return failure();
    }

    Value catDim = catOp.getDim();
    int64_t catDimVal;
    auto dimConstantOp = catDim.getDefiningOp<torch::Torch::ConstantIntOp>();
    if (!dimConstantOp) {
      return failure();
    }
    catDimVal = dimConstantOp.getValueAttr().getInt();
    Value transposeDim;
    int64_t transposeDimVal;

    // torch.aten.transpose.int -> torch.prim.ListConstruct -> torch.aten.cat
    SmallVector<std::optional<torch::Torch::AtenTransposeIntOp>>
        candidateTransposes(tensors.size(), std::nullopt);
    for (auto [i, tensor] : llvm::enumerate(tensors)) {
      if (isOuterDimension(tensor, catDim)) {
        return failure();
      }
      auto maybeTransposeOp =
          tensor.getDefiningOp<torch::Torch::AtenTransposeIntOp>();
      if (maybeTransposeOp) {
        auto transposeOp =
            llvm::cast<torch::Torch::AtenTransposeIntOp>(*maybeTransposeOp);
        Value tDim0 = transposeOp.getDim0();
        Value tDim1 = transposeOp.getDim1();
        if (isOuterDimension(transposeOp.getSelf(), tDim0) ||
            isOuterDimension(transposeOp.getSelf(), tDim1)) {
          auto tDimOp0 = tDim0.getDefiningOp<torch::Torch::ConstantIntOp>();
          auto tDimOp1 = tDim1.getDefiningOp<torch::Torch::ConstantIntOp>();
          if (tDimOp0 && tDimOp1) {
            int64_t tDim0Val = tDimOp0.getValueAttr().getInt();
            int64_t tDim1Val = tDimOp1.getValueAttr().getInt();
            if (tDim0Val == catDimVal) {
              if (!transposeDim || tDim1Val == transposeDimVal) {
                candidateTransposes[i] = transposeOp;
                transposeDim = tDim1;
                transposeDimVal = tDim1Val;
              }
            } else if (tDim1Val == catDimVal) {
              if (!transposeDim || tDim0Val == transposeDimVal) {
                candidateTransposes[i] = transposeOp;
                transposeDim = tDim0;
                transposeDimVal = tDim0Val;
              }
            }
          }
        }
      }
    }

    if (llvm::all_of(candidateTransposes,
                     [](auto op) -> bool { return op == std::nullopt; })) {
      return failure();
    }

    SmallVector<Value> newCatInputs;
    for (auto [i, transposeOp] : llvm::enumerate(candidateTransposes)) {
      if (transposeOp) {
        newCatInputs.push_back(transposeOp->getSelf());
      } else {
        Value tensor = tensors[i];
        auto tensorType =
            llvm::dyn_cast<torch::Torch::ValueTensorType>(tensor.getType());
        if (!tensorType) {
          return failure();
        }
        torch::Torch::ValueTensorType inputTransposeType =
            getTransposedVTensorType(rewriter, tensorType, catDimVal,
                                     transposeDimVal);
        Value inputTranspose =
            rewriter.create<torch::Torch::AtenTransposeIntOp>(
                catOp.getLoc(), inputTransposeType, tensor, catDim,
                transposeDim);
        newCatInputs.push_back(inputTranspose);
      }
    }

    torch::Torch::ValueTensorType newCatResultType = getTransposedVTensorType(
        rewriter, catResultType, catDimVal, transposeDimVal);
    Value newCatInputsList = rewriter.create<torch::Torch::PrimListConstructOp>(
        catOp.getLoc(), tensorList.getType(), newCatInputs);
    Value newCatOp = rewriter.create<torch::Torch::AtenCatOp>(
        catOp.getLoc(), newCatResultType, newCatInputsList, transposeDim);
    rewriter.replaceOpWithNewOp<torch::Torch::AtenTransposeIntOp>(
        catOp, catResultType, newCatOp, catDim, transposeDim);
    return success();
  }
};
} // namespace

namespace {

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct PropagateCatPass : public PropagateCatBase<PropagateCatPass> {

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<PropagateAtenCatOp>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createPropagateCatPass() {
  return std::make_unique<PropagateCatPass>();
}

} // namespace TorchInput
} // namespace iree_compiler
} // namespace mlir
