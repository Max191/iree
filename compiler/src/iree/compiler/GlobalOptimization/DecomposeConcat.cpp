// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

namespace {

static Value createTranspose(OpBuilder &builder, Value source,
                             SmallVector<int64_t> perm) {
  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(builder, source.getLoc(), source);
  applyPermutationToVector(mixedSizes, perm);
  Type elemType = cast<RankedTensorType>(source.getType()).getElementType();
  Value empty =
      builder.create<tensor::EmptyOp>(source.getLoc(), mixedSizes, elemType)
          .getResult();
  return builder
      .create<linalg::TransposeOp>(source.getLoc(), source, empty, perm)
      ->getResult(0);
}

// Transposes the concatenation dimension to happen along the outer most
// non-unit dim of the inputs. The idea is that outer dim concatentations
// can lower to `flow.tensor.update` and ideally disappear, in the worst case
// becoming a sequence of copies. The hope then is that the transposes on the
// inputs and output is then fusable with surrounding operations.
struct TransposeInnerConcatenation : public OpRewritePattern<tensor::ConcatOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ConcatOp concatOp,
                                PatternRewriter &rewriter) const override {
    // Get the outer most non-unit dim to transpose to.
    RankedTensorType concatType = concatOp.getResultType();
    ArrayRef<int64_t> concatShape = concatType.getShape();
    int64_t outerMostNonUnitDim = 0;
    while (outerMostNonUnitDim < concatOp.getRank()) {
      if (concatShape[outerMostNonUnitDim] != 1)
        break;
      outerMostNonUnitDim++;
    }

    // Nothing to do if the concat is already the outer most non-unit
    int64_t dim = concatOp.getDim();
    if (dim <= outerMostNonUnitDim) {
      return failure();
    }

    SmallVector<int64_t> permutation = computePermutationVector(
        concatOp.getRank(), {dim}, {outerMostNonUnitDim});
    SmallVector<Value> transposedInputs;
    for (auto input : concatOp.getInputs()) {
      transposedInputs.push_back(createTranspose(rewriter, input, permutation));
    }

    SmallVector<int64_t> newShape = applyPermutation(concatShape, permutation);
    auto newConcatType = RankedTensorType::get(
        newShape, concatOp.getResultType().getElementType());
    Value newConcat = rewriter.create<tensor::ConcatOp>(
        concatOp.getLoc(), newConcatType, /*dim=*/outerMostNonUnitDim,
        transposedInputs);
    auto invPerm = invertPermutationVector(permutation);
    Value transposedConcat = createTranspose(rewriter, newConcat, invPerm);
    rewriter.replaceOp(concatOp, transposedConcat);
    return success();
  }
};

static LogicalResult insertFoldingPrecondition(tensor::InsertSliceOp insertOp) {
  auto definingOp = insertOp.getDest().getDefiningOp();
  if (isa<tensor::ExpandShapeOp, tensor::EmptyOp>(definingOp)) {
    return success();
  }
  return failure();
}

static void getNewInsertOperandMetadata(
    tensor::InsertSliceOp insertOp,
    std::optional<SmallVector<ReassociationIndices>> &collapsedSrcRe,
    std::optional<SmallVector<ReassociationIndices>> &collapsedDstRe,
    SmallVector<OpFoldResult> &collapsedOffsets,
    SmallVector<OpFoldResult> &collapsedSizes,
    SmallVector<OpFoldResult> &collapsedStrides) {
  auto dstShape = insertOp.getDestType().getShape();
  auto srcShape = insertOp.getSourceType().getShape();
  auto mixedOffsets = insertOp.getMixedOffsets();
  auto mixedSizes = insertOp.getMixedSizes();
  auto mixedStrides = insertOp.getMixedStrides();
  SmallVector<int64_t> collapsedDstShape, collapsedSrcShape;
  SmallVector<int> srcNonUnitDims, dstNonUnitDims;
  int rankDiff = dstShape.size() - srcShape.size();
  for (auto [dstIdx, dstSize] : enumerate(dstShape)) {
    if (dstSize != 1) {
      collapsedDstShape.push_back(dstSize);
      dstNonUnitDims.push_back(dstIdx);
      int srcIdx = dstIdx - rankDiff;
      if (srcIdx >= 0) {
        collapsedSrcShape.push_back(srcShape[srcIdx]);
        srcNonUnitDims.push_back(srcIdx);
      }
      collapsedOffsets.push_back(mixedOffsets[dstIdx]);
      collapsedSizes.push_back(mixedSizes[dstIdx]);
      collapsedStrides.push_back(mixedStrides[dstIdx]);
    }
  }
  auto getReassociationIndices = [](SmallVector<int> nonUnitDims,
                                    ArrayRef<int64_t> shape)
      -> std::optional<SmallVector<ReassociationIndices>> {
    SetVector<int> s(nonUnitDims.begin(), nonUnitDims.end());
    SmallVector<ReassociationIndices> reInd(nonUnitDims.size(),
                                            ReassociationIndices{});
    int reListIdx = 0;
    for (int idx = 0; idx < shape.size(); idx++) {
      reInd[reListIdx].push_back(idx);
      if (s.contains(idx)) {
        reListIdx++;
      }
    }
    if (llvm::all_of(reInd,
                     [](ReassociationIndices re) { return re.size() == 1; })) {
      return std::nullopt;
    }
    return reInd;
  };
  collapsedSrcRe = getReassociationIndices(srcNonUnitDims, srcShape);
  collapsedDstRe = getReassociationIndices(dstNonUnitDims, dstShape);
}

// Concatenations with unit dims will decompose into insert slices with
// Unit dims. This pattern rewrites the insert slices without unit dims
// so unit dim folding is not blocked by these ops later on.
struct FoldInsertSliceUnitDims
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    if (failed(insertFoldingPrecondition(insertOp))) {
      return failure();
    }

    Location loc = insertOp.getLoc();
    Value source = insertOp.getSource();
    Value dest = insertOp.getDest();
    std::optional<SmallVector<ReassociationIndices>> collapsedSrcRe,
        collapsedDstRe;
    SmallVector<OpFoldResult> collapsedOffsets, collapsedSizes,
        collapsedStrides;
    getNewInsertOperandMetadata(insertOp, collapsedSrcRe, collapsedDstRe,
                                collapsedOffsets, collapsedSizes,
                                collapsedStrides);
    if (!collapsedDstRe) {
      return success();
    }
    if (collapsedSrcRe) {
      source = rewriter.create<tensor::CollapseShapeOp>(loc, source,
                                                        collapsedSrcRe.value());
    }
    if (collapsedDstRe) {
      dest = rewriter.create<tensor::CollapseShapeOp>(loc, dest,
                                                      collapsedDstRe.value());
    }
    Value newInsert = rewriter.create<tensor::InsertSliceOp>(
        loc, source, dest, collapsedOffsets, collapsedSizes, collapsedStrides);
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        insertOp, insertOp.getDestType(), newInsert, collapsedDstRe.value());
    return success();
  }
};

struct DecomposeConcatPass : public DecomposeConcatBase<DecomposeConcatPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }
  DecomposeConcatPass(bool enableConcatTransposition) {
    this->enableConcatTransposition = enableConcatTransposition;
  }
  DecomposeConcatPass(const DecomposeConcatPass &pass)
      : DecomposeConcatPass(pass.enableConcatTransposition) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    if (enableConcatTransposition) {
      patterns.insert<TransposeInnerConcatenation>(context, /*benefit=*/2);
    }
    tensor::populateDecomposeTensorConcatPatterns(patterns);
    patterns.insert<FoldInsertSliceUnitDims>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass>
createDecomposeConcatPass(bool enableConcatTransposition) {
  return std::make_unique<DecomposeConcatPass>(enableConcatTransposition);
}

} // namespace mlir::iree_compiler::GlobalOptimization
