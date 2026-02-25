// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/Im2colUtils.h"

#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

/// Helper to check if a slice will be contiguous given the offset and
/// slice size. Checks that `inputSize` and `offset` are both evenly
/// divisible by `tileSize`.
static bool willBeContiguousSlice(OpFoldResult inputSize, OpFoldResult tileSize,
                                  OpFoldResult offset) {
  std::optional<int64_t> constInputSize = getConstantIntValue(inputSize);
  std::optional<int64_t> constTileSize = getConstantIntValue(tileSize);
  if (!constTileSize.has_value() || !constInputSize.has_value() ||
      constInputSize.value() % constTileSize.value() != 0) {
    return false;
  }
  std::optional<int64_t> constOffset = getConstantIntValue(offset);
  if (constOffset.has_value() &&
      constOffset.value() % constTileSize.value() == 0) {
    return true;
  }
  auto affineOp = cast<Value>(offset).getDefiningOp<affine::AffineApplyOp>();
  return affineOp &&
         affineOp.getMap().getResult(0).isMultipleOf(constTileSize.value());
}

std::optional<int64_t>
chooseDimToVectorize(OpBuilder &b, Location loc, Im2colOp im2colOp,
                     SmallVector<Range> iterationDomain,
                     SmallVector<OpFoldResult> inputSizes,
                     OpFoldResult kOffset) {
  int64_t innerInputDim = im2colOp.getInputRank() - 1;
  SmallVector<SmallVector<int64_t>> vectorizationMap =
      im2colOp.getInputToOutputDimVectorizationMap();
  SmallVector<int64_t> vectorizableOutputDims = vectorizationMap[innerInputDim];
  if (vectorizableOutputDims.empty()) {
    return std::nullopt;
  }
  SetVector<int64_t> kDimSet(llvm::from_range, im2colOp.getKOutputDims());
  // There may be multiple output dims that we can vectorize, so prioritize the
  // innermost dims first.
  llvm::sort(vectorizableOutputDims);
  // Check each dim in order from innermost to outermost, and return the first
  // one that is vectorizable.
  while (!vectorizableOutputDims.empty()) {
    int64_t outputDimToVectorize = vectorizableOutputDims.pop_back_val();
    // If a K dim is being vectorized, then it is contiguous along either the
    // input channel dimension, or the filter kernel window. If it is contiguous
    // along the kernel window, then the actual inner slice size is equal to the
    // size of the corresponding kernel window dimension. Otherwise, the inner
    // slice size is just the size of the input tensor's inner dimension.
    OpFoldResult innerSliceSize = inputSizes[innerInputDim];
    if (kDimSet.contains(outputDimToVectorize)) {
      for (auto [kernelSize, mPos] :
           llvm::zip_equal(im2colOp.getMixedKernelSize(),
                           im2colOp.getMPos())) {
        if (mPos == innerInputDim) {
          innerSliceSize = kernelSize;
        }
      }
    }

    // If the input slice is contiguous along the innermost dimension, then it
    // is vectorizable. If it is not, then move on to the next innermost dim.
    SetVector<int64_t> mDimSet(llvm::from_range, im2colOp.getMOutputDims());
    OpFoldResult offset = b.getIndexAttr(0);
    if (kDimSet.contains(outputDimToVectorize)) {
      offset = kOffset;
    } else if (mDimSet.contains(outputDimToVectorize)) {
      // TODO(Max191): Support vectorization along the M dimension.
      continue;
    }
    OpFoldResult outputDimSize = iterationDomain[outputDimToVectorize].size;
    if (!willBeContiguousSlice(innerSliceSize, outputDimSize, offset)) {
      continue;
    }
    return outputDimToVectorize;
  }
  return std::nullopt;
}

OpFoldResult linearizeIm2colIndex(OpBuilder &b, Location loc,
                                  ArrayRef<OpFoldResult> inds,
                                  ArrayRef<OpFoldResult> basis) {
  MLIRContext *ctx = b.getContext();
  SmallVector<AffineExpr> dims(inds.size()), symbols(basis.size());
  bindDimsList<AffineExpr>(ctx, dims);
  bindSymbolsList<AffineExpr>(ctx, symbols);
  AffineExpr linearExpr = mlir::linearize(ctx, dims, symbols);
  SmallVector<OpFoldResult> mapOperands(inds);
  mapOperands.append(basis.begin(), basis.end());
  auto linearMap = AffineMap::get(
      /*dimCount=*/inds.size(), /*symbolCount=*/basis.size(), linearExpr);
  return affine::makeComposedFoldedAffineApply(b, loc, linearMap, mapOperands);
}

Im2colMBasisResult
computeIm2colMBasis(OpBuilder &b, Location loc, Im2colOp im2colOp,
                    ArrayRef<OpFoldResult> inputSizes) {
  SmallVector<OpFoldResult> mStrides = im2colOp.getMixedMStrides();
  int64_t nMOutputDims = static_cast<int64_t>(mStrides.size());
  int64_t nMPosDims = static_cast<int64_t>(im2colOp.getMPos().size());

  if (nMOutputDims == nMPosDims) {
    // EXPANDED M: There is one M output dimension per input spatial dimension.
    // The m_strides attribute encodes the stride of each M output dimension in
    // the linearized M space (e.g., [OW, 1] for 2D M with OH rows of OW each).
    //
    // The inner delinearization bases are m_strides[0..N-2] (all but the last
    // stride which is always 1). For example, m_strides = [OW, 1] → basis =
    // [OW], and delinearize(M, [OW], hasOuterBound=false) → (M/OW, M%OW).
    //
    // This approach is correct for:
    //   - Forward convolution (m_strides[0] = OW = the full output width).
    //   - Backward-weight convolution with dilation > 1 (m_strides[0] = KW,
    //     the kernel width — not a formula-derived approximation that would
    //     be wrong here).
    //   - Tiled im2col ops: m_strides are UNCHANGED by tiling, whereas the
    //     output tensor shape shrinks to tile sizes (which is why reading sizes
    //     directly from the output tensor would give wrong results after tiling).
    return Im2colMBasisResult{
        SmallVector<OpFoldResult>(mStrides.begin(), mStrides.end() - 1),
        /*hasOuterBound=*/false};
  }

  // FLAT M (rank-reduced): The M output has fewer dimensions than the number
  // of input M-position (spatial) dimensions. This occurs when multiple spatial
  // M dimensions are merged into a single output M dimension. In this case the
  // m_strides attribute is [1] and does not encode the inner spatial sizes.
  //
  // Fall back to computing the output spatial sizes from the input tensor using
  // the convolution output size formula. This is correct for forward convolution
  // (the primary use case for flat M). Use hasOuterBound=true since we compute
  // all N spatial dimension sizes.
  SmallVector<OpFoldResult> mBasis;
  ArrayRef<int64_t> strides = im2colOp.getStrides();
  ArrayRef<int64_t> dilations = im2colOp.getDilations();
  SmallVector<OpFoldResult> kernelSize = im2colOp.getMixedKernelSize();
  for (auto [idx, pos] : llvm::enumerate(im2colOp.getMPos())) {
    AffineExpr x, k;
    bindDims(im2colOp.getContext(), x, k);
    AffineExpr mapExpr =
        (x - 1 - (k - 1) * dilations[idx]).floorDiv(strides[idx]) + 1;
    OpFoldResult size = affine::makeComposedFoldedAffineApply(
        b, loc, AffineMap::get(2, 0, {mapExpr}, im2colOp.getContext()),
        {inputSizes[pos], kernelSize[idx]});
    mBasis.push_back(size);
  }
  return Im2colMBasisResult{mBasis, /*hasOuterBound=*/true};
}

SmallVector<OpFoldResult>
computeIm2colKBasis(Im2colOp im2colOp,
                    ArrayRef<OpFoldResult> inputSizes) {
  SmallVector<OpFoldResult> kBasis;
  SetVector<int64_t> mPosSet(im2colOp.getMPos().begin(),
                             im2colOp.getMPos().end());
  SetVector<int64_t> batchPosSet(im2colOp.getBatchPos().begin(),
                                 im2colOp.getBatchPos().end());
  SmallVector<OpFoldResult> kernelSize = im2colOp.getMixedKernelSize();
  SmallVector<int64_t> mKernelIdx(im2colOp.getInputRank(), -1);
  for (auto [idx, mPos] : enumerate(im2colOp.getMPos())) {
    mKernelIdx[mPos] = idx;
  }
  for (auto [idx, size] : enumerate(inputSizes)) {
    if (batchPosSet.contains(idx)) {
      continue;
    }
    if (mPosSet.contains(idx)) {
      kBasis.push_back(kernelSize[mKernelIdx[idx]]);
      continue;
    }
    kBasis.push_back(size);
  }
  // Transpose according to input_k_perm.
  ArrayRef<int64_t> inputKPerm = im2colOp.getInputKPerm();
  applyPermutationToVector(kBasis, inputKPerm);
  return kBasis;
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
