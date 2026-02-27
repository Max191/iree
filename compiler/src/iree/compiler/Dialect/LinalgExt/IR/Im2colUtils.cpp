// Copyright 2026 The IREE Authors
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
  if (constOffset.has_value()) {
    return constOffset.value() % constTileSize.value() == 0;
  }
  auto val = cast<Value>(offset);
  auto affineOp = val.getDefiningOp<affine::AffineApplyOp>();
  return affineOp &&
         affineOp.getMap().getResult(0).isMultipleOf(constTileSize.value());
}

std::optional<int64_t>
chooseDimToVectorize(OpBuilder &b, Location loc, Im2colOp im2colOp,
                     ArrayRef<Range> iterationDomain,
                     ArrayRef<OpFoldResult> inputSizes,
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

OpFoldResult linearizeIm2colKOffsets(OpBuilder &b, Location loc,
                                     Im2colOp im2colOp) {
  SmallVector<OpFoldResult> mixedOffsets = im2colOp.getMixedOffsets();
  SmallVector<SmallVector<OpFoldResult>> mixedOutputSizes =
      im2colOp.getMixedOutputSizes();
  int64_t batchSize = im2colOp.getBatchPos().size();
  int64_t numMOutputDims = im2colOp.getNumMOutputDims();
  int64_t kBegin = batchSize + numMOutputDims;
  int64_t kEnd = im2colOp.getOutputRank();

  if (kBegin == kEnd) {
    return b.getIndexAttr(0);
  }
  if (kEnd - kBegin == 1) {
    return mixedOffsets[kBegin];
  }

  // Compute per-K-output-dim extents.
  SmallVector<OpFoldResult> dimExtents;
  for (int64_t i = kBegin; i < kEnd; ++i) {
    OpFoldResult extent = b.getIndexAttr(1);
    for (auto s : mixedOutputSizes[i]) {
      extent = mulOfrs(b, loc, extent, s);
    }
    dimExtents.push_back(extent);
  }
  // Build strides from per-dim extents (suffix product).
  SmallVector<OpFoldResult> dimStrides(dimExtents.size());
  dimStrides.back() = b.getIndexAttr(1);
  for (int64_t i = dimExtents.size() - 2; i >= 0; --i) {
    dimStrides[i] = mulOfrs(b, loc, dimStrides[i + 1], dimExtents[i + 1]);
  }
  // Linear offset = sum(offset[i] * dimStrides[i]).
  OpFoldResult result = b.getIndexAttr(0);
  for (int64_t i = 0; i < static_cast<int64_t>(dimExtents.size()); ++i) {
    OpFoldResult term =
        mulOfrs(b, loc, mixedOffsets[kBegin + i], dimStrides[i]);
    result = addOfrs(b, loc, result, term);
  }
  return result;
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
