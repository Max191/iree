// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/Im2colUtils.h"

#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

Im2colSourceIndices computeIm2colSourceIndices(OpBuilder &b, Location loc,
                                                Im2colOp im2colOp,
                                                ArrayRef<Value> ivs,
                                                OpFoldResult innerTileSize) {
  int64_t inputRank = im2colOp.getInputRank();
  SmallVector<OpFoldResult> offsets = im2colOp.getMixedOffsets();
  SmallVector<SmallVector<OpFoldResult>> outputSizes =
      im2colOp.getMixedOutputSizes();
  int64_t batchSize = im2colOp.getBatchPos().size();
  int64_t numMOutputDims = im2colOp.getNumMOutputDims();
  llvm::SmallDenseSet<int64_t, 4> mPosSet(im2colOp.getMPos().begin(),
                                          im2colOp.getMPos().end());
  llvm::SmallDenseSet<int64_t, 4> batchPosSet(im2colOp.getBatchPos().begin(),
                                               im2colOp.getBatchPos().end());
  ArrayRef<int64_t> strides = im2colOp.getStrides();
  ArrayRef<int64_t> dilations = im2colOp.getDilations();
  ArrayRef<int64_t> inputKPerm = im2colOp.getInputKPerm();

  // For each K output dim: delinearize (offset[d] + iv[d]) using
  // output_sizes[d]. Concatenate results -> combined window+channel coords.
  SmallVector<Value> kCoords;
  SmallVector<int64_t> kOutputDims = im2colOp.getKOutputDims();
  int64_t numKOutputDims =
      im2colOp.getOutputRank() - batchSize - numMOutputDims;
  for (int64_t i = 0; i < numKOutputDims; ++i) {
    int64_t canonicalIdx = batchSize + numMOutputDims + i;
    int64_t actualDim = kOutputDims[i];
    OpFoldResult idx = addOfrs(b, loc, offsets[canonicalIdx], ivs[actualDim]);
    const SmallVector<OpFoldResult> &innerSizes = outputSizes[canonicalIdx];
    if (innerSizes.size() == 1) {
      kCoords.push_back(getValueOrCreateConstantIndexOp(b, loc, idx));
    } else {
      // Use non-wrapping delinearize (hasOuterBound=false) so that
      // positions beyond the product of output_sizes produce out-of-bounds
      // coordinates instead of wrapping. This lets the bounds computation
      // naturally handle oversized outputs (e.g. from GEMM alignment).
      SmallVector<OpFoldResult> innerBasis(innerSizes.begin() + 1,
                                           innerSizes.end());
      ValueRange delinCoords =
          affine::AffineDelinearizeIndexOp::create(
              b, loc, getValueOrCreateConstantIndexOp(b, loc, idx),
              innerBasis, /*hasOuterBound=*/false)
              .getResults();
      kCoords.append(delinCoords.begin(), delinCoords.end());
    }
  }

  // Apply input_k_perm to kCoords, then split into window offsets and
  // channel offsets. The inverse permutation maps from output K order to
  // the canonical input order (m_pos spatial + k_pos channel).
  SmallVector<int64_t> invInputKPerm = invertPermutationVector(inputKPerm);
  // applyPermutationToVector performs a gather (result[i] = src[perm[i]]),
  // which is the correct inverse mapping from output K order to input order.
  applyPermutationToVector(kCoords, invInputKPerm);
  SmallVector<Value> windowOffset, inputKOffset;
  int64_t kIdx = 0;
  for (int64_t i = 0; i < inputRank; ++i) {
    if (batchPosSet.contains(i)) {
      continue;
    }
    if (mPosSet.contains(i)) {
      windowOffset.push_back(kCoords[kIdx++]);
      continue;
    }
    inputKOffset.push_back(kCoords[kIdx++]);
  }

  // For each M output dim: delinearize (offset[d] + iv[d]) using
  // output_sizes[d]. Concatenate results -> spatial coordinates.
  SmallVector<Value> mCoords;
  SmallVector<int64_t> mOutputDims = im2colOp.getMOutputDims();
  for (int64_t i = 0; i < numMOutputDims; ++i) {
    int64_t canonicalIdx = batchSize + i;
    int64_t actualDim = mOutputDims[i];
    OpFoldResult idx = addOfrs(b, loc, offsets[canonicalIdx], ivs[actualDim]);
    const SmallVector<OpFoldResult> &innerSizes = outputSizes[canonicalIdx];
    if (innerSizes.size() == 1) {
      mCoords.push_back(getValueOrCreateConstantIndexOp(b, loc, idx));
    } else {
      // Non-wrapping delinearize: same rationale as the K dim case above.
      SmallVector<OpFoldResult> innerBasis(innerSizes.begin() + 1,
                                           innerSizes.end());
      ValueRange delinCoords =
          affine::AffineDelinearizeIndexOp::create(
              b, loc, getValueOrCreateConstantIndexOp(b, loc, idx),
              innerBasis, /*hasOuterBound=*/false)
              .getResults();
      mCoords.append(delinCoords.begin(), delinCoords.end());
    }
  }

  // Compute final offsets into the input tensor.
  OpFoldResult zero = b.getIndexAttr(0);
  OpFoldResult one = b.getIndexAttr(1);
  SmallVector<OpFoldResult> sliceOffsets(inputRank, zero);
  SmallVector<OpFoldResult> sliceSizes(inputRank, one);

  // Apply strides and dilations for spatial dimensions.
  AffineExpr mOff, wOff;
  bindDims(b.getContext(), mOff, wOff);
  for (auto [idx, mPos] : llvm::enumerate(im2colOp.getMPos())) {
    auto map =
        AffineMap::get(2, 0, {mOff * strides[idx] + wOff * dilations[idx]});
    OpFoldResult offset = affine::makeComposedFoldedAffineApply(
        b, loc, map, {mCoords[idx], windowOffset[idx]});
    sliceOffsets[mPos] = offset;
  }

  // Set K offsets.
  for (auto [kPos, kOff] :
       llvm::zip_equal(im2colOp.getKPos(), inputKOffset)) {
    sliceOffsets[kPos] = kOff;
  }

  // Set batch offsets from offset attribute + loop IVs.
  // Batch dims are first in canonical [Batch, M, K] order, so
  // canonicalIdx = ivIdx. The actual output dim comes from the inverse
  // output permutation.
  SmallVector<int64_t> inverseOutputPerm =
      invertPermutationVector(im2colOp.getOutputPerm());
  for (auto [ivIdx, bPos] : llvm::enumerate(im2colOp.getBatchPos())) {
    int64_t canonicalIdx = ivIdx;
    int64_t actualDim = inverseOutputPerm[canonicalIdx];
    sliceOffsets[bPos] = addOfrs(b, loc, offsets[canonicalIdx], ivs[actualDim]);
  }

  // The innermost input dimension gets innerTileSize as its size.
  int64_t innerInputDim = inputRank - 1;
  sliceSizes[innerInputDim] = innerTileSize;

  return Im2colSourceIndices{sliceOffsets, sliceSizes};
}

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
  auto val = dyn_cast<Value>(offset);
  if (!val)
    return false;
  auto affineOp = val.getDefiningOp<affine::AffineApplyOp>();
  return affineOp &&
         affineOp.getMap().getResult(0).isMultipleOf(constTileSize.value());
}

std::optional<int64_t>
chooseDimToVectorize(OpBuilder &b, Location loc, Im2colOp im2colOp,
                     ArrayRef<Range> iterationDomain,
                     ArrayRef<OpFoldResult> inputSizes,
                     ArrayRef<OpFoldResult> offsets) {
  int64_t innerInputDim = im2colOp.getInputRank() - 1;
  SmallVector<SmallVector<int64_t>> vectorizationMap =
      im2colOp.getInputToOutputDimVectorizationMap();
  SmallVector<int64_t> vectorizableOutputDims = vectorizationMap[innerInputDim];
  if (vectorizableOutputDims.empty()) {
    return std::nullopt;
  }
  SetVector<int64_t> kDimSet(llvm::from_range, im2colOp.getKOutputDims());

  // Build a map from actual output dim to canonical index for K dims.
  SmallVector<int64_t> kOutputDims = im2colOp.getKOutputDims();
  int64_t batchSize = im2colOp.getBatchPos().size();
  int64_t numMOutputDims = im2colOp.getNumMOutputDims();
  DenseMap<int64_t, int64_t> kDimToCanonicalIdx;
  for (auto [i, actualDim] : llvm::enumerate(kOutputDims)) {
    kDimToCanonicalIdx[actualDim] = batchSize + numMOutputDims + i;
  }

  // There may be multiple output dims that we can vectorize, so prioritize the
  // innermost dims first.
  llvm::sort(vectorizableOutputDims);
  // Check each dim in order from innermost to outermost, and return the first
  // one that is vectorizable.
  for (int64_t outputDimToVectorize : llvm::reverse(vectorizableOutputDims)) {
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
      // Use the offset of this specific K dim directly (no linearization).
      offset = offsets[kDimToCanonicalIdx[outputDimToVectorize]];
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

} // namespace mlir::iree_compiler::IREE::LinalgExt
