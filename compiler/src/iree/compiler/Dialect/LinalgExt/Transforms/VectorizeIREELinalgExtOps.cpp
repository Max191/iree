// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/Im2colUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Utils/Indexing.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

#define GEN_PASS_DEF_VECTORIZEIREELINALGEXTOPSPASS
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"

namespace {

struct VectorizeStaticMapStoreOpPattern final
    : OpRewritePattern<IREE::LinalgExt::MapStoreOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::LinalgExt::MapStoreOp mapStoreOp,
                                PatternRewriter &rewriter) const override {
    if (mapStoreOp.isVectorized()) {
      return rewriter.notifyMatchFailure(mapStoreOp,
                                         "map_store is already vectorized");
    }
    ShapedType inputType = mapStoreOp.getInputType();
    if (!inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(mapStoreOp,
                                         "map_store has non-static shape");
    }
    const int64_t innerSize = inputType.getShape()[inputType.getRank() - 1];
    const int64_t bitWidth = inputType.getElementTypeBitWidth();
    if ((innerSize * bitWidth % 8) != 0) {
      return rewriter.notifyMatchFailure(mapStoreOp,
                                         "map_store on sub-byte type");
    }
    // In case of a sub-byte bitwidth, we check that there is a contiguous copy
    // on the inner dimension that is a multiple of a byte. Note that the mask
    // shouldn't depend on the inner index for this.
    if (bitWidth < 8) {
      // First check that the mask is not the forward slice of the inner index.
      Value innermostInputIdx =
          mapStoreOp.getInputIndex(mapStoreOp.getInputRank() - 1);
      SetVector<Operation *> slice;
      getForwardSlice(innermostInputIdx, &slice);
      Operation *maskOp = mapStoreOp.getMask().getDefiningOp();
      if (maskOp && slice.contains(maskOp)) {
        return rewriter.notifyMatchFailure(
            mapStoreOp, "map_store on sub-byte type with potentially non "
                        "byte aligned transformation");
      }
      // Next check that the inner index of the yield is a unit function of
      // the inner input index.
      Value innermostOutputIdx =
          mapStoreOp.getOutputIndex(mapStoreOp.getOutputRank() - 1);
      if (!isUnitFunctionOf(innermostOutputIdx, innermostInputIdx)) {
        return rewriter.notifyMatchFailure(
            mapStoreOp, "map_store on sub-byte type with potentially non "
                        "byte aligned transformation");
      }
    }
    Location loc = mapStoreOp.getLoc();
    rewriter.setInsertionPoint(mapStoreOp);
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    SmallVector<Value> zeros(inputType.getRank(), zero);
    auto inputVectorType =
        VectorType::get(inputType.getShape(), inputType.getElementType());
    Value inputVector = vector::TransferReadOp::create(
        rewriter, loc, inputVectorType, mapStoreOp.getInput(),
        /*indices=*/zeros,
        /*padding=*/std::nullopt);
    auto vectorizedMapStoreOp =
        clone(rewriter, mapStoreOp, mapStoreOp.getResultTypes(),
              {inputVector, mapStoreOp.getOutput()});
    rewriter.replaceOp(mapStoreOp, vectorizedMapStoreOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Im2col Vectorization Helpers
//===----------------------------------------------------------------------===//

/// Holds the computed source indices for an im2col operation at a given
/// output position. These indices describe where to read from the input tensor.
struct Im2colSourceIndices {
  /// Full set of offsets into the input tensor, one per input dimension.
  SmallVector<OpFoldResult> sliceOffsets;
  /// Sizes for each input dimension (1 except for the vectorized dim).
  SmallVector<OpFoldResult> sliceSizes;
};

/// Compute source (input tensor) indices for a given im2col output position.
///
/// Given the loop induction variables representing the current output position,
/// compute the corresponding offsets and sizes into the input tensor. Uses the
/// unified offsets + output_sizes attributes to delinearize each output dim
/// independently, then maps to input coordinates via strides, dilations, and
/// input_k_perm.
static Im2colSourceIndices computeIm2colSourceIndices(
    OpBuilder &b, Location loc, Im2colOp im2colOp, ArrayRef<Value> ivs,
    ArrayRef<OpFoldResult> offsets,
    ArrayRef<SmallVector<OpFoldResult>> outputSizes,
    int64_t batchSize, int64_t numMOutputDims, OpFoldResult vecWidth) {
  int64_t inputRank = im2colOp.getInputRank();
  SetVector<int64_t> mPosSet(im2colOp.getMPos().begin(),
                             im2colOp.getMPos().end());
  SetVector<int64_t> batchPosSet(im2colOp.getBatchPos().begin(),
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
      ValueRange delinCoords =
          affine::AffineDelinearizeIndexOp::create(
              b, loc, getValueOrCreateConstantIndexOp(b, loc, idx), innerSizes,
              /*hasOuterBound=*/true)
              .getResults();
      kCoords.append(delinCoords.begin(), delinCoords.end());
    }
  }

  // Apply input_k_perm to kCoords, then split into window offsets and
  // channel offsets.
  SmallVector<int64_t> invInputKPerm = invertPermutationVector(inputKPerm);
  SmallVector<Value> permutedKCoords(kCoords.size());
  for (size_t i = 0; i < kCoords.size(); ++i) {
    permutedKCoords[invInputKPerm[i]] = kCoords[i];
  }
  SmallVector<Value> windowOffset, inputKOffset;
  int64_t permIdx = 0;
  for (int64_t i = 0; i < inputRank; ++i) {
    if (batchPosSet.contains(i)) {
      continue;
    }
    if (mPosSet.contains(i)) {
      windowOffset.push_back(permutedKCoords[permIdx++]);
      continue;
    }
    inputKOffset.push_back(permutedKCoords[permIdx++]);
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
      ValueRange delinCoords =
          affine::AffineDelinearizeIndexOp::create(
              b, loc, getValueOrCreateConstantIndexOp(b, loc, idx), innerSizes,
              /*hasOuterBound=*/true)
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

  // Set batch offsets from loop IVs.
  SmallVector<int64_t> inverseOutputPerm =
      invertPermutationVector(im2colOp.getOutputPerm());
  for (auto [ivIdx, bPos] : llvm::enumerate(im2colOp.getBatchPos())) {
    sliceOffsets[bPos] = ivs[inverseOutputPerm[ivIdx]];
  }

  // The innermost input dimension gets vecWidth as its size.
  int64_t innerInputDim = inputRank - 1;
  sliceSizes[innerInputDim] = vecWidth;

  return Im2colSourceIndices{sliceOffsets, sliceSizes};
}

/// Compute the padding mask and adjusted read indices for im2col
/// vectorization. When padding is present, source coordinates are in the
/// padded coordinate space and need adjustment for reading from the unpadded
/// input tensor. A mask gates the vector.transfer_read so that out-of-bounds
/// positions read pad_value instead.
///
/// The mask combines:
///   1. Source padding mask: spatial coords within unpadded input bounds
///   2. Result padding mask: (m, k) within real GEMM extent (handles
///      oversized output from GEMM alignment padding)
static Value computeIm2colPaddingMask(
    OpBuilder &b, Location loc, Im2colOp im2colOp, ArrayRef<Value> ivs,
    const Im2colSourceIndices &srcIndices, ArrayRef<OpFoldResult> inputSizes,
    ArrayRef<OpFoldResult> padLow, ArrayRef<OpFoldResult> padHigh,
    ArrayRef<OpFoldResult> offsets,
    ArrayRef<SmallVector<OpFoldResult>> outputSizes,
    int64_t batchSize, int64_t numMOutputDims,
    int64_t vecDim, int64_t vecWidth, SmallVector<Value> &readIndices) {
  int64_t inputRank = im2colOp.getInputRank();
  SetVector<int64_t> kDimSet(llvm::from_range, im2colOp.getKOutputDims());
  auto vecI1Type = VectorType::get({vecWidth}, b.getI1Type());

  // The source padding mask uses scalar bounds checks per input dimension.
  // This is correct for non-vectorized dimensions (one coordinate per
  // iteration), but the vectorized read spans vecWidth contiguous elements
  // along the innermost input dimension. If that dimension has non-zero
  // padding, individual vector lanes could straddle the boundary and the
  // scalar check would not correctly mask them. The caller (matchAndRewrite)
  // checks this precondition and bails out with a match failure before
  // reaching this point.

  // --- Adjust read indices: subtract padLow from source offsets ---
  for (int64_t d = 0; d < inputRank; ++d) {
    std::optional<int64_t> constPadLow = getConstantIntValue(padLow[d]);
    OpFoldResult adjusted = srcIndices.sliceOffsets[d];
    if (!constPadLow || *constPadLow != 0) {
      adjusted = subOfrs(b, loc, srcIndices.sliceOffsets[d], padLow[d]);
    }
    readIndices.push_back(getValueOrCreateConstantIndexOp(b, loc, adjusted));
  }

  // --- Source padding mask: check spatial coords in unpadded bounds ---
  // For each input dimension with non-zero padding, check that the source
  // coordinate is within [padLow, padLow + inputSize). This ensures we only
  // read from the real (unpadded) data region.
  Value scalarMask;
  for (int64_t d = 0; d < inputRank; ++d) {
    std::optional<int64_t> constPadLow = getConstantIntValue(padLow[d]);
    std::optional<int64_t> constPadHigh = getConstantIntValue(padHigh[d]);
    if (constPadLow && *constPadLow == 0 && constPadHigh &&
        *constPadHigh == 0) {
      continue;
    }
    Value coord =
        getValueOrCreateConstantIndexOp(b, loc, srcIndices.sliceOffsets[d]);
    Value lo = getValueOrCreateConstantIndexOp(b, loc, padLow[d]);
    Value hi = getValueOrCreateConstantIndexOp(
        b, loc, addOfrs(b, loc, padLow[d], inputSizes[d]));
    Value geqLo =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, coord, lo);
    Value ltHi =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, coord, hi);
    Value dimOk = arith::AndIOp::create(b, loc, geqLo, ltHi);
    scalarMask = scalarMask ? arith::AndIOp::create(b, loc, scalarMask, dimOk)
                            : dimOk;
  }

  // --- Result padding mask: check per output dim within real extent ---
  // When the output is oversized (GEMM alignment padding), some output
  // positions don't correspond to real data. Check each M and K output dim
  // independently against its real extent (product of output_sizes[d]).
  ArrayRef<int64_t> outputShape = im2colOp.getOutputType().getShape();
  SmallVector<int64_t> mOutputDims = im2colOp.getMOutputDims();
  SmallVector<int64_t> kOutputDims = im2colOp.getKOutputDims();
  int64_t numKOutputDims =
      im2colOp.getOutputRank() - batchSize - numMOutputDims;

  // Compute the real extent for an output dim from its output_sizes.
  auto computeDimExtent = [&](int64_t canonicalIdx) -> OpFoldResult {
    OpFoldResult extent = b.getIndexAttr(1);
    for (auto s : outputSizes[canonicalIdx]) {
      extent = mulOfrs(b, loc, extent, s);
    }
    return extent;
  };

  // Check each M output dim.
  for (int64_t i = 0; i < numMOutputDims; ++i) {
    int64_t canonicalIdx = batchSize + i;
    int64_t actualDim = mOutputDims[i];
    OpFoldResult realSize = computeDimExtent(canonicalIdx);
    std::optional<int64_t> constRealSize = getConstantIntValue(realSize);
    if (constRealSize && outputShape[actualDim] <= *constRealSize) {
      continue; // No masking needed for this dim.
    }
    OpFoldResult pos = addOfrs(b, loc, offsets[canonicalIdx], ivs[actualDim]);
    Value posVal = getValueOrCreateConstantIndexOp(b, loc, pos);
    Value realVal = getValueOrCreateConstantIndexOp(b, loc, realSize);
    Value ok = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt,
                                     posVal, realVal);
    scalarMask = scalarMask ? arith::AndIOp::create(b, loc, scalarMask, ok)
                            : ok;
  }

  // Check each K output dim.
  bool vecAlongK = vecDim >= 0 && kDimSet.contains(vecDim);
  for (int64_t i = 0; i < numKOutputDims; ++i) {
    int64_t canonicalIdx = batchSize + numMOutputDims + i;
    int64_t actualDim = kOutputDims[i];
    OpFoldResult realSize = computeDimExtent(canonicalIdx);
    std::optional<int64_t> constRealSize = getConstantIntValue(realSize);
    if (constRealSize && outputShape[actualDim] <= *constRealSize) {
      continue; // No masking needed for this dim.
    }

    if (vecAlongK && actualDim == vecDim) {
      // Vectorized K dim: produce per-element mask.
      // ivs[vecDim] = 0, so position = offsets[canonicalIdx] + lane.
      OpFoldResult pos = offsets[canonicalIdx];
      Value posVal = getValueOrCreateConstantIndexOp(b, loc, pos);
      Value realVal = getValueOrCreateConstantIndexOp(b, loc, realSize);
      Value diff = arith::SubIOp::create(b, loc, realVal, posVal);
      Value zeroIdx = arith::ConstantIndexOp::create(b, loc, 0);
      Value validLanes = arith::MaxSIOp::create(b, loc, diff, zeroIdx);
      Value kMask =
          vector::CreateMaskOp::create(b, loc, vecI1Type, validLanes);
      // Combine scalar source mask with per-element K mask.
      if (scalarMask) {
        Value broadcastScalar =
            vector::BroadcastOp::create(b, loc, vecI1Type, scalarMask);
        return arith::AndIOp::create(b, loc, broadcastScalar, kMask);
      }
      return kMask;
    }

    // Non-vectorized K dim: scalar check.
    OpFoldResult pos = addOfrs(b, loc, offsets[canonicalIdx], ivs[actualDim]);
    Value posVal = getValueOrCreateConstantIndexOp(b, loc, pos);
    Value realVal = getValueOrCreateConstantIndexOp(b, loc, realSize);
    Value ok = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt,
                                     posVal, realVal);
    scalarMask = scalarMask ? arith::AndIOp::create(b, loc, scalarMask, ok)
                            : ok;
  }

  // Broadcast the combined scalar mask to vector<Wxi1>.
  if (scalarMask) {
    return vector::BroadcastOp::create(b, loc, vecI1Type, scalarMask);
  }

  // No masking needed (all padding values are statically zero and output
  // matches real extent). This shouldn't happen when hasPadding() is true
  // in practice, but handle it gracefully.
  return nullptr;
}

/// Vectorization pattern for im2col ops.
///
/// Matches a tiled im2col op with static output shape and directly emits
/// vector.transfer_read + vector.transfer_write ops, bypassing the
/// decompose-then-vectorize flow (loops + extract_slice + linalg.copy +
/// insert_slice + vectorize copy).
///
/// The pattern:
/// 1. Chooses a vectorization dimension (most contiguous in input)
/// 2. Iterates over all non-vectorized output positions
/// 3. For each position, computes source indices and emits vector ops
///
/// When padding is present (hasPadding() == true):
/// - Computes effective input sizes (unpadded + padding) for basis computation
/// - Adjusts source coordinates for reading from the unpadded input
/// - Emits masked vector.transfer_read with combined source + result mask
///
/// Falls back to scalar unrolling (vec_width=1) when no dimension is
/// vectorizable.
struct VectorizeIm2colOpPattern final
    : OpRewritePattern<IREE::LinalgExt::Im2colOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::LinalgExt::Im2colOp im2colOp,
                                PatternRewriter &rewriter) const override {
    // Require static output shape for vectorization.
    ShapedType outputType = im2colOp.getOutputType();
    if (!outputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(im2colOp,
                                         "im2col has non-static output shape");
    }

    Location loc = im2colOp.getLoc();
    bool hasPadding = im2colOp.hasPadding();

    // Get offsets and outputSizes from unified attributes.
    SmallVector<OpFoldResult> mixedOffsets = im2colOp.getMixedOffsets();
    SmallVector<SmallVector<OpFoldResult>> mixedOutputSizes =
        im2colOp.getMixedOutputSizes();
    int64_t batchSize = im2colOp.getBatchPos().size();
    int64_t numMOutputDims = im2colOp.getNumMOutputDims();

    // Compute linearized K offset for chooseDimToVectorize.
    OpFoldResult kOffset = linearizeIm2colKOffsets(rewriter, loc, im2colOp);

    // Compute input sizes.
    SmallVector<OpFoldResult> inputSizes =
        tensor::getMixedSizes(rewriter, loc, im2colOp.getInput());
    SmallVector<OpFoldResult> padLow, padHigh;
    if (hasPadding) {
      padLow = im2colOp.getMixedInputPadLow();
      padHigh = im2colOp.getMixedInputPadHigh();
    }

    // Choose vectorization dimension (uses unpadded sizes for contiguity).
    SmallVector<Range> iterationDomain(
        im2colOp.getIterationDomain(rewriter));
    std::optional<int64_t> maybeVecDim = chooseDimToVectorize(
        rewriter, loc, im2colOp, iterationDomain, inputSizes, kOffset);

    int64_t outputRank = im2colOp.getOutputRank();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    Type elemType = outputType.getElementType();

    // Determine vectorization width and dimension.
    int64_t vecDim = -1;
    int64_t vecWidth = 1;
    if (maybeVecDim.has_value()) {
      vecDim = maybeVecDim.value();
      vecWidth = outputShape[vecDim];
    }

    auto vecType = VectorType::get({vecWidth}, elemType);

    // Check that the vectorized input dimension has no padding. The source
    // padding mask uses scalar bounds checks which cannot correctly mask
    // individual vector lanes straddling a padding boundary.
    if (hasPadding && vecWidth > 1) {
      int64_t vecInputDim = im2colOp.getInputRank() - 1;
      if (!isConstantIntValue(padLow[vecInputDim], 0) ||
          !isConstantIntValue(padHigh[vecInputDim], 0)) {
        return rewriter.notifyMatchFailure(
            im2colOp,
            "padding on the vectorized input dimension is not supported");
      }
    }

    // Determine pad value for vector.transfer_read.
    Value padValue;
    if (hasPadding) {
      padValue = im2colOp.getPadValue();
    } else {
      padValue = arith::ConstantOp::create(rewriter, loc, elemType,
                                           rewriter.getZeroAttr(elemType));
    }

    // Build permutation map for vector.transfer_write.
    // Write goes along vecDim, which may not be the innermost output dim
    // (e.g., CHWN layout where batch is innermost in input but outermost
    // in output). In that case we need an explicit permutation map.
    // Read always goes along the innermost input dimension (the default
    // minor identity map), so no explicit map is needed.
    int64_t writeDim = (vecDim >= 0) ? vecDim : (outputRank - 1);
    AffineMap writePermMap = AffineMap::get(
        outputRank, 0, rewriter.getAffineDimExpr(writeDim),
        rewriter.getContext());

    // Enumerate all positions in non-vectorized dimensions.
    // For each, compute indices and emit vector.transfer_read/write.
    SmallVector<int64_t> loopDims;
    SmallVector<int64_t> loopBounds;
    for (int64_t d = 0; d < outputRank; ++d) {
      if (d == vecDim) {
        continue;
      }
      loopDims.push_back(d);
      loopBounds.push_back(outputShape[d]);
    }

    // Compute total number of iterations across non-vectorized dims.
    int64_t totalIters = 1;
    for (int64_t bound : loopBounds) {
      totalIters *= bound;
    }

    // Conservative upper bound for static unrolling of im2col vectorization.
    // In practice, iGEMM tile sizes produce much fewer iterations (typically
    // 16-128). 1024 prevents IR explosion for unexpectedly large tiles while
    // still being permissive enough for any reasonable tile configuration.
    constexpr int64_t kMaxUnrollIters = 1024;
    if (totalIters > kMaxUnrollIters) {
      return rewriter.notifyMatchFailure(
          im2colOp, "im2col output too large for static unrolling");
    }

    // Start with the output tensor.
    Value result = im2colOp.getOutput();
    Value zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);

    for (int64_t iter = 0; iter < totalIters; ++iter) {
      // Compute multi-dimensional indices for this iteration.
      SmallVector<Value> ivs(outputRank, zeroIdx);

      // Delinearize the iteration index into per-dimension indices.
      // This is intentional compile-time (constant) delinearization: the loop
      // bounds and iteration index are all known at pattern-match time, so we
      // emit a constant index op per dimension rather than using
      // AffineDelinearizeIndexOp (which would introduce unnecessary runtime
      // arithmetic).
      int64_t remaining = iter;
      for (int64_t i = loopDims.size() - 1; i >= 0; --i) {
        int64_t idx = remaining % loopBounds[i];
        remaining /= loopBounds[i];
        ivs[loopDims[i]] =
            arith::ConstantIndexOp::create(rewriter, loc, idx);
      }

      // Compute source indices for this output position.
      Im2colSourceIndices srcIndices = computeIm2colSourceIndices(
          rewriter, loc, im2colOp, ivs, mixedOffsets, mixedOutputSizes,
          batchSize, numMOutputDims, rewriter.getIndexAttr(vecWidth));

      SmallVector<Value> readIndices;
      Value mask;

      if (hasPadding) {
        // Compute adjusted read indices and padding mask.
        mask = computeIm2colPaddingMask(
            rewriter, loc, im2colOp, ivs, srcIndices, inputSizes, padLow,
            padHigh, mixedOffsets, mixedOutputSizes, batchSize,
            numMOutputDims, vecDim, vecWidth, readIndices);
      } else {
        // No padding: use source indices directly.
        for (auto ofr : srcIndices.sliceOffsets) {
          readIndices.push_back(
              getValueOrCreateConstantIndexOp(rewriter, loc, ofr));
        }
      }

      // Emit vector.transfer_read from input tensor.
      Value readVec;
      if (mask) {
        AffineMap readPermMap = AffineMap::getMinorIdentityMap(
            im2colOp.getInputRank(), 1, rewriter.getContext());
        // in_bounds = [false]: the mask controls which lanes are valid,
        // so the transfer_read itself may access out-of-bounds positions.
        auto inBoundsAttr = rewriter.getBoolArrayAttr({false});
        readVec = vector::TransferReadOp::create(
            rewriter, loc, vecType, im2colOp.getInput(), readIndices,
            readPermMap, padValue, mask, inBoundsAttr);
      } else {
        readVec = vector::TransferReadOp::create(
            rewriter, loc, vecType, im2colOp.getInput(), readIndices,
            padValue);
      }

      // Emit vector.transfer_write to output tensor.
      SmallVector<Value> writeIndices(ivs);
      if (vecDim >= 0) {
        writeIndices[vecDim] = zeroIdx;
      }
      result = vector::TransferWriteOp::create(rewriter, loc, readVec, result,
                                               writeIndices, writePermMap)
                   .getResult();
    }

    rewriter.replaceOp(im2colOp, result);
    return success();
  }
};

struct VectorizeIREELinalgExtOpsPass final
    : impl::VectorizeIREELinalgExtOpsPassBase<VectorizeIREELinalgExtOpsPass> {
  void runOnOperation() {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<VectorizeStaticMapStoreOpPattern>(context);
    patterns.add<VectorizeIm2colOpPattern>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::IREE::LinalgExt
