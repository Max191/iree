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

/// Compute an OpFoldResult subtraction: a - b.
static OpFoldResult subOfrs(OpBuilder &b, Location loc, OpFoldResult a,
                            OpFoldResult rhs) {
  AffineExpr d0, d1;
  bindDims(b.getContext(), d0, d1);
  return affine::makeComposedFoldedAffineApply(
      b, loc, AffineMap::get(2, 0, {d0 - d1}, b.getContext()), {a, rhs});
}

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
/// compute the corresponding offsets and sizes into the input tensor. Handles:
///   - Linearizing and delinearizing M and K indices
///   - Applying strides and dilations for spatial dimensions
///   - Handling batch dimension offsets via output_perm
///   - Respecting input_k_perm
static Im2colSourceIndices computeIm2colSourceIndices(
    OpBuilder &b, Location loc, Im2colOp im2colOp, ArrayRef<Value> ivs,
    OpFoldResult mOffset, OpFoldResult kOffset,
    const Im2colMBasisResult &mBasisResult,
    ArrayRef<OpFoldResult> kBasis, OpFoldResult vecWidth) {
  int64_t inputRank = im2colOp.getInputRank();
  SetVector<int64_t> mPosSet(im2colOp.getMPos().begin(),
                             im2colOp.getMPos().end());
  SetVector<int64_t> batchPosSet(im2colOp.getBatchPos().begin(),
                                 im2colOp.getBatchPos().end());
  ArrayRef<int64_t> strides = im2colOp.getStrides();
  ArrayRef<int64_t> dilations = im2colOp.getDilations();
  ArrayRef<int64_t> inputKPerm = im2colOp.getInputKPerm();

  // Compute the linearized K index from loop IVs and k_offset.
  OpFoldResult kIndex = kOffset;
  for (auto [ivIdx, stride] : llvm::zip_equal(im2colOp.getKOutputDims(),
                                              im2colOp.getMixedKStrides())) {
    if (isConstantIntValue(ivs[ivIdx], 0)) {
      continue;
    }
    OpFoldResult ivOffset = mulOfrs(b, loc, stride, ivs[ivIdx]);
    kIndex = addOfrs(b, loc, kIndex, ivOffset);
  }

  // Delinearize K index.
  SmallVector<OpFoldResult> kBasisValues(kBasis);
  ValueRange delinKOffset =
      affine::AffineDelinearizeIndexOp::create(
          b, loc, getValueOrCreateConstantIndexOp(b, loc, kIndex), kBasisValues,
          /*hasOuterBound=*/true)
          .getResults();

  // Split delinearized K offsets into window offsets and input K offsets.
  SmallVector<Value> windowOffset, inputKOffset;
  int64_t delinKIdx = 0;
  SmallVector<int64_t> invInputKPerm = invertPermutationVector(inputKPerm);
  for (int64_t i = 0; i < inputRank; ++i) {
    if (batchPosSet.contains(i)) {
      continue;
    }
    if (mPosSet.contains(i)) {
      windowOffset.push_back(delinKOffset[invInputKPerm[delinKIdx++]]);
      continue;
    }
    inputKOffset.push_back(delinKOffset[invInputKPerm[delinKIdx++]]);
  }

  // Compute linearized M index.
  SmallVector<OpFoldResult> mIvs;
  SmallVector<OpFoldResult> mOutStrides(im2colOp.getMixedMStrides());
  for (auto dim : im2colOp.getMOutputDims()) {
    mIvs.push_back(ivs[dim]);
  }
  OpFoldResult linearMIv = linearizeIm2colIndex(b, loc, mIvs, mOutStrides);
  OpFoldResult linearMOffset = addOfrs(b, loc, linearMIv, mOffset);

  // Delinearize M index using the basis and hasOuterBound from
  // computeIm2colMBasis.
  // For expanded M: basis = m_strides[:-1], hasOuterBound=false.
  // For flat M: basis = formula-computed sizes, hasOuterBound=true.
  SmallVector<OpFoldResult> mBasisValues(mBasisResult.basis);
  ValueRange delinMOffset =
      affine::AffineDelinearizeIndexOp::create(
          b, loc, getValueOrCreateConstantIndexOp(b, loc, linearMOffset),
          mBasisValues, mBasisResult.hasOuterBound)
          .getResults();

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
        b, loc, map, {delinMOffset[idx], windowOffset[idx]});
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
    OpFoldResult mOffset, OpFoldResult kOffset,
    ArrayRef<OpFoldResult> mBasis, ArrayRef<OpFoldResult> kBasis,
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

  // --- Result padding mask: check M and K within real GEMM extent ---
  // When the output is oversized (GEMM alignment padding), some output
  // positions don't correspond to real data. The "real" extent is the
  // product of the basis vectors computed from the effective input sizes.
  OpFoldResult realM = b.getIndexAttr(1);
  for (auto basis : mBasis) {
    realM = mulOfrs(b, loc, realM, basis);
  }
  OpFoldResult realK = b.getIndexAttr(1);
  for (auto basis : kBasis) {
    realK = mulOfrs(b, loc, realK, basis);
  }

  // Check if result masking is needed (output exceeds real extent).
  ArrayRef<int64_t> outputShape = im2colOp.getOutputType().getShape();
  int64_t outputM = 1;
  for (auto dim : im2colOp.getMOutputDims()) {
    outputM *= outputShape[dim];
  }
  int64_t outputK = 1;
  for (auto dim : im2colOp.getKOutputDims()) {
    outputK *= outputShape[dim];
  }
  std::optional<int64_t> constRealM = getConstantIntValue(realM);
  std::optional<int64_t> constRealK = getConstantIntValue(realK);
  bool needMResultMask =
      !constRealM || outputM > *constRealM;
  bool needKResultMask =
      !constRealK || outputK > *constRealK;

  if (needMResultMask) {
    // Compute linearized M index from loop IVs.
    SmallVector<OpFoldResult> mIvs;
    for (auto dim : im2colOp.getMOutputDims()) {
      mIvs.push_back(ivs[dim]);
    }
    OpFoldResult linearMIv =
        linearizeIm2colIndex(b, loc, mIvs, im2colOp.getMixedMStrides());
    OpFoldResult linearMIdx = addOfrs(b, loc, linearMIv, mOffset);
    Value mIdx = getValueOrCreateConstantIndexOp(b, loc, linearMIdx);
    Value realMVal = getValueOrCreateConstantIndexOp(b, loc, realM);
    Value mOk =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, mIdx,
                               realMVal);
    scalarMask = scalarMask ? arith::AndIOp::create(b, loc, scalarMask, mOk)
                            : mOk;
  }

  // Build the K result mask. When vectorizing along a K dim, the mask is
  // per-element (vector<Wxi1>). Otherwise it's scalar.
  bool vecAlongK = vecDim >= 0 && kDimSet.contains(vecDim);

  // Compute linearized K index from loop IVs. Shared by both the vecAlongK
  // and scalar K paths.
  auto linearizeKIndex = [&]() -> Value {
    OpFoldResult kIndex = kOffset;
    for (auto [ivIdx, stride] : llvm::zip_equal(
             im2colOp.getKOutputDims(), im2colOp.getMixedKStrides())) {
      if (isConstantIntValue(ivs[ivIdx], 0)) {
        continue;
      }
      OpFoldResult ivOffset = mulOfrs(b, loc, stride, ivs[ivIdx]);
      kIndex = addOfrs(b, loc, kIndex, ivOffset);
    }
    return getValueOrCreateConstantIndexOp(b, loc, kIndex);
  };

  if (needKResultMask && vecAlongK) {
    Value kIdx = linearizeKIndex();
    Value realKVal = getValueOrCreateConstantIndexOp(b, loc, realK);
    // valid_lanes = max(0, realK - kIndex).
    Value diff = arith::SubIOp::create(b, loc, realKVal, kIdx);
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

  if (needKResultMask && !vecAlongK) {
    // Scalar K check: all K elements are at the same k_index.
    Value kIdx = linearizeKIndex();
    Value realKVal = getValueOrCreateConstantIndexOp(b, loc, realK);
    Value kOk = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt,
                                       kIdx, realKVal);
    scalarMask = scalarMask ? arith::AndIOp::create(b, loc, scalarMask, kOk)
                            : kOk;
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
  using Base::Base;

  LogicalResult matchAndRewrite(IREE::LinalgExt::Im2colOp im2colOp,
                                PatternRewriter &rewriter) const override {
    // Require static output shape for vectorization.
    ShapedType outputType = im2colOp.getOutputType();
    if (!outputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(im2colOp,
                                         "im2col has non-static output shape");
    }

    Location loc = im2colOp.getLoc();
    OpBuilder &b = rewriter;
    bool hasPadding = im2colOp.hasPadding();

    // Linearize the m_offset and k_offset.
    OpFoldResult mOffset =
        linearizeIm2colIndex(b, loc, im2colOp.getMixedMOffset(),
                             im2colOp.getMixedMStrides());
    OpFoldResult kOffset =
        linearizeIm2colIndex(b, loc, im2colOp.getMixedKOffset(),
                             im2colOp.getMixedKStrides());

    // Compute input sizes. When padding is present, the input operand is
    // the unpadded tensor. We compute "effective" sizes that include padding
    // for basis computation (mBasis, kBasis use the padded spatial extents).
    SmallVector<OpFoldResult> inputSizes =
        tensor::getMixedSizes(b, loc, im2colOp.getInput());
    SmallVector<OpFoldResult> effectiveInputSizes(inputSizes);
    SmallVector<OpFoldResult> padLow, padHigh;
    if (hasPadding) {
      padLow = im2colOp.getMixedInputPadLow();
      padHigh = im2colOp.getMixedInputPadHigh();
      for (size_t i = 0; i < inputSizes.size(); ++i) {
        effectiveInputSizes[i] =
            addOfrs(b, loc, inputSizes[i],
                    addOfrs(b, loc, padLow[i], padHigh[i]));
      }
    }

    // Compute bases using effective sizes (padded for mBasis/kBasis).
    Im2colMBasisResult mBasisResult =
        computeIm2colMBasis(b, loc, im2colOp, effectiveInputSizes);
    SmallVector<OpFoldResult> kBasis =
        computeIm2colKBasis(im2colOp, effectiveInputSizes);

    // Choose vectorization dimension (uses unpadded sizes for contiguity).
    SmallVector<Range> iterationDomain(im2colOp.getIterationDomain(b));
    std::optional<int64_t> maybeVecDim = chooseDimToVectorize(
        b, loc, im2colOp, iterationDomain, inputSizes, kOffset);

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
      padValue = arith::ConstantOp::create(b, loc, elemType,
                                           b.getZeroAttr(elemType));
    }

    // Build permutation map for vector.transfer_write.
    // Write goes along vecDim, which may not be the innermost output dim
    // (e.g., CHWN layout where batch is innermost in input but outermost
    // in output). In that case we need an explicit permutation map.
    // Read always goes along the innermost input dimension (the default
    // minor identity map), so no explicit map is needed.
    int64_t writeDim = (vecDim >= 0) ? vecDim : (outputRank - 1);
    AffineMap writePermMap = AffineMap::get(
        outputRank, 0, b.getAffineDimExpr(writeDim), b.getContext());

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
    Value zeroIdx = arith::ConstantIndexOp::create(b, loc, 0);

    for (int64_t iter = 0; iter < totalIters; ++iter) {
      // Compute multi-dimensional indices for this iteration.
      SmallVector<Value> ivs(outputRank, zeroIdx);

      // Delinearize the iteration index into per-dimension indices.
      int64_t remaining = iter;
      for (int64_t i = loopDims.size() - 1; i >= 0; --i) {
        int64_t idx = remaining % loopBounds[i];
        remaining /= loopBounds[i];
        ivs[loopDims[i]] =
            arith::ConstantIndexOp::create(b, loc, idx);
      }

      // Compute source indices for this output position.
      Im2colSourceIndices srcIndices = computeIm2colSourceIndices(
          b, loc, im2colOp, ivs, mOffset, kOffset, mBasisResult, kBasis,
          b.getIndexAttr(vecWidth));

      SmallVector<Value> readIndices;
      Value mask;

      if (hasPadding) {
        // Compute adjusted read indices and padding mask.
        mask = computeIm2colPaddingMask(
            b, loc, im2colOp, ivs, srcIndices, inputSizes, padLow, padHigh,
            mOffset, kOffset, mBasisResult.basis, kBasis, vecDim, vecWidth, readIndices);
      } else {
        // No padding: use source indices directly.
        for (auto ofr : srcIndices.sliceOffsets) {
          readIndices.push_back(
              getValueOrCreateConstantIndexOp(b, loc, ofr));
        }
      }

      // Emit vector.transfer_read from input tensor.
      Value readVec;
      if (mask) {
        int64_t inputRank = im2colOp.getInputRank();
        AffineMap readPermMap = AffineMap::getMinorIdentityMap(
            inputRank, 1, b.getContext());
        // in_bounds = [false]: the mask controls which lanes are valid,
        // so the transfer_read itself may access out-of-bounds positions.
        auto inBoundsAttr = b.getBoolArrayAttr({false});
        readVec = vector::TransferReadOp::create(
            b, loc, vecType, im2colOp.getInput(), readIndices, readPermMap,
            padValue, mask, inBoundsAttr);
      } else {
        readVec = vector::TransferReadOp::create(
            b, loc, vecType, im2colOp.getInput(), readIndices, padValue);
      }

      // Emit vector.transfer_write to output tensor.
      SmallVector<Value> writeIndices(ivs);
      if (vecDim >= 0) {
        writeIndices[vecDim] = zeroIdx;
      }
      result = vector::TransferWriteOp::create(b, loc, readVec, result,
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
