// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-linalg-ext-convert-conv-to-im2col-op"

namespace mlir::iree_compiler::IREE::LinalgExt {

#define GEN_PASS_DEF_CONVERTCONVTOIM2COLOPPASS
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"

static bool hasAllOneValues(ArrayRef<int64_t> attr) {
  return llvm::all_of(attr, [](int64_t element) { return element == 1; });
}

static Value createAdd(Location loc, Value x, Value y, OpBuilder &builder) {
  bool isInt = isa<IntegerType>(x.getType());
  if (isInt) {
    return arith::AddIOp::create(builder, loc, x, y);
  }
  return arith::AddFOp::create(builder, loc, x, y);
}

static Value createMul(Location loc, Value x, Value y, OpBuilder &builder) {
  bool isInt = isa<IntegerType>(x.getType());
  if (isInt) {
    return arith::MulIOp::create(builder, loc, x, y);
  }
  return arith::MulFOp::create(builder, loc, x, y);
}

// TODO : Upstream utility that does this pruning is broken for LinalgOp. Drop
// this if that gets fixed.
static SmallVector<NamedAttribute> getPrunedAttributeList(linalg::LinalgOp op) {
  const StringLiteral memoAttr =
      linalg::LinalgDialect::kMemoizedIndexingMapsAttrName;
  SmallVector<NamedAttribute> prunedAttributeList;
  for (auto attr : op->getDiscardableAttrs()) {
    if (attr.getName() != memoAttr) {
      prunedAttributeList.push_back(attr);
    }
  }
  return prunedAttributeList;
}

// Computes `inputKPerm` for the im2col op. Each non-batch input dimension
// contributes exactly one K-output coordinate (a window offset for M dims,
// including the trivial size-1 offset for passthrough M dims; a channel
// coordinate for K dims). The permutation maps each position in output-K
// delinearization order to the position of the corresponding input dim in
// input-dim iteration order (ascending over non-batch input dims).
//
// Output-K order is:
//   [passthrough-M coords (input-dim ascending order),
//    filter-reduction coords (in filter map order: inputChannel + filterLoop)]
static FailureOr<SmallVector<int64_t>>
computeInputKPerm(AffineMap inputMap, AffineMap filterMap,
                  const mlir::linalg::ConvolutionDimensions &convDims,
                  ArrayRef<int64_t> batchPos,
                  ArrayRef<int64_t> mPassthroughInputDims) {
  // Map from input dim to its position in input-dim iteration order (ascending
  // over non-batch dims).
  int64_t inputRank = inputMap.getNumResults();
  llvm::SmallDenseSet<int64_t, 4> batchSet(batchPos.begin(), batchPos.end());
  DenseMap<int64_t, int64_t> inputDimToIterPos;
  int64_t iterPos = 0;
  for (int64_t d = 0; d < inputRank; ++d) {
    if (!batchSet.contains(d)) {
      inputDimToIterPos[d] = iterPos++;
    }
  }

  SmallVector<int64_t> inputKPerm;
  llvm::SmallDenseSet<int64_t, 8> usedInputDims;
  auto appendInputDim = [&](int64_t inputDim) -> LogicalResult {
    auto it = inputDimToIterPos.find(inputDim);
    if (it == inputDimToIterPos.end()) {
      return failure();
    }
    if (!usedInputDims.insert(inputDim).second) {
      return failure();
    }
    inputKPerm.push_back(it->second);
    return success();
  };

  // Passthrough-M K coords come first, in input-dim ascending order.
  SmallVector<int64_t> sortedPassthrough(mPassthroughInputDims);
  llvm::sort(sortedPassthrough);
  for (int64_t inputDim : sortedPassthrough) {
    if (failed(appendInputDim(inputDim))) {
      return failure();
    }
  }

  // Filter reduction coords next, in filter map order. For each reduction dim
  // that appears in the filter, locate the input dim that references it.
  auto reductionDims =
      llvm::concat<const unsigned>(convDims.inputChannel, convDims.filterLoop);
  for (AffineExpr dimExpr : filterMap.getResults()) {
    bool foundFilterReduction = false;
    for (unsigned reductionDim : reductionDims) {
      if (!dimExpr.isFunctionOfDim(reductionDim)) {
        continue;
      }
      if (foundFilterReduction) {
        return failure();
      }
      foundFilterReduction = true;
      int64_t inputDim = -1;
      for (auto [idx, e] : llvm::enumerate(inputMap.getResults())) {
        if (e.isFunctionOfDim(reductionDim)) {
          inputDim = idx;
          break;
        }
      }
      if (inputDim < 0 || failed(appendInputDim(inputDim))) {
        return failure();
      }
    }
  }
  if (inputKPerm.size() != inputDimToIterPos.size()) {
    return failure();
  }
  return inputKPerm;
}

namespace {

using ControlFnTy = std::function<bool(Operation *)>;
// Converts non-depthwise convs into into linalg.generic (for img2col packing)
// and linalg.matmul.
// The following explains this for a linalg.conv_2d_nhwc_hwcf op.
//
// A convolution operation can be written as a matrix-matrix multiplication by
// unfolding the cross correlation between input and filter and explicitly copy
// overlapped sliding window inputs.
//
// Consider 2D input X with single channel input and output and 2x2 filter W:
// [x(0, 0)  , x(0, 1)  , ...,   x(0, n)  ]
// [x(1, 0)  , x(1, 1)  , ...,   x(1, n)  ]
// [.        ,  .       ,.   ,      .     ]            [w(0, 0), w(0, 1)]
// [.        ,  .       , .  ,      .     ]    (conv)  [w(1, 0), w(1, 1)]
// [.        ,  .       ,   .,      .     ]
// [x(n-1, 0), x(n-1, 1), ..., x(n-1, n-1)]
//
// The packed input data (img2col) is a matrix with |rows| = output spatial
// size, |columns| = filter spatial size. To compute the output Y(i, j) we need
// to calculate the dot product between filter window at input X(x, y)) and the
// filter which will look like the following where r.h.s is the img2col matrix
// and l.h.s is the flattened filter:
//
// clang-format off
// [x(0, 0), x(0, 1), x(1, 0), x(1, 1)]
// [x(0, 1), x(1, 1), x(0, 2), x(1, 2)] (matmul) [w(0, 0), w(0, 1), w(1, 0), w(1, 1)]
// [x(0, 1), x(1, 1), x(0, 2), x(1, 2)]
// [   .   ,    .   ,    .   ,    .   ]
// clang-format on
//
// In general for 2D case with (N, H, W, C) input and (Kh, Kw, C, D) filter
// and output (N, Ho, Wo, D) the convolution is the following matrix-matrix
// multiplication (Ho x Wo, Kh x Kw x C) * (Kh x Kw x C, D) for each input in
// the N batches. For the case where N > 1 its a batched matrxi-matrix
// multiplication.

class ConvertConvGeneric final
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
public:
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  ConvertConvGeneric(MLIRContext *context, std::optional<ControlFnTy> controlFn)
      : OpInterfaceRewritePattern(context), controlFn(controlFn) {}
  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (controlFn.has_value() && !controlFn.value()(linalgOp)) {
      return rewriter.notifyMatchFailure(linalgOp, "controlFn failed.");
    }

    auto igemmConvDetailsOrFailure =
        LinalgExt::getIGEMMGenericConvDetails(linalgOp);
    if (failed(igemmConvDetailsOrFailure)) {
      return rewriter.notifyMatchFailure(linalgOp,
                                         "Failed to extract IGEMM details");
    }

    LinalgExt::IGEMMGenericConvDetails igemmConvDetails =
        *igemmConvDetailsOrFailure;

    SmallVector<AffineMap> igemmContractionMaps =
        igemmConvDetails.igemmContractionMaps;
    mlir::linalg::ConvolutionDimensions convDims = igemmConvDetails.convDims;
    SmallVector<ReassociationIndices> filterReassocIndices =
        igemmConvDetails.filterReassocIndices;
    bool isOutputChannelFirst = igemmConvDetails.isOutputChannelFirst;
    SmallVector<int64_t> igemmLoopBounds = igemmConvDetails.igemmLoopBounds;
    SmallVector<utils::IteratorType> igemmLoopIterators =
        igemmConvDetails.igemmLoopIterators;

    Value input = linalgOp.getDpsInputs()[0];
    Value filter = linalgOp.getDpsInputs()[1];
    Value output = linalgOp.getDpsInits()[0];
    auto inputType = cast<ShapedType>(input.getType());
    auto filterType = cast<ShapedType>(filter.getType());
    auto outputType = cast<ShapedType>(output.getType());

    ArrayRef<int64_t> filterShape = filterType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
    AffineMap inputMap = indexingMaps[0];
    AffineMap filterMap = indexingMaps[1];
    AffineMap outputMap = indexingMaps[2];

    // The im2col dimension classification mirrors the GEMM roles:
    //   batch_pos = GEMM batch = conv depth/group dims.
    //   m_pos     = GEMM M     = conv batch dims (passthrough) and conv
    //                            outputImage dims (windowed).
    //   k_pos     = GEMM K     = conv inputChannel dims.
    // Passthrough M dims use stride=dilation=kernel_size=1 and carry a size-1
    // window-offset slot in the K output.
    SmallVector<int64_t> outputPerm = igemmConvDetails.im2colOutputPerm;
    // Locate the input-tensor dim that uses a given conv iteration dim. Handles
    // compound result exprs (e.g., `d_h + d_kh` for a spatial output dim).
    auto getIm2colInputDim = [&](unsigned convDim) -> FailureOr<int64_t> {
      for (auto [idx, e] : llvm::enumerate(inputMap.getResults())) {
        if (e.isFunctionOfDim(convDim)) {
          return idx;
        }
      }
      return failure();
    };
    auto getCanonicalPos = [&](unsigned convDim) -> int64_t {
      AffineExpr igemmDimExpr = igemmConvDetails.convToIgemmDimMap.at(convDim);
      int64_t igemmInputDim = igemmConvDetails.getIgemmInputImageMap()
                                  .getResultPosition(igemmDimExpr)
                                  .value();
      return outputPerm[igemmInputDim];
    };

    // Build batch_pos from conv depth dims, positioned by canonical layout.
    SmallVector<int64_t> batchPos(convDims.depth.size());
    for (unsigned convDim : convDims.depth) {
      int64_t canonicalPos = getCanonicalPos(convDim);
      FailureOr<int64_t> inputDim = getIm2colInputDim(convDim);
      if (failed(inputDim)) {
        return rewriter.notifyMatchFailure(linalgOp,
                                           "conv dim not found in input map");
      }
      batchPos[canonicalPos] = *inputDim;
    }
    const int64_t numBatch = batchPos.size();

    // Build m_pos from conv batch + outputImage dims, positioned by canonical
    // layout. Track per-entry stride/dilation/kernel_size and M output sizes.
    const int64_t numM = convDims.batch.size() + convDims.outputImage.size();
    SmallVector<int64_t> mPos(numM, -1);
    SmallVector<int64_t> mStrides(numM, 1);
    SmallVector<int64_t> mDilations(numM, 1);
    SmallVector<OpFoldResult> mKernelSizes(numM, rewriter.getIndexAttr(1));
    SmallVector<int64_t> mShape(numM, -1);
    SmallVector<char> mSlotAssigned(numM, false);
    SmallVector<int64_t> mPassthroughInputDims;
    auto assignMSlot = [&](int64_t canonicalPos) -> FailureOr<int64_t> {
      int64_t mSlot = canonicalPos - numBatch;
      if (mSlot < 0 || mSlot >= numM || mSlotAssigned[mSlot]) {
        return failure();
      }
      mSlotAssigned[mSlot] = true;
      return mSlot;
    };
    for (auto [spatialIdx, outputImageDim] :
         llvm::enumerate(convDims.outputImage)) {
      int64_t canonicalPos = getCanonicalPos(outputImageDim);
      FailureOr<int64_t> mSlot = assignMSlot(canonicalPos);
      if (failed(mSlot)) {
        return rewriter.notifyMatchFailure(linalgOp,
                                           "failed to assign im2col M slot");
      }
      FailureOr<int64_t> inputDim = getIm2colInputDim(outputImageDim);
      if (failed(inputDim)) {
        return rewriter.notifyMatchFailure(linalgOp,
                                           "conv dim not found in input map");
      }
      int64_t slot = *mSlot;
      int64_t dim = *inputDim;
      mPos[slot] = dim;
      mStrides[slot] = convDims.strides[spatialIdx];
      mDilations[slot] = convDims.dilations[spatialIdx];
      unsigned filterLoopDim = convDims.filterLoop[spatialIdx];
      std::optional<int64_t> maybeFilterDim = filterMap.getResultPosition(
          getAffineDimExpr(filterLoopDim, filterMap.getContext()));
      if (!maybeFilterDim) {
        return rewriter.notifyMatchFailure(linalgOp,
                                           "failed to infer filter shape");
      }
      mKernelSizes[slot] =
          rewriter.getIndexAttr(filterShape[maybeFilterDim.value()]);
      // Output spatial size lives in the conv output tensor.
      std::optional<int64_t> maybeOutDim = outputMap.getResultPosition(
          getAffineDimExpr(outputImageDim, outputMap.getContext()));
      if (!maybeOutDim) {
        return rewriter.notifyMatchFailure(
            linalgOp, "output image dim not found in output map");
      }
      mShape[slot] = outputShape[maybeOutDim.value()];
    }
    for (unsigned batchDim : convDims.batch) {
      int64_t canonicalPos = getCanonicalPos(batchDim);
      FailureOr<int64_t> mSlot = assignMSlot(canonicalPos);
      if (failed(mSlot)) {
        return rewriter.notifyMatchFailure(linalgOp,
                                           "failed to assign im2col M slot");
      }
      FailureOr<int64_t> inputDim = getIm2colInputDim(batchDim);
      if (failed(inputDim)) {
        return rewriter.notifyMatchFailure(linalgOp,
                                           "conv dim not found in input map");
      }
      int64_t slot = *mSlot;
      int64_t dim = *inputDim;
      mPos[slot] = dim;
      // Passthrough M uses stride=dilation=kernel=1 (already initialized).
      mShape[slot] = inputShape[dim];
      mPassthroughInputDims.push_back(dim);
    }
    if (llvm::any_of(mSlotAssigned, [](char assigned) { return !assigned; })) {
      return rewriter.notifyMatchFailure(linalgOp,
                                         "failed to assign all im2col M slots");
    }
    const int64_t numPassthroughM =
        static_cast<int64_t>(mPassthroughInputDims.size());

    SmallVector<int64_t> kPos;
    for (auto reductionDim : convDims.inputChannel) {
      for (auto [idx, e] : llvm::enumerate(inputMap.getResults())) {
        if (e.isFunctionOfDim(reductionDim)) {
          kPos.push_back(idx);
        }
      }
    }
    FailureOr<SmallVector<int64_t>> inputKPerm = computeInputKPerm(
        inputMap, filterMap, convDims, batchPos, mPassthroughInputDims);
    if (failed(inputKPerm)) {
      return rewriter.notifyMatchFailure(linalgOp,
                                         "failed to infer input K permutation");
    }

    // Classify each original filter dim as parallel, inputChannel, or
    // filterLoop, for partitioning the K output inner sizes.
    llvm::SmallDenseSet<int64_t, 4> parallelFilterDims;
    for (auto iterDim :
         llvm::concat<const unsigned>(convDims.depth, convDims.outputChannel)) {
      std::optional<int64_t> maybeDim = filterMap.getResultPosition(
          getAffineDimExpr(iterDim, filterMap.getContext()));
      if (maybeDim) {
        parallelFilterDims.insert(maybeDim.value());
      }
    }
    llvm::SmallDenseSet<int64_t, 4> inputChannelFilterDims;
    for (unsigned iterDim : convDims.inputChannel) {
      std::optional<int64_t> maybeDim = filterMap.getResultPosition(
          getAffineDimExpr(iterDim, filterMap.getContext()));
      if (maybeDim) {
        inputChannelFilterDims.insert(maybeDim.value());
      }
    }

    // Collect K output dim inner sizes from filter reassociation indices,
    // separated into inputChannel and filterLoop groups. The canonical
    // im2col output order is: [batch, M, inputChannel K, filterLoop K].
    SmallVector<SmallVector<int64_t>> inputChannelInnerSizes;
    SmallVector<SmallVector<int64_t>> filterLoopInnerSizes;
    for (const auto &indices : filterReassocIndices) {
      bool isParallel =
          indices.size() == 1 && parallelFilterDims.contains(indices[0]);
      if (isParallel) {
        continue;
      }
      SmallVector<int64_t> innerSizes;
      for (int64_t idx : indices) {
        innerSizes.push_back(filterShape[idx]);
      }
      // Classify as inputChannel if all filter dims in the group are
      // inputChannel dims; otherwise classify as filterLoop.
      bool isInputChannel = llvm::all_of(indices, [&](int64_t idx) {
        return inputChannelFilterDims.contains(idx);
      });
      if (isInputChannel) {
        inputChannelInnerSizes.push_back(innerSizes);
      } else {
        filterLoopInnerSizes.push_back(innerSizes);
      }
    }

    // Each passthrough M input dim contributes a size-1 window-offset slot to
    // the K output. Prepend them to the first non-empty canonical K group to
    // preserve the existing split between input-channel and filter-loop output
    // dims while making the K delinearization cover every non-batch input dim in
    // the same order used by input_k_perm.
    if (numPassthroughM > 0) {
      if (inputChannelInnerSizes.empty() && filterLoopInnerSizes.empty()) {
        return rewriter.notifyMatchFailure(
            linalgOp,
            "expected at least one K output dim with passthrough M dims");
      }
      SmallVector<int64_t> &firstK = inputChannelInnerSizes.empty()
                                         ? filterLoopInnerSizes.front()
                                         : inputChannelInnerSizes.front();
      firstK.insert(firstK.begin(), numPassthroughM, 1);
    }

    int64_t numOutputDims = numBatch + numM + inputChannelInnerSizes.size() +
                            filterLoopInnerSizes.size();
    SmallVector<OpFoldResult> offsets(numOutputDims, rewriter.getIndexAttr(0));
    SmallVector<SmallVector<OpFoldResult>> outputSizes;
    // Batch dims: each has a single inner size equal to the input dim size.
    for (int64_t dim : batchPos) {
      outputSizes.push_back({rewriter.getIndexAttr(inputShape[dim])});
    }
    // M dims: one output dim per m_pos entry (passthrough + spatial).
    for (int64_t m : mShape) {
      outputSizes.push_back({rewriter.getIndexAttr(m)});
    }
    // InputChannel K dims first, then filterLoop K dims. Passthrough M
    // window-offset slots live as size-1 prefixes inside the first K group.
    for (const auto &innerSizes : inputChannelInnerSizes) {
      outputSizes.push_back(getAsIndexOpFoldResult(getContext(), innerSizes));
    }
    for (const auto &innerSizes : filterLoopInnerSizes) {
      outputSizes.push_back(getAsIndexOpFoldResult(getContext(), innerSizes));
    }

    auto loc = linalgOp.getLoc();
    // Shape of the resulting tensor from im2col. Each output dim is the
    // product of its inner sizes.
    SmallVector<int64_t> colTensorShape;
    for (const auto &innerSizes : outputSizes) {
      int64_t dimSize = 1;
      for (OpFoldResult s : innerSizes) {
        std::optional<int64_t> constVal = getConstantIntValue(s);
        if (!constVal) {
          return rewriter.notifyMatchFailure(
              linalgOp, "dynamic inner sizes not supported");
        }
        dimSize *= *constVal;
      }
      colTensorShape.push_back(dimSize);
    }

    applyPermutationToVector(colTensorShape, outputPerm);
    Value colTensor = tensor::EmptyOp::create(rewriter, loc, colTensorShape,
                                              inputType.getElementType());
    Value img2ColTensor =
        IREE::LinalgExt::Im2colOp::create(
            rewriter, loc, input, /*output=*/colTensor, mStrides, mDilations,
            mKernelSizes, offsets, outputSizes, batchPos, mPos, kPos,
            *inputKPerm, outputPerm)
            .getResult(0);

    Value reshapedFilter = tensor::CollapseShapeOp::create(
        rewriter, loc, filter, filterReassocIndices);

    auto genericGEMMOp = linalg::GenericOp::create(
        rewriter, loc, outputType,
        /*inputs=*/
        isOutputChannelFirst ? ValueRange{reshapedFilter, img2ColTensor}
                             : ValueRange{img2ColTensor, reshapedFilter},
        /*outputs=*/ValueRange{output}, igemmContractionMaps,
        igemmLoopIterators,
        [](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          Value lhs = convertScalarToDtype(nestedBuilder, nestedLoc, args[0],
                                           args[2].getType(),
                                           /*isUnsignedCast=*/false);
          Value rhs = convertScalarToDtype(nestedBuilder, nestedLoc, args[1],
                                           args[2].getType(),
                                           /*isUnsignedCast=*/false);
          Value mul = createMul(nestedLoc, lhs, rhs, nestedBuilder);
          Value add = createAdd(nestedLoc, mul, args[2], nestedBuilder);
          linalg::YieldOp::create(nestedBuilder, nestedLoc, add);
        });
    genericGEMMOp->setDiscardableAttrs(getPrunedAttributeList(linalgOp));

    rewriter.replaceOp(linalgOp, genericGEMMOp.getResults().front());
    return success();
  }

private:
  std::optional<ControlFnTy> controlFn;
};

struct ConvertConvToIm2ColOpPass final
    : impl::ConvertConvToIm2ColOpPassBase<ConvertConvToIm2ColOpPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, IREELinalgExtDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateConvToIm2colOpPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

void populateConvToIm2colOpPatterns(RewritePatternSet &patterns,
                                    std::optional<ControlFnTy> controlFn) {
  patterns.insert<ConvertConvGeneric>(patterns.getContext(),
                                      std::move(controlFn));
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
