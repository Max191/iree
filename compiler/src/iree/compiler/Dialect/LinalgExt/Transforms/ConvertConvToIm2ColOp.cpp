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
    SmallVector<ReassociationIndices> filterReassocIndices =
        igemmConvDetails.filterReassocIndices;
    bool isOutputChannelFirst = igemmConvDetails.isOutputChannelFirst;
    SmallVector<utils::IteratorType> igemmLoopIterators =
        igemmConvDetails.igemmLoopIterators;
    const LinalgExt::Im2colMetadata &im2colMetadata =
        igemmConvDetails.im2colMetadata;

    Value input = linalgOp.getDpsInputs()[0];
    Value filter = linalgOp.getDpsInputs()[1];
    Value output = linalgOp.getDpsInits()[0];
    auto inputType = cast<ShapedType>(input.getType());
    auto outputType = cast<ShapedType>(output.getType());

    SmallVector<int64_t> outputPerm = igemmConvDetails.im2colOutputPerm;
    SmallVector<OpFoldResult> mKernelSizes =
        getAsIndexOpFoldResult(getContext(), im2colMetadata.kernelSizes);
    SmallVector<SmallVector<OpFoldResult>> outputSizes;
    outputSizes.reserve(im2colMetadata.outputSizes.size());
    for (const SmallVector<int64_t> &innerSizes :
         im2colMetadata.outputSizes) {
      outputSizes.push_back(getAsIndexOpFoldResult(getContext(), innerSizes));
    }
    int64_t numOutputDims = outputSizes.size();
    SmallVector<OpFoldResult> offsets(numOutputDims, rewriter.getIndexAttr(0));

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
            rewriter, loc, input, /*output=*/colTensor,
            im2colMetadata.strides, im2colMetadata.dilations, mKernelSizes,
            offsets, outputSizes, im2colMetadata.batchPos, im2colMetadata.mPos,
            im2colMetadata.kPos, im2colMetadata.inputKPerm, outputPerm)
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
