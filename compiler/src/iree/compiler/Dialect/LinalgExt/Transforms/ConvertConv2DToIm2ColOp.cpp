// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

static bool hasAllOneValues(DenseIntElementsAttr attr) {
  return llvm::all_of(
      attr, [](APInt element) { return element.getSExtValue() == 1; });
}

static Value createAdd(Location loc, Value x, Value y, bool isInt,
                       OpBuilder &builder) {
  if (isInt)
    return builder.create<arith::AddIOp>(loc, x, y);
  return builder.create<arith::AddFOp>(loc, x, y);
}

static Value createMul(Location loc, Value x, Value y, bool isInt,
                       OpBuilder &builder) {
  if (isInt)
    return builder.create<arith::MulIOp>(loc, x, y);
  return builder.create<arith::MulFOp>(loc, x, y);
}

namespace {

// Convert linalg.conv_2d_nhwc_hwcf into linalg.generic (for img2col packing)
// and linalg.matmul.
//
// A convolution operaton can be written as a matrix-matrix multiplication by
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
// and l.h.s is the flattned filter:
//
// clang-format off
// [x(0, 0), x(0, 1), x(1, 0), x(1, 1)]
// [x(0, 1), x(1, 1), x(0, 2), x(1, 2)] (matmul) [w(0, 0), w(0, 1), w(1, 0), w(1, 1)]
// [x(0, 1), x(1, 1), x(0, 2), x(1, 2)]
// [   .   ,    .   ,    .   ,    .   ]
// clang-format on
//
// In general for 2D case with (N, H, W, C) input and (Kh, Kw, C, D) filter
// and output (N, Ho, Wo, D) the convolutin is the following matrix-matrix
// multiplication (Ho x Wo, Kh x Kw x C) * (Kh x Kw x C, D) for each input in
// the N input. For the case where N > 1 its a batched matrxi-matrix
// multplication.
class ConvertConv2DNhwcHwcf final
    : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp convOp,
                                PatternRewriter &rewriter) const override {
    auto inputType = llvm::cast<ShapedType>(convOp.getInputs()[0].getType());
    auto filterType = llvm::cast<ShapedType>(convOp.getInputs()[1].getType());
    auto outputType = llvm::cast<ShapedType>(convOp.getOutputs()[0].getType());

    if (!filterType.hasStaticShape() || !inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(convOp, [](Diagnostic &diag) {
        diag << "[unimplemented] "
             << "expected 'filterType' and 'inputType' to have static shapes.";
      });
    }

    // TODO: Support dilation.
    if (!hasAllOneValues(convOp.getDilations())) {
      return rewriter.notifyMatchFailure(convOp, [](Diagnostic &diag) {
        diag << "[unimplemented] "
             << "expected no dilations (expected dilations to all be one).";
      });
    }

    Value input = convOp.getInputs()[0];
    Value filter = convOp.getInputs()[1];
    Value output = convOp.getOutputs()[0];

    auto filterShape = filterType.getShape();
    auto outputShape = outputType.getShape();

    const int n = outputShape[0];
    const int oh = outputShape[1];
    const int ow = outputShape[2];
    const int oc = outputShape[3];
    const int fh = filterShape[0];
    const int fw = filterShape[1];
    const int ic = filterShape[2];

    auto loc = convOp.getLoc();

    SmallVector<int64_t> colTensorShape = {n, oh * ow, fh * fw * ic};

    SmallVector<ReassociationIndices> outputReassocIndices;
    RankedTensorType reshapedOutputType;
    if (n == 1) {
      outputReassocIndices = {{0, 1, 2}, {3}};
      reshapedOutputType =
          RankedTensorType::get({oh * ow, oc}, outputType.getElementType());
    } else {
      outputReassocIndices = {{0}, {1, 2}, {3}};
      reshapedOutputType =
          RankedTensorType::get({n, oh * ow, oc}, outputType.getElementType());
    }

    Value colTensor = rewriter.create<tensor::EmptyOp>(
        loc, colTensorShape, inputType.getElementType());
    SmallVector<int64_t> strides(convOp.getStrides().getValues<int64_t>());
    SmallVector<int64_t> dilations(convOp.getDilations().getValues<int64_t>());
    SmallVector<OpFoldResult> kernelSize = {
        getAsIndexOpFoldResult(convOp->getContext(), fh),
        getAsIndexOpFoldResult(convOp->getContext(), fw)};
    SmallVector<OpFoldResult> kOffset = {
        getAsIndexOpFoldResult(convOp->getContext(), 0)};
    SmallVector<OpFoldResult> mOffset = {
        getAsIndexOpFoldResult(convOp->getContext(), 0)};
    SmallVector<int64_t> batchPos = {0};
    SmallVector<int64_t> mPos = {1, 2};
    SmallVector<int64_t> kPos = {3};
    Value img2ColTensor =
        rewriter
            .create<IREE::LinalgExt::Im2colOp>(
                loc, input, /*output=*/colTensor, strides, dilations,
                kernelSize, mOffset, kOffset, batchPos, mPos, kPos)
            .getResult(0);

    SmallVector<ReassociationIndices> filterReassocIndices = {{0, 1, 2}, {3}};
    auto reshapedFilterType =
        RankedTensorType::get({fh * fw * ic, oc}, inputType.getElementType());

    Value reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedFilterType, filter, filterReassocIndices);

    Value reshapedOutput = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedOutputType, output, outputReassocIndices);

    Value result;
    if (n == 1) {
      auto matmulOp = rewriter.create<linalg::MatmulOp>(
          loc, reshapedOutputType,
          ArrayRef<Value>{img2ColTensor, reshapedFilter},
          ArrayRef<Value>{reshapedOutput});
      result = matmulOp.getResults().front();
    } else {
      // For cases where batch is not 1, we need to keep the batch dimension
      // separate. Because the filter does not share the same batch dimension,
      // the batch dimension is only used in indexing the input and output. Thus
      // we cannot use existing linalg named ops like linalg.batch_matmul.
      // i.e. (B x) M x K * K x N = (B x) M x N
      AffineExpr bDim, mDim, nDim, kDim;
      bindDims(getContext(), bDim, mDim, nDim, kDim);
      auto lhsMap = AffineMap::get(4, 0, {bDim, mDim, kDim}, getContext());
      auto rhsMap = AffineMap::get(4, 0, {kDim, nDim}, getContext());
      auto resultMap = AffineMap::get(4, 0, {bDim, mDim, nDim}, getContext());
      auto parallel = utils::IteratorType::parallel;
      auto reduction = utils::IteratorType::reduction;
      SmallVector<utils::IteratorType> genericIterators = {parallel, parallel,
                                                           parallel, reduction};
      bool isInt = llvm::isa<IntegerType>(outputType.getElementType());
      auto genericOp = rewriter.create<linalg::GenericOp>(
          loc, reshapedOutputType,
          /*inputs=*/ValueRange{img2ColTensor, reshapedFilter},
          /*outputs=*/ValueRange{reshapedOutput},
          ArrayRef<AffineMap>{lhsMap, rhsMap, resultMap}, genericIterators,
          [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
            Value lhs = convertScalarToDtype(nestedBuilder, nestedLoc, args[0],
                                             args[2].getType(),
                                             /*isUnsignedCast=*/false);
            Value rhs = convertScalarToDtype(nestedBuilder, nestedLoc, args[1],
                                             args[2].getType(),
                                             /*isUnsignedCast=*/false);
            Value mul = createMul(nestedLoc, lhs, rhs, isInt, nestedBuilder);
            Value add =
                createAdd(nestedLoc, mul, args[2], isInt, nestedBuilder);
            nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
          });
      result = genericOp.getResults().front();
    }

    auto reshapedResult = rewriter.create<tensor::ExpandShapeOp>(
        loc, outputType, result, outputReassocIndices);

    rewriter.replaceOp(convOp, ArrayRef<Value>{reshapedResult});

    return success();
  }
};

// For nchw, because the channels are to the left of the image shape dimensions,
// the position of the contraction dimension in the resulting matmul is
// reversed. This swaps the LHS and RHS of the matmul when compared with nhwc
// (i.e. (D, C x Kh x Kw) * (C x Kh x Kw, Ho x Wo))
class ConvertConv2DNchwFchw final
    : public OpRewritePattern<linalg::Conv2DNchwFchwOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwOp convOp,
                                PatternRewriter &rewriter) const override {
    auto inputType = llvm::cast<ShapedType>(convOp.getInputs()[0].getType());
    auto filterType = llvm::cast<ShapedType>(convOp.getInputs()[1].getType());
    auto outputType = llvm::cast<ShapedType>(convOp.getOutputs()[0].getType());

    if (!filterType.hasStaticShape() || !inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(convOp, [](Diagnostic &diag) {
        diag << "[unimplemented] "
             << "expected 'filterType' and 'inputType' to have static shapes.";
      });
    }

    // TODO: Support dilation.
    if (!hasAllOneValues(convOp.getDilations()))
      return rewriter.notifyMatchFailure(convOp, [](Diagnostic &diag) {
        diag << "[unimplemented] "
             << "expected no dilations (expected dilations to all be one).";
      });

    Value input = convOp.getInputs()[0];
    Value filter = convOp.getInputs()[1];
    Value output = convOp.getOutputs()[0];

    auto filterShape = filterType.getShape();
    auto outputShape = outputType.getShape();

    const int n = outputShape[0];
    const int oc = outputShape[1];
    const int oh = outputShape[2];
    const int ow = outputShape[3];
    const int ic = filterShape[1];
    const int fh = filterShape[2];
    const int fw = filterShape[3];

    auto loc = convOp.getLoc();

    SmallVector<int64_t> colTensorShape = {n, oh * ow, ic * fh * fw};

    Value colTensor = rewriter.create<tensor::EmptyOp>(
        loc, colTensorShape, inputType.getElementType());
    SmallVector<int64_t> strides(convOp.getStrides().getValues<int64_t>());
    SmallVector<int64_t> dilations(convOp.getDilations().getValues<int64_t>());
    SmallVector<OpFoldResult> kernelSize = {
        getAsIndexOpFoldResult(convOp->getContext(), fh),
        getAsIndexOpFoldResult(convOp->getContext(), fw)};
    SmallVector<OpFoldResult> kOffset = {
        getAsIndexOpFoldResult(convOp->getContext(), 0)};
    SmallVector<OpFoldResult> mOffset = {
        getAsIndexOpFoldResult(convOp->getContext(), 0)};
    SmallVector<int64_t> batchPos = {0};
    SmallVector<int64_t> mPos = {2, 3};
    SmallVector<int64_t> kPos = {1};
    Value img2ColTensor =
        rewriter
            .create<IREE::LinalgExt::Im2colOp>(
                loc, input, /*output=*/colTensor, strides, dilations,
                kernelSize, mOffset, kOffset, batchPos, mPos, kPos)
            .getResult(0);

    SmallVector<ReassociationIndices> filterReassocIndices = {{0}, {1, 2, 3}};
    auto reshapedFilterType =
        RankedTensorType::get({oc, fh * fw * ic}, inputType.getElementType());
    Value reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedFilterType, filter, filterReassocIndices);

    SmallVector<ReassociationIndices> outputReassocIndices;
    RankedTensorType reshapedOutputType;
    if (n == 1) {
      outputReassocIndices = {{0, 1}, {2, 3}};
      reshapedOutputType =
          RankedTensorType::get({oc, oh * ow}, outputType.getElementType());
    } else {
      outputReassocIndices = {{0}, {1}, {2, 3}};
      reshapedOutputType =
          RankedTensorType::get({n, oc, oh * ow}, outputType.getElementType());
    }

    Value reshapedOutput = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedOutputType, output, outputReassocIndices);

    Value result;
    if (n == 1) {
      auto matmulOp = rewriter.create<linalg::MatmulTransposeBOp>(
          loc, reshapedOutputType,
          ArrayRef<Value>{reshapedFilter, img2ColTensor},
          ArrayRef<Value>{reshapedOutput});
      result = matmulOp.getResults().front();
    } else {
      // For cases where batch is not 1, we need to keep the batch dimension
      // separate. Because the filter does not share the same batch dimension,
      // the batch dimension is only used in indexing the input and output. Thus
      // we cannot use existing linalg named ops like linalg.batch_matmul.
      // i.e. M x K * (B x) K x N = (B x) M x N
      AffineExpr bDim, mDim, nDim, kDim;
      bindDims(getContext(), bDim, mDim, nDim, kDim);
      auto lhsMap = AffineMap::get(4, 0, {mDim, kDim}, getContext());
      auto rhsMap = AffineMap::get(4, 0, {bDim, nDim, kDim}, getContext());
      auto resultMap = AffineMap::get(4, 0, {bDim, mDim, nDim}, getContext());
      auto parallel = utils::IteratorType::parallel;
      auto reduction = utils::IteratorType::reduction;
      SmallVector<utils::IteratorType> genericIterators = {parallel, parallel,
                                                           parallel, reduction};
      bool isInt = llvm::isa<IntegerType>(outputType.getElementType());
      auto genericOp = rewriter.create<linalg::GenericOp>(
          loc, reshapedOutputType,
          /*inputs=*/ValueRange{reshapedFilter, img2ColTensor},
          /*outputs=*/ValueRange{reshapedOutput},
          ArrayRef<AffineMap>{lhsMap, rhsMap, resultMap}, genericIterators,
          [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
            Value lhs = convertScalarToDtype(nestedBuilder, nestedLoc, args[0],
                                             args[2].getType(),
                                             /*isUnsignedCast=*/false);
            Value rhs = convertScalarToDtype(nestedBuilder, nestedLoc, args[1],
                                             args[2].getType(),
                                             /*isUnsignedCast=*/false);
            Value mul = createMul(nestedLoc, lhs, rhs, isInt, nestedBuilder);
            Value add =
                createAdd(nestedLoc, mul, args[2], isInt, nestedBuilder);
            nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
          });
      result = genericOp.getResults().front();
    }

    auto reshapedResult = rewriter.create<tensor::ExpandShapeOp>(
        loc, outputType, result, outputReassocIndices);

    rewriter.replaceOp(convOp, ArrayRef<Value>{reshapedResult});

    return success();
  }
};

struct ConvertConv2DToIm2ColOpPass
    : ConvertConv2DToIm2ColOpBase<ConvertConv2DToIm2ColOpPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, IREE::LinalgExt::IREELinalgExtDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<ConvertConv2DNhwcHwcf, ConvertConv2DNchwFchw>(context);
    linalg::FillOp::getCanonicalizationPatterns(patterns, context);
    tensor::populateFoldTensorEmptyPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createConvertConv2DToIm2ColOpPass() {
  return std::make_unique<ConvertConv2DToIm2ColOpPass>();
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
