// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/WinogradConstants.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::LinalgExt {
namespace {

/// Pattern to decompose the tiled im2col op.
struct DecomposeIm2col : public OpRewritePattern<Im2colOp> {
  using OpRewritePattern<Im2colOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(Im2colOp im2colOp,
                                PatternRewriter &rewriter) const override {
    Location loc = im2colOp.getLoc();
    Value inputSlice = im2colOp.getInput();
    // Unroll all but the K loop
    SmallVector<OpFoldResult> kOffset = im2colOp.getMixedKOffset();
    // Only support single K dimension for now.
    if (kOffset.size() != 1) {
      return failure();
    }
    SmallVector<OpFoldResult> basis(im2colOp.getMixedKernelSize());
    SmallVector<OpFoldResult> inputSizes =
        tensor::getMixedSizes(rewriter, loc, inputSlice);
    basis.push_back(inputSizes[im2colOp.getKPos().front()]);
    FailureOr<SmallVector<Value>> maybeDelinKOffset = affine::delinearizeIndex(
        rewriter, loc,
        getValueOrCreateConstantIndexOp(rewriter, loc, kOffset.front()),
        getValueOrCreateConstantIndexOp(rewriter, loc, (basis)));
    if (failed(maybeDelinKOffset)) {
      return failure();
    }
    SmallVector<Value> delinKOffset = maybeDelinKOffset.value();
    SmallVector<Range> iterationDomain(im2colOp.getIterationDomain(rewriter));
    OpFoldResult kTileSize = iterationDomain.back().size;
    if (auto constKTileSize = getConstantIntValue(kTileSize)) {
      kTileSize = rewriter.getIndexAttr(constKTileSize.value());
    }
    iterationDomain.pop_back();
    SmallVector<Value> lbs, ubs, steps;
    for (auto range : iterationDomain) {
      lbs.push_back(
          getValueOrCreateConstantIndexOp(rewriter, loc, range.offset));
      ubs.push_back(getValueOrCreateConstantIndexOp(rewriter, loc, range.size));
      steps.push_back(
          getValueOrCreateConstantIndexOp(rewriter, loc, range.stride));
    }
    scf::LoopNest loopNest = scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps, im2colOp.getOutput(),
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange outputIvs,
            ValueRange iterArgs) -> scf::ValueVector { return iterArgs; });
    SmallVector<Value> ivs;
    for (scf::ForOp loop : loopNest.loops) {
      ivs.push_back(loop.getInductionVar());
    }
    SmallVector<OpFoldResult> offsets(
        im2colOp.getInputRank(),
        getAsIndexOpFoldResult(rewriter.getContext(), 0));
    SmallVector<OpFoldResult> strides(
        im2colOp.getInputRank(),
        getAsIndexOpFoldResult(rewriter.getContext(), 1));
    SmallVector<OpFoldResult> sizes(inputSizes);
    SmallVector<Value> mBasis;
    SmallVector<OpFoldResult> dilations = im2colOp.getMixedDilations();
    SmallVector<OpFoldResult> kernelSize = im2colOp.getMixedKernelSize();
    for (auto [idx, mPos] : llvm::enumerate(im2colOp.getMPos())) {
      AffineExpr x, k, d;
      bindDims(rewriter.getContext(), x, k, d);
      auto map = AffineMap::get(3, 0, {x - (k - 1) * d});
      OpFoldResult adjustedBase = affine::makeComposedFoldedAffineApply(
          rewriter, loc, map, {sizes[mPos], kernelSize[idx], dilations[idx]});
      mBasis.push_back(
          getValueOrCreateConstantIndexOp(rewriter, loc, adjustedBase));
    }
    Location nestedLoc =
        loopNest.loops.back().getBody()->getTerminator()->getLoc();
    rewriter.setInsertionPointToStart(loopNest.loops.back().getBody());
    Value mArg = ivs.back();
    FailureOr<SmallVector<Value>> maybeDelinMArg = affine::delinearizeIndex(
        rewriter, nestedLoc,
        getValueOrCreateConstantIndexOp(rewriter, nestedLoc, mArg), mBasis);
    if (failed(maybeDelinMArg)) {
      return failure();
    }
    SmallVector<Value> delinMArg = maybeDelinMArg.value();
    AffineExpr d0, d1;
    bindDims(rewriter.getContext(), d0, d1);
    auto addMap = AffineMap::get(2, 0, {d0 + d1});
    for (auto [idx, mPos] : llvm::enumerate(im2colOp.getMPos())) {
      OpFoldResult offset = affine::makeComposedFoldedAffineApply(
          rewriter, nestedLoc, addMap, {delinMArg[idx], delinKOffset[idx]});
      offsets[mPos] = offset;
      sizes[mPos] = getAsIndexOpFoldResult(rewriter.getContext(), 1);
    }
    const int64_t kPos = im2colOp.getKPos().front();
    offsets[kPos] = delinKOffset.back();
    sizes[kPos] = kTileSize;
    ShapedType outputType = im2colOp.getOutputType();
    auto extractType = RankedTensorType::get({outputType.getShape().back()},
                                             outputType.getElementType());
    auto extract = rewriter.create<tensor::ExtractSliceOp>(
        nestedLoc, extractType, inputSlice, offsets, sizes, strides);
    SmallVector<OpFoldResult> copySize = {kTileSize};
    Value copyInit = rewriter.create<tensor::EmptyOp>(
        nestedLoc, copySize, outputType.getElementType());
    auto copiedSlice = rewriter.create<linalg::CopyOp>(
        nestedLoc, extract.getResult(), copyInit);
    offsets = SmallVector<OpFoldResult>(
        im2colOp.getOutputRank(),
        getAsIndexOpFoldResult(rewriter.getContext(), 0));
    sizes = SmallVector<OpFoldResult>(
        im2colOp.getOutputRank(),
        getAsIndexOpFoldResult(rewriter.getContext(), 1));
    strides = SmallVector<OpFoldResult>(
        im2colOp.getOutputRank(),
        getAsIndexOpFoldResult(rewriter.getContext(), 1));
    sizes.back() = kTileSize;
    for (auto [idx, iv] : llvm::enumerate(ivs)) {
      offsets[idx] = iv;
    }
    auto insert = rewriter.create<tensor::InsertSliceOp>(
        nestedLoc, copiedSlice.getResult(0),
        loopNest.loops.back().getRegionIterArg(0), offsets, sizes, strides);
    auto yieldOp =
        cast<scf::YieldOp>(loopNest.loops.back().getBody()->getTerminator());
    rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, insert.getResult());
    rewriter.replaceOp(im2colOp, loopNest.results[0]);
    return success();
  }
};

} // namespace

namespace {
struct DecomposeIm2colPass : public DecomposeIm2colBase<DecomposeIm2colPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        affine::AffineDialect, IREE::LinalgExt::IREELinalgExtDialect,
        linalg::LinalgDialect, scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void DecomposeIm2colPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.add<DecomposeIm2col>(context);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createDecomposeIm2colPass() {
  return std::make_unique<DecomposeIm2colPass>();
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
