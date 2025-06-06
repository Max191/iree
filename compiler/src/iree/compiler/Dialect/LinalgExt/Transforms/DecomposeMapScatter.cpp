// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

#define GEN_PASS_DEF_DECOMPOSEMAPSCATTERPASS
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"

static LogicalResult decomposeMapScatter(MapScatterOp mapScatterOp,
                                         RewriterBase &rewriter) {
  auto inputType = dyn_cast<VectorType>(mapScatterOp.getInputType());
  if (!inputType || !inputType.hasStaticShape()) {
    return success();
  }
  auto outputType = dyn_cast<MemRefType>(mapScatterOp.getOutputType());
  if (!outputType) {
    return success();
  }
  SmallVector<ReassociationIndices> reassociations;
  reassociations.push_back(
      llvm::to_vector(llvm::seq<int64_t>(outputType.getRank())));
  if (!memref::CollapseShapeOp::isGuaranteedCollapsible(outputType,
                                                        reassociations)) {
    return failure();
  }
  Location loc = mapScatterOp.getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(mapScatterOp);
  Value flatOutputBuffer = rewriter.create<memref::CollapseShapeOp>(
      loc, mapScatterOp.getOutput(), reassociations);

  Value idxInit = rewriter.create<tensor::EmptyOp>(loc, inputType.getShape(),
                                                   rewriter.getIndexType());
  Value maskInit = rewriter.create<tensor::EmptyOp>(loc, inputType.getShape(),
                                                    rewriter.getIntegerType(1));
  SmallVector<OpFoldResult> outputSizes =
      memref::getMixedSizes(rewriter, loc, mapScatterOp.getOutput());

  auto bodyBuilder = [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
    auto inlineBodyBuilder = [&](OpBuilder inlineBuilder, Location inlineLoc,
                                 ArrayRef<Value> yieldedValues) {
      SmallVector<Value> outputIndices(yieldedValues);
      Value mask = outputIndices.pop_back_val();
      Value linearIdx = inlineBuilder.create<affine::AffineLinearizeIndexOp>(
          inlineLoc, outputIndices, outputSizes, /*disjoint=*/true);
      inlineBuilder.create<linalg::YieldOp>(inlineLoc,
                                            ValueRange{linearIdx, mask});
    };
    SmallVector<Value> indices = llvm::map_to_vector(
        llvm::seq<int64_t>(inputType.getRank()), [&](int64_t dim) -> Value {
          return b.create<linalg::IndexOp>(nestedLoc, b.getIndexType(), dim);
        });
    mapScatterOp.inlineMapScatterBody(b, nestedLoc, indices, inlineBodyBuilder);
  };
  SmallVector<AffineMap> maps(
      2, rewriter.getMultiDimIdentityMap(inputType.getRank()));
  SmallVector<utils::IteratorType> iterTypes(inputType.getRank(),
                                             utils::IteratorType::parallel);
  SmallVector<Value> outs = {idxInit, maskInit};
  auto genericOp = rewriter.create<linalg::GenericOp>(
      loc, TypeRange(outs), ValueRange(), outs, maps, iterTypes, bodyBuilder);

  // Lower linearize and delinearize ops before vectorizing, because the
  // vectorizer can't hendle them.
  SmallVector<affine::AffineLinearizeIndexOp> linearizeOps(
      genericOp.getBody()->getOps<affine::AffineLinearizeIndexOp>());
  for (auto linearizeOp : linearizeOps) {
    rewriter.setInsertionPoint(linearizeOp);
    if (failed(affine::lowerAffineLinearizeIndexOp(rewriter, linearizeOp))) {
      return failure();
    }
  }
  SmallVector<affine::AffineDelinearizeIndexOp> delinearizeOps(
      genericOp.getBody()->getOps<affine::AffineDelinearizeIndexOp>());
  for (auto delinearizeOp : delinearizeOps) {
    rewriter.setInsertionPoint(delinearizeOp);
    if (failed(
            affine::lowerAffineDelinearizeIndexOp(rewriter, delinearizeOp))) {
      return failure();
    }
  }

  SmallVector<Value> newResults;
  if (failed(linalg::vectorize(rewriter, genericOp, newResults))) {
    return failure();
  }

  auto indexWriteOp = newResults[0].getDefiningOp<vector::TransferWriteOp>();
  auto maskWriteOp = newResults[1].getDefiningOp<vector::TransferWriteOp>();
  if (!indexWriteOp || !maskWriteOp) {
    return failure();
  }
  Value indexVector = indexWriteOp.getVector();
  Value maskVector = maskWriteOp.getVector();

  // Flatten all the vectors, since the scatter op lowering expects 1D vectors.
  int64_t flatVectorSize =
      std::reduce(inputType.getShape().begin(), inputType.getShape().end(), 1,
                  std::multiplies<int64_t>());
  rewriter.setInsertionPoint(mapScatterOp);
  auto flatIndexType =
      VectorType::get({flatVectorSize}, rewriter.getIndexType());
  indexVector =
      rewriter.create<vector::ShapeCastOp>(loc, flatIndexType, indexVector);
  auto flatMaskType =
      VectorType::get({flatVectorSize}, rewriter.getIntegerType(1));
  maskVector =
      rewriter.create<vector::ShapeCastOp>(loc, flatMaskType, maskVector);
  auto flatInputType =
      VectorType::get({flatVectorSize}, inputType.getElementType());
  Value inputVector = rewriter.create<vector::ShapeCastOp>(
      loc, flatInputType, mapScatterOp.getInput());

  SmallVector<Value> offsets = {
      rewriter.create<arith::ConstantIndexOp>(loc, 0)};
  rewriter.replaceOpWithNewOp<vector::ScatterOp>(mapScatterOp, flatOutputBuffer,
                                                 offsets, indexVector,
                                                 maskVector, inputVector);
  rewriter.eraseOp(genericOp);
  return success();
}

namespace {
struct DecomposeMapScatterPass final
    : impl::DecomposeMapScatterPassBase<DecomposeMapScatterPass> {
  using impl::DecomposeMapScatterPassBase<
      DecomposeMapScatterPass>::DecomposeMapScatterPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto funcOp = getOperation();

    SmallVector<MapScatterOp> candidates;
    funcOp->walk([&](MapScatterOp op) { candidates.push_back(op); });
    IRRewriter rewriter(context);
    for (auto mapScatterOp : candidates) {
      if (failed(decomposeMapScatter(mapScatterOp, rewriter))) {
        return signalPassFailure();
      }
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::IREE::LinalgExt
