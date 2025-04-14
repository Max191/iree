// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-combine-layout-transformation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_COMBINELAYOUTTRANSFORMATIONPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

static IREE::LinalgExt::MapScatterOp
foldIdentityLikeOpIntoMapScatter(RewriterBase &rewriter, Operation *op,
                                 IREE::LinalgExt::MapScatterOp mapScatterOp) {
  assert(mapScatterOp.getInput() == op->getResult(0) &&
         "expected op to be the producer of mapScatterOp");
  rewriter.modifyOpInPlace(
      mapScatterOp, [&]() { mapScatterOp->setOperand(0, op->getOperand(0)); });
  return mapScatterOp;
}

static IREE::LinalgExt::MapScatterOp
foldTransposeIntoMapScatter(RewriterBase &rewriter,
                            linalg::TransposeOp transposeOp,
                            IREE::LinalgExt::MapScatterOp mapScatterOp) {
  assert(mapScatterOp.getInput() == transposeOp->getResult(0) &&
         "expected transposeOp to be the producer of mapScatterOp");
  ShapedType inputType = mapScatterOp.getInput().getType();
  ArrayRef<int64_t> transposePerm = transposeOp.getPermutation();
  SmallVector<AffineExpr> newInputDims = llvm::map_to_vector(
      llvm::seq<int64_t>(inputType.getRank()), [&](int64_t dim) {
        return rewriter.getAffineDimExpr(transposePerm[dim]);
      });
  rewriter.modifyOpInPlace(mapScatterOp, [&]() {
    mapScatterOp.replaceInputDims(newInputDims);
    mapScatterOp->setOperand(0, transposeOp.getInput());
  });
  return mapScatterOp;
}

static IREE::LinalgExt::MapScatterOp
foldExpandShapeIntoMapScatter(RewriterBase &rewriter,
                              tensor::ExpandShapeOp expandShapeOp,
                              IREE::LinalgExt::MapScatterOp mapScatterOp) {
  assert(mapScatterOp.getInput() == expandShapeOp->getResult(0) &&
         "expected expandShapeOp to be the producer of mapScatterOp");

  // Create some AffineExpr lists for the source and result shape of the
  // expandShapeOp.
  ShapedType expandSrcType = expandShapeOp.getSrcType();
  int64_t numDims = expandSrcType.getRank();
  SmallVector<Value> newExtraDims;
  SmallVector<AffineExpr> expandSrcDims =
      llvm::map_to_vector(llvm::seq<int64_t>(numDims), [&](int64_t dim) {
        return rewriter.getAffineDimExpr(dim);
      });
  SmallVector<AffineExpr> expandResultDims = llvm::map_to_vector(
      expandShapeOp.getMixedOutputShape(), [&](OpFoldResult size) {
        auto constSize = getConstantIntValue(size);
        if (constSize.has_value()) {
          return rewriter.getAffineConstantExpr(constSize.value());
        }
        newExtraDims.push_back(cast<Value>(size));
        return rewriter.getAffineDimExpr(numDims++);
      });

  // Compute the dim replacements for the AffineDimExpr corresponding to each
  // input dimension of the mapScatterOp.
  int64_t inputRank = mapScatterOp.getInput().getType().getRank();
  SmallVector<AffineExpr> inputDimReplacements(inputRank);
  SmallVector<ReassociationIndices> reassociations =
      expandShapeOp.getReassociationIndices();
  for (auto [groupIdx, group] : llvm::enumerate(reassociations)) {
    AffineExpr stride = rewriter.getAffineConstantExpr(1);
    for (int64_t dim : llvm::reverse(group)) {
      inputDimReplacements[dim] =
          expandSrcDims[groupIdx].floorDiv(stride) % expandResultDims[dim];
      stride = stride * expandResultDims[dim];
    }
  }
  rewriter.modifyOpInPlace(mapScatterOp, [&]() {
    mapScatterOp.replaceInputDims(inputDimReplacements, expandSrcType.getRank(),
                                  newExtraDims);
    mapScatterOp->setOperand(0, expandShapeOp.getSrc());
  });
  return mapScatterOp;
}

static IREE::LinalgExt::MapScatterOp
foldCollapseShapeIntoMapScatter(RewriterBase &rewriter,
                                tensor::CollapseShapeOp collapseShapeOp,
                                IREE::LinalgExt::MapScatterOp mapScatterOp) {
  assert(mapScatterOp.getInput() == collapseShapeOp->getResult(0) &&
         "expected collapseShapeOp to be the producer of mapScatterOp");

  ShapedType collapseSrcType = collapseShapeOp.getSrcType();
  int64_t numDims = collapseSrcType.getRank();
  SmallVector<AffineExpr> collapseSrcDims =
      llvm::map_to_vector(llvm::seq<int64_t>(numDims), [&](int64_t dim) {
        return rewriter.getAffineDimExpr(dim);
      });
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(collapseShapeOp);
  SmallVector<Value> newExtraDims;
  SmallVector<OpFoldResult> collapseSrcMixedSizes = tensor::getMixedSizes(
      rewriter, collapseShapeOp->getLoc(), collapseShapeOp.getSrc());
  SmallVector<AffineExpr> collapseSrcSizeDims =
      llvm::map_to_vector(collapseSrcMixedSizes, [&](OpFoldResult size) {
        auto constSize = getConstantIntValue(size);
        if (constSize.has_value()) {
          return rewriter.getAffineConstantExpr(constSize.value());
        }
        newExtraDims.push_back(cast<Value>(size));
        return rewriter.getAffineDimExpr(numDims++);
      });

  int64_t inputRank = mapScatterOp.getInput().getType().getRank();
  SmallVector<AffineExpr> inputDimReplacements(inputRank);
  SmallVector<ReassociationIndices> reassociations =
      collapseShapeOp.getReassociationIndices();
  for (auto [groupIdx, group] : llvm::enumerate(reassociations)) {
    AffineExpr stride = rewriter.getAffineConstantExpr(1);
    inputDimReplacements[groupIdx] = rewriter.getAffineConstantExpr(0);
    for (int64_t dim : group) {
      inputDimReplacements[groupIdx] =
          inputDimReplacements[groupIdx] + stride * collapseSrcDims[dim];
      stride = stride * collapseSrcSizeDims[dim];
    }
  }
  rewriter.modifyOpInPlace(mapScatterOp, [&]() {
    mapScatterOp.replaceInputDims(inputDimReplacements,
                                  collapseSrcType.getRank(), newExtraDims);
    mapScatterOp->setOperand(0, collapseShapeOp.getSrc());
  });
  return mapScatterOp;
}

static FailureOr<IREE::LinalgExt::MapScatterOp>
foldExtractSliceIntoMapScatter(RewriterBase &rewriter,
                               tensor::ExtractSliceOp extractSliceOp,
                               IREE::LinalgExt::MapScatterOp mapScatterOp) {
  assert(mapScatterOp.getInput() == extractSliceOp->getResult(0) &&
         "expected extractSliceOp to be the producer of mapScatterOp");
  if (mapScatterOp.getBoundsMap().has_value()) {
    return rewriter.notifyMatchFailure(mapScatterOp,
                                       "map_scatter already has bounds");
  }
  if (extractSliceOp.getSourceType().getRank() !=
      extractSliceOp.getResultType().getRank()) {
    return rewriter.notifyMatchFailure(
        extractSliceOp, "rank reducing extract_slice op is not supported");
  }
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(extractSliceOp);
  SmallVector<Value> sliceSizes = getValueOrCreateConstantIndexOp(
      rewriter, extractSliceOp->getLoc(), extractSliceOp.getMixedSizes());
  AffineMap boundsMap =
      rewriter.getMultiDimIdentityMap(extractSliceOp.getResultType().getRank());
  rewriter.modifyOpInPlace(mapScatterOp, [&]() {
    mapScatterOp.setBoundsMap(boundsMap);
    mapScatterOp.getBoundsMutable().append(sliceSizes);
    mapScatterOp.setStaticBounds(extractSliceOp.getStaticSizes());
    mapScatterOp->setOperand(0, extractSliceOp.getSource());
  });
  return mapScatterOp;
}

static FailureOr<IREE::LinalgExt::MapScatterOp>
foldIntoMapScatter(RewriterBase &rewriter, Operation *op,
                   IREE::LinalgExt::MapScatterOp mapScatterOp) {
  if (isa<linalg::CopyOp>(op)) {
    return foldIdentityLikeOpIntoMapScatter(rewriter, op, mapScatterOp);
  }
  if (auto padOp = dyn_cast<tensor::PadOp>(op)) {
    return foldIdentityLikeOpIntoMapScatter(rewriter, padOp, mapScatterOp);
  }
  if (auto transposeOp = dyn_cast<linalg::TransposeOp>(op)) {
    return foldTransposeIntoMapScatter(rewriter, transposeOp, mapScatterOp);
  }
  if (auto expandShapeOp = dyn_cast<tensor::ExpandShapeOp>(op)) {
    return foldExpandShapeIntoMapScatter(rewriter, expandShapeOp, mapScatterOp);
  }
  if (auto collapseShapeOp = dyn_cast<tensor::CollapseShapeOp>(op)) {
    return foldCollapseShapeIntoMapScatter(rewriter, collapseShapeOp,
                                           mapScatterOp);
  }
  if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
    return foldExtractSliceIntoMapScatter(rewriter, extractSliceOp,
                                          mapScatterOp);
  }
  return failure();
}

static void combineRelayoutOpChain(RewriterBase &rewriter, OpOperand &root) {
  Operation *rootOp = root.get().getDefiningOp();
  if (!rootOp) {
    return;
  }
  auto rootTensorType = dyn_cast<RankedTensorType>(root.get().getType());
  if (!rootTensorType) {
    return;
  }
  Location loc = rootOp->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfterValue(root.get());
  auto relayoutDest =
      rewriter
          .create<tensor::EmptyOp>(
              loc, tensor::getMixedSizes(rewriter, loc, root.get()),
              rootTensorType.getElementType())
          .getResult();
  auto combinedRelayoutOp = rewriter.create<IREE::LinalgExt::MapScatterOp>(
      loc, rootOp->getResult(0), relayoutDest);
  rewriter.replaceUsesWithIf(root.get(), combinedRelayoutOp.getResult(0),
                             [&](OpOperand &use) { return root == use; });
  while (rootOp) {
    FailureOr<IREE::LinalgExt::MapScatterOp> maybeCombinedRelayoutOp =
        foldIntoMapScatter(rewriter, rootOp, combinedRelayoutOp);
    if (failed(maybeCombinedRelayoutOp)) {
      break;
    }
    combinedRelayoutOp = maybeCombinedRelayoutOp.value();
    rootOp = combinedRelayoutOp.getInput().getDefiningOp();
  }
}

namespace {

struct CombineLayoutTransformationPass final
    : impl::CombineLayoutTransformationPassBase<
          CombineLayoutTransformationPass> {
  using impl::CombineLayoutTransformationPassBase<
      CombineLayoutTransformationPass>::CombineLayoutTransformationPassBase;

  void runOnOperation() override {
    auto funcOp = getOperation();
    SmallVector<IREE::Flow::DispatchTensorStoreOp> dispatchResults(
        funcOp.getFunctionBody().getOps<IREE::Flow::DispatchTensorStoreOp>());
    IRRewriter rewriter(&getContext());
    for (IREE::Flow::DispatchTensorStoreOp dispatchResult : dispatchResults) {
      combineRelayoutOpChain(rewriter, dispatchResult.getValueMutable());
    }

    RewritePatternSet cleanupPatterns(&getContext());
    memref::populateResolveRankedShapedTypeResultDimsPatterns(cleanupPatterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(cleanupPatterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
