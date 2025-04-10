// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-combine-layout-transformation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_COMBINELAYOUTTRANSFORMATIONPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Simplifying Complex Ops
//===----------------------------------------------------------------------===//

/// Convert complex ops into simpler ops by decomposing or raising to a named
/// op. PackOps and UnPackOps are decomposed, and transpose GenericOps are
/// raised to linalg::TransposeOps.
static void simplifyComplexRelayoutOps(RewriterBase &rewriter,
                                       FunctionOpInterface funcOp) {
  OpBuilder::InsertionGuard g(rewriter);
  SmallVector<linalg::PackOp> packOps(
      funcOp.getFunctionBody().getOps<linalg::PackOp>());
  for (auto packOp : packOps) {
    rewriter.setInsertionPoint(packOp);
    (void)linalg::lowerPack(rewriter, packOp,
                            /*lowerPadLikeWithInsertSlice=*/false);
  }
  SmallVector<linalg::UnPackOp> unPackOps(
      funcOp.getFunctionBody().getOps<linalg::UnPackOp>());
  for (auto unPackOp : unPackOps) {
    rewriter.setInsertionPoint(unPackOp);
    (void)linalg::lowerUnPack(rewriter, unPackOp,
                              /*lowerUnpadLikeWithExtractSlice=*/false);
  }
  SmallVector<linalg::GenericOp> genericOps(
      funcOp.getFunctionBody().getOps<linalg::GenericOp>());
  for (auto genericOp : genericOps) {
    if (linalg::isaTransposeOpInterface(genericOp)) {
      rewriter.setInsertionPoint(genericOp);
      (void)linalg::specializeGenericOp(rewriter, genericOp);
    }
  }
}

//===----------------------------------------------------------------------===//
// Combining Layout Transformation Ops
//===----------------------------------------------------------------------===//

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

  ArrayRef<int64_t> perm = transposeOp.getPermutation();
  auto indexTransformBuilder =
      [&](ArrayRef<BlockArgument> srcIndices) -> SmallVector<Value> {
    SmallVector<Value> indexValues(srcIndices);
    return applyPermutation(indexValues, perm);
  };
  rewriter.modifyOpInPlace(mapScatterOp, [&]() {
    mapScatterOp.insertTransformationAtStart(rewriter, indexTransformBuilder,
                                             perm.size());
    mapScatterOp->setOperand(0, transposeOp.getInput());
  });
  return mapScatterOp;
}

static IREE::LinalgExt::MapScatterOp
foldReshapeIntoMapScatter(RewriterBase &rewriter, Operation *reshapeOp,
                          SmallVector<OpFoldResult> srcDims,
                          SmallVector<OpFoldResult> resultDims,
                          IREE::LinalgExt::MapScatterOp mapScatterOp) {
  assert(mapScatterOp.getInput() == reshapeOp->getResult(0) &&
         "expected reshapeOp to be the producer of mapScatterOp");

  auto indexTransformBuilder =
      [&](ArrayRef<BlockArgument> srcIndices) -> SmallVector<Value> {
    auto linearizeIndexOp = rewriter.create<affine::AffineLinearizeIndexOp>(
        mapScatterOp->getLoc(), srcIndices, srcDims);
    auto delinearizeIndexOp = rewriter.create<affine::AffineDelinearizeIndexOp>(
        mapScatterOp->getLoc(), linearizeIndexOp.getResult(), resultDims);
    return delinearizeIndexOp->getResults();
  };
  rewriter.modifyOpInPlace(mapScatterOp, [&]() {
    mapScatterOp.insertTransformationAtStart(rewriter, indexTransformBuilder,
                                             srcDims.size());
    mapScatterOp->setOperand(0, reshapeOp->getOperand(0));
  });
  return mapScatterOp;
}

static IREE::LinalgExt::MapScatterOp
foldCollapseShapeIntoMapScatter(RewriterBase &rewriter,
                                tensor::CollapseShapeOp collapseShapeOp,
                                IREE::LinalgExt::MapScatterOp mapScatterOp) {
  assert(mapScatterOp.getInput() == collapseShapeOp->getResult(0) &&
         "expected collapseShapeOp to be the producer of mapScatterOp");
  Location loc = collapseShapeOp->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(collapseShapeOp);
  SmallVector<OpFoldResult> mixedSrcSizes =
      tensor::getMixedSizes(rewriter, loc, collapseShapeOp.getSrc());

  auto prod = [&](ArrayRef<OpFoldResult> vals) -> OpFoldResult {
    AffineExpr prodExpr = rewriter.getAffineConstantExpr(1);
    for (auto [idx, ofr] : llvm::enumerate(vals)) {
      AffineExpr d = rewriter.getAffineDimExpr(idx);
      prodExpr = prodExpr * d;
    }
    auto prodMap =
        AffineMap::get(vals.size(), 0, prodExpr, rewriter.getContext());
    return affine::makeComposedFoldedAffineApply(rewriter, loc, prodMap, vals);
  };
  SmallVector<OpFoldResult> mixedResultSizes;
  for (ReassociationIndices group : collapseShapeOp.getReassociationIndices()) {
    if (group.size() == 1) {
      mixedResultSizes.push_back(mixedSrcSizes[group[0]]);
      continue;
    }
    SmallVector<OpFoldResult> groupSizes = llvm::map_to_vector(
        group, [&](int64_t idx) { return mixedSrcSizes[idx]; });
    mixedResultSizes.push_back(prod(groupSizes));
  }
  return foldReshapeIntoMapScatter(rewriter, collapseShapeOp, mixedSrcSizes,
                                   mixedResultSizes, mapScatterOp);
}

static IREE::LinalgExt::MapScatterOp
foldExpandShapeIntoMapScatter(RewriterBase &rewriter,
                              tensor::ExpandShapeOp expandShapeOp,
                              IREE::LinalgExt::MapScatterOp mapScatterOp) {
  assert(mapScatterOp.getInput() == expandShapeOp->getResult(0) &&
         "expected expandShapeOp to be the producer of mapScatterOp");
  Location loc = expandShapeOp->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(expandShapeOp);
  SmallVector<OpFoldResult> srcMixedSizes =
      tensor::getMixedSizes(rewriter, loc, expandShapeOp.getSrc());
  return foldReshapeIntoMapScatter(rewriter, expandShapeOp, srcMixedSizes,
                                   expandShapeOp.getMixedOutputShape(),
                                   mapScatterOp);
}

static FailureOr<IREE::LinalgExt::MapScatterOp>
foldExtractSliceIntoMapScatter(RewriterBase &rewriter,
                               tensor::ExtractSliceOp extractSliceOp,
                               IREE::LinalgExt::MapScatterOp mapScatterOp) {
  assert(mapScatterOp.getInput() == extractSliceOp->getResult(0) &&
         "expected extractSliceOp to be the producer of mapScatterOp");
  if (extractSliceOp.getSourceType().getRank() !=
      extractSliceOp.getResultType().getRank()) {
    return rewriter.notifyMatchFailure(
        extractSliceOp, "rank reducing extract_slice op is not supported");
  }
  SmallVector<OpFoldResult> bounds(extractSliceOp.getMixedSizes());
  Block &transformBody = mapScatterOp.getTransformationRegion().front();
  auto yieldOp = cast<IREE::LinalgExt::YieldOp>(transformBody.getTerminator());
  Value mask = yieldOp->getOperands().back();

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(yieldOp);
  Location loc = mapScatterOp->getLoc();
  ArrayRef<BlockArgument> srcIndices = transformBody.getArguments();
  for (auto [bound, srcIdx] : llvm::zip_equal(bounds, srcIndices)) {
    Value boundValue = getValueOrCreateConstantIndexOp(rewriter, loc, bound);
    auto isOutOfBounds =
        rewriter
            .create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, srcIdx,
                                   boundValue)
            ->getResult(0);
    mask = rewriter.create<arith::AndIOp>(loc, mask, isOutOfBounds);
  }
  rewriter.modifyOpInPlace(yieldOp, [&]() {
    yieldOp->setOperand(yieldOp->getNumOperands() - 1, mask);
  });
  rewriter.modifyOpInPlace(mapScatterOp, [&]() {
    mapScatterOp->setOperand(0, extractSliceOp.getSource());
  });
  return mapScatterOp;
}

/// Fold the `op` into the `mapScatterOp`, if possible. The resulting
/// map_scatter op is returned, if the `op` was folded. Otherwise, return
/// failure.
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

/// Starting from the `root`, iteratively combine any relayout op profucers
/// into a single iree_linalg_ext.map_scatter op. An identity map_scatter op
/// is inserted before the root, and then the producers of the map_scatter op
/// are folded into the map_scatter until an unsupported op is reached.
static IREE::LinalgExt::MapScatterOp
combineRelayoutOpChain(RewriterBase &rewriter,
                       IREE::LinalgExt::MapScatterOp mapScatterOp) {
  Operation *relayoutOp = mapScatterOp.getInput().getDefiningOp();
  if (!relayoutOp) {
    return mapScatterOp;
  }
  IREE::LinalgExt::MapScatterOp combinedRelayoutOp = mapScatterOp;
  while (relayoutOp) {
    LDBG("Attempting to fold " << relayoutOp->getName()
                               << " into map_scatter op:\n"
                               << *relayoutOp << "\n");
    FailureOr<IREE::LinalgExt::MapScatterOp> maybeCombinedRelayoutOp =
        foldIntoMapScatter(rewriter, relayoutOp, combinedRelayoutOp);
    if (failed(maybeCombinedRelayoutOp)) {
      LDBG("Failed to fold " << relayoutOp->getName()
                             << " into map_scatter op");
      break;
    }
    combinedRelayoutOp = maybeCombinedRelayoutOp.value();
    LDBG("Successfully folded " << relayoutOp->getName()
                                << " into map_scatter. New map_scatter op:\n"
                                << combinedRelayoutOp << "\n");
    relayoutOp = combinedRelayoutOp.getInput().getDefiningOp();
  }
  return combinedRelayoutOp;
}

static IREE::LinalgExt::MapScatterOp
insertIdentityMapScatter(RewriterBase &rewriter,
                         IREE::Codegen::StoreToMemrefOp storeOp) {
  Location loc = storeOp->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(storeOp);
  auto mapScatterDest =
      rewriter
          .create<tensor::EmptyOp>(
              loc, memref::getMixedSizes(rewriter, loc, storeOp.getTarget()),
              storeOp.getValue().getType().getElementType())
          .getResult();
  auto mapScatterOp = rewriter.create<IREE::LinalgExt::MapScatterOp>(
      loc, storeOp.getValue(), mapScatterDest);
  rewriter.modifyOpInPlace(
      storeOp, [&]() { storeOp->setOperand(0, mapScatterOp.getResult(0)); });
  LDBG("Created identity map_scatter:\n" << mapScatterOp << "\n");
  return mapScatterOp;
}

namespace {

struct CombineLayoutTransformationPass final
    : impl::CombineLayoutTransformationPassBase<
          CombineLayoutTransformationPass> {
  using impl::CombineLayoutTransformationPassBase<
      CombineLayoutTransformationPass>::CombineLayoutTransformationPassBase;

  void runOnOperation() override {
    auto funcOp = getOperation();

    // Apply some preprocessing to convert complex layout transformation
    // ops like pack and unpack into simpler supported ops.
    IRRewriter rewriter(&getContext());
    simplifyComplexRelayoutOps(rewriter, funcOp);

    // Start from iree_codegen.store_to_memref ops, and combine producer
    // relayout ops into a single map_scatter.
    SmallVector<IREE::Codegen::StoreToMemrefOp> dispatchResults(
        funcOp.getFunctionBody().getOps<IREE::Codegen::StoreToMemrefOp>());
    for (IREE::Codegen::StoreToMemrefOp dispatchResult : dispatchResults) {
      IREE::LinalgExt::MapScatterOp mapScatterOp =
          insertIdentityMapScatter(rewriter, dispatchResult);
      IREE::LinalgExt::MapScatterOp combinedRelayoutOp =
          combineRelayoutOpChain(rewriter, mapScatterOp);
      (void)combinedRelayoutOp;
      // // If no relayout ops were folded into the map_scatter, then remove it,
      // // since it will just be an identity transformation.
      // if (combinedRelayoutOp.getTransformations().empty()) {
      //   LDBG(
      //       "No relayout ops were combined. Removing identity map_scatter
      //       op.");
      //   rewriter.replaceOp(combinedRelayoutOp,
      //   combinedRelayoutOp.getInput());
      // }
    }

    // Cleanup any tensor.dim ops that may be present after relayout
    // combination.
    RewritePatternSet cleanupPatterns(&getContext());
    memref::populateResolveRankedShapedTypeResultDimsPatterns(cleanupPatterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(cleanupPatterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
