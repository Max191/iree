// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
  auto transposeTransformAttr = IREE::LinalgExt::TransposeIndicesAttr::get(
      transposeOp->getContext(), transposeOp.getPermutation());
  rewriter.modifyOpInPlace(mapScatterOp, [&]() {
    mapScatterOp.insertTransformation(0, transposeTransformAttr);
    mapScatterOp->setOperand(0, transposeOp.getInput());
  });
  return mapScatterOp;
}

static IREE::LinalgExt::MapScatterOp
foldReshapeIntoMapScatter(RewriterBase &rewriter, Operation *reshapeOp,
                          SmallVector<Value> srcDynamicDims,
                          SmallVector<Value> resultDynamicDims,
                          IREE::LinalgExt::MapScatterOp mapScatterOp) {
  assert(mapScatterOp.getInput() == reshapeOp->getResult(0) &&
         "expected reshapeOp to be the producer of mapScatterOp");
  ArrayRef<int64_t> staticLinearizeBasis =
      cast<ShapedType>(reshapeOp->getOperandTypes()[0]).getShape();
  auto linearizeAttr = IREE::LinalgExt::LinearizeIndicesAttr::get(
      reshapeOp->getContext(), staticLinearizeBasis);
  assert(linearizeAttr.getNumDynamicIndices() == srcDynamicDims.size() &&
         "expected number of dims in srcDynamicDims to equal the number of "
         "dynamic dims in the reshape source operand.");

  ArrayRef<int64_t> staticDelinearizeBasis =
      cast<ShapedType>(reshapeOp->getResultTypes()[0]).getShape();
  auto delinearizeAttr = IREE::LinalgExt::DelinearizeIndicesAttr::get(
      reshapeOp->getContext(), staticDelinearizeBasis);
  assert(delinearizeAttr.getNumDynamicIndices() == resultDynamicDims.size() &&
         "expected number of dims in resultDynamicDims to equal the number of "
         "dynamic dims in the reshape result.");

  SmallVector<Value> newCapturedDynamicIndices = srcDynamicDims;
  newCapturedDynamicIndices.append(resultDynamicDims);
  newCapturedDynamicIndices.append(
      mapScatterOp.getCapturedDynamicIndices().begin(),
      mapScatterOp.getCapturedDynamicIndices().end());
  rewriter.modifyOpInPlace(mapScatterOp, [&]() {
    mapScatterOp.insertTransformation(0, delinearizeAttr);
    mapScatterOp.insertTransformation(0, linearizeAttr);
    mapScatterOp.getCapturedDynamicIndicesMutable().assign(
        newCapturedDynamicIndices);
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
  SmallVector<Value> srcDynamicSizes;
  std::tie(std::ignore, srcDynamicSizes) = decomposeMixedValues(mixedSrcSizes);

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
  SmallVector<Value> resultDynamicSizes;
  std::tie(std::ignore, resultDynamicSizes) =
      decomposeMixedValues(mixedResultSizes);
  return foldReshapeIntoMapScatter(rewriter, collapseShapeOp, srcDynamicSizes,
                                   resultDynamicSizes, mapScatterOp);
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
  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(rewriter, loc, expandShapeOp.getSrc());
  SmallVector<Value> srcDynamicSizes;
  std::tie(std::ignore, srcDynamicSizes) = decomposeMixedValues(mixedSizes);
  return foldReshapeIntoMapScatter(rewriter, expandShapeOp, srcDynamicSizes,
                                   expandShapeOp.getOutputShape(),
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
  SmallVector<int64_t> bounds;
  SmallVector<Value> dynamicSizes;
  std::tie(bounds, dynamicSizes) =
      decomposeMixedValues(extractSliceOp.getMixedSizes());
  auto clampIndicesAttr = IREE::LinalgExt::ClampIndicesAttr::get(
      extractSliceOp->getContext(), bounds);

  SmallVector<Value> newCapturedDynamicIndices = dynamicSizes;
  newCapturedDynamicIndices.append(
      mapScatterOp.getCapturedDynamicIndices().begin(),
      mapScatterOp.getCapturedDynamicIndices().end());
  rewriter.modifyOpInPlace(mapScatterOp, [&]() {
    mapScatterOp.insertTransformation(0, clampIndicesAttr);
    mapScatterOp.getCapturedDynamicIndicesMutable().assign(
        newCapturedDynamicIndices);
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
  LDBG("Created identity map_scatter:\n" << combinedRelayoutOp << "\n");
  Operation *relayoutOp = rootOp;
  while (relayoutOp) {
    LDBG("Attempting to fold " << relayoutOp->getName() <<
         " into map_scatter op:\n" << *relayoutOp << "\n");
    FailureOr<IREE::LinalgExt::MapScatterOp> maybeCombinedRelayoutOp =
        foldIntoMapScatter(rewriter, relayoutOp, combinedRelayoutOp);
    if (failed(maybeCombinedRelayoutOp)) {
      LDBG("Failed to fold " << relayoutOp->getName() <<
           " into map_scatter op");
      break;
    }
    combinedRelayoutOp = maybeCombinedRelayoutOp.value();
    LDBG("Successfully folded " << relayoutOp->getName() <<
         " into map_scatter. New map_scatter op:\n" << combinedRelayoutOp <<
         "\n");
    relayoutOp = combinedRelayoutOp.getInput().getDefiningOp();
  }
  // If no relayout ops were folded into the map_scatter, then remove it, since
  // it will just be an identity transformation.
  if (relayoutOp == rootOp) {
    LDBG("No relayout ops were combined. Removing identity map_scatter op.");
    rewriter.replaceOp(combinedRelayoutOp, combinedRelayoutOp.getInput());
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

    // Apply some preprocessing to convert complex layout transformation
    // ops like pack and unpack into simpler supported ops.
    IRRewriter rewriter(&getContext());
    simplifyComplexRelayoutOps(rewriter, funcOp);

    // Start from iree_codegen.store_to_memref ops, and combine producer
    // relayout ops into a single map_scatter.
    SmallVector<IREE::Codegen::StoreToMemrefOp> dispatchResults(
        funcOp.getFunctionBody().getOps<IREE::Codegen::StoreToMemrefOp>());
    for (IREE::Codegen::StoreToMemrefOp dispatchResult : dispatchResults) {
      combineRelayoutOpChain(rewriter, dispatchResult.getValueMutable());
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
