// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_PROPAGATERESHAPESBYEXPANSIONPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

// Indicates whether the given linalg op represents a transpose. In particular,
// it requires a single input where the indexing maps are full permutations and
// non-equal.
static bool isaTransposeOpInterface(linalg::LinalgOp linalgOp) {
  if (linalgOp.getNumParallelLoops() != linalgOp.getNumLoops())
    return false;

  if (linalgOp.getNumDpsInputs() != 1 || linalgOp.getNumDpsInits() != 1)
    return false;
  auto mapRange = linalgOp.getIndexingMapsArray();
  if (mapRange.size() != 2 || !mapRange.front().isPermutation() ||
      !mapRange.back().isPermutation() || mapRange.front() == mapRange.back()) {
    return false;
  }
  return llvm::hasSingleElement(linalgOp.getBlock()->getOperations());
}

struct RaiseLinalgTransposeGeneric final
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!isaTransposeOpInterface(genericOp)) {
      return failure();
    }

    auto mapRange = genericOp.getIndexingMapsArray();
    AffineMap outMap = mapRange.back();
    AffineMap inMap = mapRange.front();
    SmallVector<int64_t> perm;
    // To get the permutation, look at each output index and find which
    // dimension in the input we're reading from for that index.
    for (AffineExpr expr : outMap.getResults()) {
      perm.push_back(*inMap.getResultPosition(expr));
    }
    rewriter.replaceOpWithNewOp<linalg::TransposeOp>(
        genericOp, genericOp.getDpsInputs()[0], genericOp.getDpsInits()[0], perm);
    return success();
  }
};

struct PropagateReshapesByExpansionPass final
    : impl::PropagateReshapesByExpansionPassBase<
          PropagateReshapesByExpansionPass> {
  void runOnOperation() override;
};
} // namespace

void PropagateReshapesByExpansionPass::runOnOperation() {
  MLIRContext *context = &getContext();

  {
    RewritePatternSet patterns(context);
    // Preemptively attempt to fold any reshapes into interface bindings if
    // possible to simplify subsequent reshape propagation.
    populateReshapeToInterfaceTensorPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  RewritePatternSet bubbleExpandShapePatterns(context);
  linalg::ControlFusionFn bubbleUpExpansionControlFn =
      [](OpOperand *fusedOperand) {
        Operation *producer = fusedOperand->get().getDefiningOp();
        Operation *consumer = fusedOperand->getOwner();

        // Block only if one of the operations has a lowering configuration
        // which means it likely expects tiling specific to its original shape.
        if (getLoweringConfig(producer) || getLoweringConfig(consumer)) {
          return false;
        }
        return true;
      };
  linalg::populateFoldReshapeOpsByExpansionPatterns(bubbleExpandShapePatterns,
                                                    bubbleUpExpansionControlFn);
  // Add patterns to do some additional cleanup (on top of canonicalizations
  // that can be done later) of reshape ops.
  tensor::populateFoldTensorEmptyPatterns(bubbleExpandShapePatterns);
  linalg::FillOp::getCanonicalizationPatterns(bubbleExpandShapePatterns,
                                              context);
  tensor::CollapseShapeOp::getCanonicalizationPatterns(
      bubbleExpandShapePatterns, context);
  tensor::EmptyOp::getCanonicalizationPatterns(bubbleExpandShapePatterns,
                                               context);
  tensor::ExpandShapeOp::getCanonicalizationPatterns(bubbleExpandShapePatterns,
                                                     context);
  populateReshapeToInterfaceTensorPatterns(bubbleExpandShapePatterns);

  bubbleExpandShapePatterns.add<RaiseLinalgTransposeGeneric>(context);
  linalg::TransposeOp::getCanonicalizationPatterns(
      bubbleExpandShapePatterns, context);
  linalg::GenericOp::getCanonicalizationPatterns(
      bubbleExpandShapePatterns, context);

  if (failed(applyPatternsAndFoldGreedily(
          getOperation(), std::move(bubbleExpandShapePatterns)))) {
    getOperation()->emitOpError("Failed to propagate reshapes");
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler
