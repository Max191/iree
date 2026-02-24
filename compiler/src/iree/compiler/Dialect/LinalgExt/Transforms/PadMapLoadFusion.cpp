// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

namespace {

/// Fold a `tensor.pad` producer into a consumer `iree_linalg_ext.map_load`.
///
/// Matches:
///   %padded = tensor.pad %source low[...] high[...] { yield %pad_val }
///   %result = iree_linalg_ext.map_load %padded into %output { ... }
///
/// Rewrites to a map_load that reads from the unpadded source directly,
/// with adjusted source indices (subtract low_pad offsets) and the pad's
/// constant value replacing the poison padding value.
struct FoldPadProducerIntoMapLoad
    : public OpRewritePattern<LinalgExt::MapLoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LinalgExt::MapLoadOp mapLoadOp,
                                PatternRewriter &rewriter) const override {
    // Check that the source is produced by a tensor.pad.
    auto padOp = mapLoadOp.getSource().getDefiningOp<tensor::PadOp>();
    if (!padOp)
      return failure();

    // Only support constant pad values.
    Value padValue = padOp.getConstantPaddingValue();
    if (!padValue) {
      return rewriter.notifyMatchFailure(
          mapLoadOp, "pad does not have a constant padding value");
    }

    // Check if the map_load already has a real (non-poison) padding value.
    // If so, we cannot safely replace it.
    Value currentPadValue = mapLoadOp.getPaddingValue();
    if (!currentPadValue.getDefiningOp<ub::PoisonOp>()) {
      return rewriter.notifyMatchFailure(
          mapLoadOp,
          "map_load already has a non-poison padding value; folding "
          "pad would overwrite it");
    }

    // Single use check — we replace the map_load and want the pad to be dead.
    if (!padOp.getResult().hasOneUse()) {
      return rewriter.notifyMatchFailure(
          mapLoadOp, "pad has multiple uses; cannot fold");
    }

    Location loc = mapLoadOp->getLoc();

    // Create a new map_load reading from the unpadded source.
    Value unpaddedSource = padOp.getSource();
    Value output = mapLoadOp.getOutput();
    auto newMapLoad = MapLoadOp::create(rewriter, loc, output.getType(),
                                        unpaddedSource, output);

    // Clone the existing transformation region.
    rewriter.cloneRegionBefore(mapLoadOp.getTransformationRegion(),
                               newMapLoad.getTransformationRegion(),
                               newMapLoad.getTransformationRegion().begin());

    // Modify the yield: subtract low_pad from each yielded source index
    // and replace the padding value.
    Block &transformBody = newMapLoad.getTransformationRegion().front();
    auto yieldOp =
        cast<LinalgExt::YieldOp>(transformBody.getTerminator());

    SmallVector<OpFoldResult> lowPad = padOp.getMixedLowPad();
    int64_t sourceRank = newMapLoad.getSourceRank();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(yieldOp);

    SmallVector<Value> newYieldOperands;
    for (int64_t i = 0; i < sourceRank; ++i) {
      Value oldIdx = yieldOp.getOperand(i);
      Value lowValue =
          getValueOrCreateConstantIndexOp(rewriter, loc, lowPad[i]);
      Value adjustedIdx = arith::SubIOp::create(
          rewriter, loc, oldIdx, lowValue, arith::IntegerOverflowFlags::nsw);
      newYieldOperands.push_back(adjustedIdx);
    }
    // Append the pad constant value (replacing the poison padding).
    newYieldOperands.push_back(padValue);

    // Replace the yield with new operands.
    rewriter.replaceOpWithNewOp<LinalgExt::YieldOp>(yieldOp, newYieldOperands);

    // Replace the old map_load with the new one and erase the pad if dead.
    rewriter.replaceOp(mapLoadOp, newMapLoad.getResult(0));
    if (padOp->use_empty())
      rewriter.eraseOp(padOp);

    return success();
  }
};

} // namespace

void populateFoldPadIntoMapLoadPatterns(RewritePatternSet &patterns) {
  patterns.add<FoldPadProducerIntoMapLoad>(patterns.getContext());
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
