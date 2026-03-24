// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-rocdl-buffer-instructions-optimization"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_ROCDLBUFFERINSTRUCTIONSOPTIMIZATIONPASS
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Simplify masked buffer reads/loads with broadcast mask.
//
// When a vector.transfer_read or vector.maskedload from a fat_raw_buffer has
// a mask that is vector.broadcast(%scalar_i1), replace with an unmasked
// read/load + arith.select. If the mask is always true, just return the
// unmasked read/load directly.
//===----------------------------------------------------------------------===//

/// Check if a value is a vector.broadcast of a scalar i1. If so, return
/// the scalar source.
static Value getBroadcastScalarI1(Value mask) {
  auto broadcastOp = mask.getDefiningOp<vector::BroadcastOp>();
  if (!broadcastOp) {
    return nullptr;
  }
  Value source = broadcastOp.getSource();
  if (!source.getType().isInteger(1)) {
    return nullptr;
  }
  return source;
}

/// Check if a scalar i1 value is a constant true.
static bool isConstantTrue(Value scalarI1) {
  return matchPattern(scalarI1, m_One());
}

/// Find the insertion point for the select op that maximizes the distance
/// from the load. Returns an iterator right before the earliest user (or
/// ancestor of a user) of |result| in |defBlock|. If any user cannot be
/// resolved to an ancestor in |defBlock|, returns std::nullopt and the caller
/// should fall back to the default insertion point (right after the load).
static std::optional<Block::iterator>
findSelectInsertionPoint(Value result, Block *defBlock) {
  Operation *earliestUser = nullptr;
  for (Operation *user : result.getUsers()) {
    Operation *candidate;
    if (user->getBlock() == defBlock) {
      candidate = user;
    } else {
      candidate = defBlock->findAncestorOpInBlock(*user);
      if (!candidate) {
        return std::nullopt;
      }
    }
    if (!earliestUser || candidate->isBeforeInBlock(earliestUser)) {
      earliestUser = candidate;
    }
  }
  if (!earliestUser) {
    return std::nullopt;
  }
  return earliestUser->getIterator();
}

/// Pattern to simplify a masked vector.transfer_read from a fat_raw_buffer.
struct SimplifyMaskedTransferRead
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    // Must have a mask.
    Value mask = readOp.getMask();
    if (!mask) {
      return failure();
    }

    // Mask must be vector.broadcast of scalar i1.
    Value scalarMask = getBroadcastScalarI1(mask);
    if (!scalarMask) {
      return failure();
    }

    // Source must be fat_raw_buffer.
    auto sourceType = dyn_cast<MemRefType>(readOp.getBase().getType());
    if (!sourceType || !hasAMDGPUFatRawBufferAddressSpace(sourceType)) {
      return failure();
    }

    // Must be fully in_bounds.
    SmallVector<bool> inBounds = readOp.getInBoundsValues();
    if (llvm::any_of(inBounds, [](bool b) { return !b; })) {
      return failure();
    }

    Location loc = readOp.getLoc();

    // Create unmasked read, preserving the original permutation map.
    auto newReadOp = vector::TransferReadOp::create(
        rewriter, loc, readOp.getVectorType(), readOp.getBase(),
        readOp.getIndices(), readOp.getPadding(), readOp.getPermutationMap(),
        ArrayRef<bool>(inBounds));

    if (isConstantTrue(scalarMask)) {
      // Always-true: just use the unmasked read directly.
      rewriter.replaceOp(readOp, newReadOp);
    } else {
      // Place the select as late as possible (right before the earliest user)
      // to maximize distance from the load for latency hiding.
      auto insertPt =
          findSelectInsertionPoint(readOp.getResult(), readOp->getBlock());
      if (insertPt) {
        rewriter.setInsertionPoint(readOp->getBlock(), *insertPt);
      }
      auto paddingBroadcast = vector::BroadcastOp::create(
          rewriter, loc, readOp.getVectorType(), readOp.getPadding());
      auto selectOp = arith::SelectOp::create(rewriter, loc, scalarMask,
                                              newReadOp, paddingBroadcast);
      rewriter.replaceOp(readOp, selectOp);
    }
    return success();
  }
};

/// Pattern to simplify a masked vector.maskedload from a fat_raw_buffer.
struct SimplifyMaskedLoad : public OpRewritePattern<vector::MaskedLoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MaskedLoadOp maskedLoadOp,
                                PatternRewriter &rewriter) const override {
    // Mask must be vector.broadcast of scalar i1.
    Value scalarMask = getBroadcastScalarI1(maskedLoadOp.getMask());
    if (!scalarMask) {
      return failure();
    }

    // Source must be fat_raw_buffer.
    auto sourceType = dyn_cast<MemRefType>(maskedLoadOp.getBase().getType());
    if (!sourceType || !hasAMDGPUFatRawBufferAddressSpace(sourceType)) {
      return failure();
    }

    Location loc = maskedLoadOp.getLoc();

    // Create unmasked vector.load.
    auto loadOp = vector::LoadOp::create(
        rewriter, loc, maskedLoadOp.getResult().getType(),
        maskedLoadOp.getBase(), maskedLoadOp.getIndices());

    if (isConstantTrue(scalarMask)) {
      // Always-true: just use the unmasked load directly.
      rewriter.replaceOp(maskedLoadOp, loadOp);
    } else {
      // Place the select as late as possible (right before the earliest user)
      // to maximize distance from the load for latency hiding.
      auto insertPt = findSelectInsertionPoint(maskedLoadOp.getResult(),
                                               maskedLoadOp->getBlock());
      if (insertPt) {
        rewriter.setInsertionPoint(maskedLoadOp->getBlock(), *insertPt);
      }
      auto selectOp = arith::SelectOp::create(rewriter, loc, scalarMask,
                                              loadOp,
                                              maskedLoadOp.getPassThru());
      rewriter.replaceOp(maskedLoadOp, selectOp);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct ROCDLBufferInstructionsOptimizationPass final
    : impl::ROCDLBufferInstructionsOptimizationPassBase<
          ROCDLBufferInstructionsOptimizationPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // Sink vector.broadcast through elementwise ops so that masks like
    // arith.andi(broadcast(a), broadcast(b)) become broadcast(arith.andi(a,b)).
    // This exposes scalar i1 broadcasts for the mask simplification patterns.
    vector::populateSinkVectorOpsPatterns(patterns);

    // Simplify masked buffer reads/loads when the mask is a broadcast of a
    // scalar i1.
    patterns.add<SimplifyMaskedTransferRead, SimplifyMaskedLoad>(context);

    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
