// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Analysis/IntegerDivisibilityAnalysis.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-linalg-ext-decompose-im2col"

namespace mlir::iree_compiler::IREE::LinalgExt {

#define GEN_PASS_DEF_DECOMPOSEIM2COLPASS
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"

static LogicalResult decomposeIm2col(Im2colOp im2colOp, RewriterBase &rewriter,
                                     bool unroll) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(im2colOp);
  FailureOr<SmallVector<Value>> decomposedIm2col =
      im2colOp.decomposeOperation(rewriter);
  if (failed(decomposedIm2col)) {
    return failure();
  }
  rewriter.replaceOp(im2colOp, decomposedIm2col.value().front());
  if (!unroll) {
    return success();
  }

  // Unroll the loop nest created by the im2col op decomposition.
  auto outerLoop = decomposedIm2col.value().front().getDefiningOp<scf::ForOp>();
  if (!outerLoop) {
    return success();
  }
  SmallVector<scf::ForOp> loopNest({outerLoop});
  while (auto innerLoop =
             outerLoop.getYieldedValues()[0].getDefiningOp<scf::ForOp>()) {
    loopNest.push_back(innerLoop);
    outerLoop = innerLoop;
  }
  for (auto loop : llvm::reverse(loopNest)) {
    std::optional<int64_t> ub = getConstantIntValue(loop.getUpperBound());
    if (!ub.has_value() || ub.value() == 1) {
      continue;
    }
    rewriter.setInsertionPoint(loop);
    if (failed(mlir::loopUnrollByFactor(loop, ub.value()))) {
      loop.emitOpError("failed to unroll loop");
      return failure();
    }
  }
  return success();
}

static uint64_t getIntegerDivisibility(OpFoldResult value,
                                       DataFlowSolver &solver) {
  if (auto constValue = getConstantIntValue(value)) {
    return constValue.value();
  }
  auto *divState = solver.lookupState<IREE::Util::IntegerDivisibilityLattice>(
      cast<Value>(value));
  if (divState && !divState->getValue().isUninitialized()) {
    return divState->getValue().getValue().udiv();
  }
  return 1;
}

static std::optional<int64_t> getUpperBound(OpFoldResult ofr) {
  if (auto constValue = getConstantIntValue(ofr)) {
    return constValue.value();
  }
  auto value = cast<Value>(ofr);
  auto ub = ValueBoundsConstraintSet::computeConstantBound(
      presburger::BoundType::UB, {value, /*dim=*/std::nullopt},
      /*stopCondition=*/nullptr, /*closedUB=*/true);
  if (failed(ub)) {
    return std::nullopt;
  }
  return ub.value();
}

// Helper method to check if a slice will be contiguous given the offset,
// slice size. This checks that `inputSize` and `offset` are both evenly
// divisible by `tileSize`.
static bool willBeContiguousSlice(OpFoldResult inputSize, OpFoldResult tileSize,
                                  OpFoldResult offset, DataFlowSolver &solver) {
  auto constInputSize = getConstantIntValue(inputSize);
  if (!constInputSize.has_value()) {
    return false;
  }
  std::optional<int64_t> tileSizeUB = getUpperBound(tileSize);
  if (!tileSizeUB.has_value()) {
    return false;
  }
  int64_t offsetDiv = getIntegerDivisibility(offset, solver);
  return *constInputSize % *tileSizeUB == 0 && offsetDiv % *tileSizeUB == 0;
}

static std::optional<int64_t>
chooseDimToVectorize(OpBuilder &b, Location loc, Im2colOp im2colOp,
                     SmallVector<Range> iterationDomain,
                     SmallVector<OpFoldResult> inputSizes, OpFoldResult kOffset,
                     DataFlowSolver &solver) {
  int64_t innerInputDim = im2colOp.getInputRank() - 1;
  SmallVector<SmallVector<int64_t>> vectorizationMap =
      im2colOp.getInputToOutputDimVectorizationMap();
  SmallVector<int64_t> vectorizableOutputDims = vectorizationMap[innerInputDim];
  if (vectorizableOutputDims.empty()) {
    return std::nullopt;
  }
  SmallVector<int64_t> kOutputDims = im2colOp.getKOutputDims();
  SetVector<int64_t> kDimSet(kOutputDims.begin(), kOutputDims.end());
  SetVector<int64_t> kPosSet(im2colOp.getKPos().begin(),
                             im2colOp.getKPos().end());
  // There may be multiple output dims that we can vectorize, so prioritize the
  // innermost dims first.
  std::sort(vectorizableOutputDims.begin(), vectorizableOutputDims.end());
  // Check each dim in order from innermost to outermost, and return the first
  // one that is vectorizable.
  while (!vectorizableOutputDims.empty()) {
    int64_t outputDimToVectorize = vectorizableOutputDims.pop_back_val();
    if (outputDimToVectorize == im2colOp.getBatchOutputDims().back()) {
      return outputDimToVectorize;
    }

    // If a K dim is being vectorized, then it is contiguous along either the
    // input channel dimension, or the filter kernel window. If it is contiguous
    // along the kernel window, then the actual inner slice size is equal to the
    // size of the corresponding kernel window dimension. Otherwise, the inner
    // slice size is just the size of the input tensor's inner dimension.
    OpFoldResult innerSliceSize = inputSizes[innerInputDim];
    if (kDimSet.contains(outputDimToVectorize)) {
      for (auto [kernelSize, mPos] :
           llvm::zip_equal(im2colOp.getMixedKernelSize(), im2colOp.getMPos())) {
        if (mPos == innerInputDim) {
          innerSliceSize = kernelSize;
        }
      }
    }

    // If the input slice is contiguous along the innermost dimension, then it
    // is vectorizable. If it is not, then move on to the next innermost dim.
    SmallVector<int64_t> mOutputDims = im2colOp.getMOutputDims();
    SetVector<int64_t> mDimSet(llvm::from_range, mOutputDims);
    OpFoldResult offset = b.getIndexAttr(0);
    if (kDimSet.contains(outputDimToVectorize)) {
      offset = kOffset;
    } else if (mDimSet.contains(outputDimToVectorize)) {
      // TODO(Max191): Support vectorization along the M dimension.
      continue;
    }
    OpFoldResult outputDimSize = iterationDomain[outputDimToVectorize].size;
    if (!willBeContiguousSlice(innerSliceSize, outputDimSize, offset, solver)) {
      continue;
    }
    return outputDimToVectorize;
  }
  return std::nullopt;
}

static std::optional<int64_t> getDimBound(Value v, int64_t dim) {
  auto ub = ValueBoundsConstraintSet::computeConstantBound(
      presburger::BoundType::UB, {v, dim},
      /*stopCondition=*/nullptr, /*closedUB=*/true);
  if (failed(ub)) {
    return std::nullopt;
  }
  return ub.value();
}

static void annotateIm2colVectorizationHints(FunctionOpInterface funcOp,
                                             DataFlowSolver &solver) {
  IRRewriter rewriter(funcOp.getContext());
  funcOp->walk([&](Im2colOp im2colOp) {
    SmallVector<Range> iterationDomain = im2colOp.getIterationDomain(rewriter);
    Location loc = im2colOp.getLoc();
    rewriter.setInsertionPoint(im2colOp);
    SmallVector<OpFoldResult> inputSizes =
        tensor::getMixedSizes(rewriter, loc, im2colOp.getInput());
    if (im2colOp.getMixedKOffset().size() > 1) {
      return;
    }
    OpFoldResult kOffset = im2colOp.getMixedKOffset()[0];
    SetVector<int64_t> dimsToVectorize;
    // We vectorize any dim that has an upper bound of 1, because they are
    // trivially contiguous.
    for (int64_t dim = 0; dim < im2colOp.getOutputRank(); ++dim) {
      std::optional<int64_t> dimBound = getDimBound(im2colOp.getOutput(), dim);
      if (!dimBound.has_value() || dimBound.value() != 1) {
        continue;
      }
      dimsToVectorize.insert(dim);
    }
    // Select the "real", non-trivial vectorization dim, which may have non-unit
    // upper bound.
    std::optional<int64_t> dimToVectorize = chooseDimToVectorize(
        rewriter, loc, im2colOp, iterationDomain, inputSizes, kOffset, solver);
    if (dimToVectorize.has_value()) {
      dimsToVectorize.insert(dimToVectorize.value());
    }
    im2colOp.setVectorizationHint(dimsToVectorize.getArrayRef());
    LDBG() << "annotated im2col op with vectorization hint: " << im2colOp
           << "\n";
  });
}

namespace {
struct DecomposeIm2colPass final
    : impl::DecomposeIm2colPassBase<DecomposeIm2colPass> {
  using Base::Base;

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
  mlir::FunctionOpInterface funcOp = getOperation();

  RewritePatternSet preprocessingPatterns(context);
  affine::populateSimplifyAffineMinMaxPatterns(preprocessingPatterns);
  // After im2col is decomposed, im2col extract slice can be swapped with input
  // padding.
  if (failed(applyPatternsGreedily(getOperation(),
                                   std::move(preprocessingPatterns)))) {
    return signalPassFailure();
  }

  DataFlowSolver solver;
  solver.load<IREE::Util::IntegerDivisibilityAnalysis>();
  solver.load<dataflow::SparseConstantPropagation>();
  solver.load<dataflow::DeadCodeAnalysis>();
  if (succeeded(solver.initializeAndRun(funcOp))) {
    annotateIm2colVectorizationHints(funcOp, solver);
  }

  SmallVector<Im2colOp> candidates;
  funcOp->walk([&](Im2colOp op) { candidates.push_back(op); });
  IRRewriter rewriter(context);
  for (auto im2colOp : candidates) {
    if (failed(decomposeIm2col(im2colOp, rewriter, unroll))) {
      return signalPassFailure();
    }
  }
  RewritePatternSet patterns(context);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  // After im2col is decomposed, im2col extract slice can be swapped with input
  // padding.
  patterns.insert<linalg::ExtractSliceOfPadTensorSwapPattern>(
      context, [](tensor::ExtractSliceOp) { return false; });
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}
} // namespace mlir::iree_compiler::IREE::LinalgExt
