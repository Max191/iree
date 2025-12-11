// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::Codegen {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

namespace {

/// Expand an InnerTiledOp by propagating a reshape through it.
/// `fusedOperand` is the operand connected to the reshape.
/// `reassociation` describes how the collapsed dims map to expanded dims.
/// `expandedShape` is the full expanded shape (outer + inner dims).
/// `expandedValue` is the value to use for the fused operand (either the
/// collapse_shape source or a newly created expand_shape).
static FailureOr<InnerTiledOp>
expandInnerTiledOp(InnerTiledOp op, OpOperand *fusedOperand,
                   ArrayRef<ReassociationIndices> reassociation,
                   ArrayRef<int64_t> expandedShape, Value expandedValue,
                   PatternRewriter &rewriter) {
  SmallVector<AffineMap> indexingMaps = op.getIndexingMapsArray();
  int64_t fusedIdx = fusedOperand->getOperandNumber();
  AffineMap fusedMap = indexingMaps[fusedIdx];
  int64_t outerRank = fusedMap.getNumResults();
  int64_t numIterDims = fusedMap.getNumDims();
  assert(reassociation.size() ==
             cast<RankedTensorType>(fusedOperand->get().getType()).getRank() &&
         "expected reassociation rank to match fused operand rank");

  // Only outer dims can be expanded because inner dims depend on the `kind`
  // attribute's implementation.
  if (llvm::any_of(reassociation.drop_front(outerRank),
                   [](ArrayRef<int64_t> group) { return group.size() != 1; })) {
    return failure();
  }

  // Get iteration bounds to know sizes of dims not in the fused operand's map.
  SmallVector<int64_t> iterationBounds;
  op.getIterationBounds(iterationBounds);

  // Build mapping: iterDim -> list of (expandedIterDim, size).
  // Only iteration dims used by the fused operand get expanded.
  SmallVector<SmallVector<std::pair<int64_t, int64_t>>> iterDimExpansion(
      numIterDims);
  int64_t expandedDimCounter = 0;
  for (auto [resultIdx, expr] : llvm::enumerate(fusedMap.getResults())) {
    int64_t iterDim = cast<AffineDimExpr>(expr).getPosition();
    for (int64_t expandedOperandIdx : reassociation[resultIdx]) {
      iterDimExpansion[iterDim].push_back(
          {expandedDimCounter++, expandedShape[expandedOperandIdx]});
    }
  }
  // Iteration dims not in fused map stay as single dims with their original
  // size from the iteration bounds.
  for (int64_t i = 0; i < numIterDims; ++i) {
    if (iterDimExpansion[i].empty())
      iterDimExpansion[i].push_back({expandedDimCounter++, iterationBounds[i]});
  }
  int64_t expandedNumDims = expandedDimCounter;

  Location loc = op.getLoc();
  MLIRContext *ctx = op.getContext();

  // Expand indexing maps.
  SmallVector<AffineMap> newIndexingMaps;
  for (AffineMap map : indexingMaps) {
    SmallVector<AffineExpr> newResults;
    for (AffineExpr expr : map.getResults()) {
      int64_t iterDim = cast<AffineDimExpr>(expr).getPosition();
      for (auto [expandedDim, size] : iterDimExpansion[iterDim])
        newResults.push_back(getAffineDimExpr(expandedDim, ctx));
    }
    newIndexingMaps.push_back(
        AffineMap::get(expandedNumDims, 0, newResults, ctx));
  }

  // Expand operands.
  SmallVector<Value> newOperands;
  for (OpOperand &operand : op->getOpOperands()) {
    if (&operand == fusedOperand) {
      newOperands.push_back(expandedValue);
      continue;
    }

    AffineMap origMap = indexingMaps[operand.getOperandNumber()];
    auto shapedType = cast<RankedTensorType>(operand.get().getType());
    int64_t operandOuterRank = origMap.getNumResults();
    ArrayRef<int64_t> origShape = shapedType.getShape();
    ArrayRef<int64_t> innerShape = origShape.drop_front(operandOuterRank);

    // Compute new shape.
    SmallVector<int64_t> newShape;
    for (AffineExpr expr : origMap.getResults()) {
      int64_t iterDim = cast<AffineDimExpr>(expr).getPosition();
      for (auto [expandedDim, size] : iterDimExpansion[iterDim])
        newShape.push_back(size);
    }
    llvm::append_range(newShape, innerShape);

    if (newShape == SmallVector<int64_t>(origShape)) {
      newOperands.push_back(operand.get());
    } else {
      // Build reassociation for this operand.
      SmallVector<ReassociationIndices> operandReassoc;
      int64_t dimCounter = 0;
      for (AffineExpr expr : origMap.getResults()) {
        int64_t iterDim = cast<AffineDimExpr>(expr).getPosition();
        ReassociationIndices group;
        for (size_t i = 0; i < iterDimExpansion[iterDim].size(); ++i)
          group.push_back(dimCounter++);
        operandReassoc.push_back(group);
      }
      for (int64_t i = 0; i < static_cast<int64_t>(innerShape.size()); ++i)
        operandReassoc.push_back({dimCounter++});

      auto newType =
          RankedTensorType::get(newShape, shapedType.getElementType());
      newOperands.push_back(tensor::ExpandShapeOp::create(
          rewriter, loc, newType, operand.get(), operandReassoc));
    }
  }

  // Expand iterator types.
  SmallVector<utils::IteratorType> newIterTypes;
  for (auto [idx, iterType] : llvm::enumerate(op.getIteratorTypesArray())) {
    for (size_t i = 0; i < iterDimExpansion[idx].size(); ++i)
      newIterTypes.push_back(iterType);
  }

  int64_t numInputs = op.getNumInputs();
  SmallVector<Value> newInputs(newOperands.begin(),
                               newOperands.begin() + numInputs);
  SmallVector<Value> newOutputs(newOperands.begin() + numInputs,
                                newOperands.end());

  std::optional<SmallVector<SmallVector<int64_t>>> permutations;
  if (op.getPermutations()) {
    permutations.emplace();
    for (auto permAttr :
         op.getPermutations()->getAsRange<DenseI64ArrayAttr>()) {
      permutations->push_back(SmallVector<int64_t>(permAttr.asArrayRef()));
    }
  }

  return InnerTiledOp::create(rewriter, loc, newInputs, newOutputs,
                              newIndexingMaps, newIterTypes, op.getKind(),
                              op.getSemantics(), permutations);
}

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

struct FoldProducerCollapseShapeWithInnerTiled
    : public OpRewritePattern<InnerTiledOp> {
  FoldProducerCollapseShapeWithInnerTiled(MLIRContext *context,
                                          linalg::ControlFusionFn controlFn,
                                          PatternBenefit benefit = 1)
      : OpRewritePattern<InnerTiledOp>(context, benefit),
        controlFn(std::move(controlFn)) {}

  LogicalResult matchAndRewrite(InnerTiledOp op,
                                PatternRewriter &rewriter) const override {
    for (OpOperand &operand : op->getOpOperands()) {
      auto collapseOp = operand.get().getDefiningOp<tensor::CollapseShapeOp>();
      if (!collapseOp || !controlFn(&operand))
        continue;

      FailureOr<InnerTiledOp> expandedOp = expandInnerTiledOp(
          op, &operand, collapseOp.getReassociationIndices(),
          collapseOp.getSrcType().getShape(), collapseOp.getSrc(), rewriter);
      if (failed(expandedOp))
        continue;

      // Collapse results back to original shape.
      SmallVector<Value> results;
      SmallVector<AffineMap> indexingMaps = op.getIndexingMapsArray();
      for (auto [idx, result] : llvm::enumerate(expandedOp->getResults())) {
        AffineMap resultMap = indexingMaps[op.getNumInputs() + idx];
        auto resultType = cast<RankedTensorType>(op.getResultTypes()[idx]);
        int64_t resultOuterRank = resultMap.getNumResults();
        int64_t innerRank = resultType.getRank() - resultOuterRank;

        // Build result reassociation using same logic as expansion.
        AffineMap fusedMap = indexingMaps[operand.getOperandNumber()];
        SmallVector<SmallVector<int64_t>> iterDimExpansion(
            fusedMap.getNumDims());
        int64_t dimCounter = 0;
        for (auto [i, expr] : llvm::enumerate(fusedMap.getResults())) {
          int64_t iterDim = cast<AffineDimExpr>(expr).getPosition();
          for (size_t j = 0; j < collapseOp.getReassociationIndices()[i].size();
               ++j)
            iterDimExpansion[iterDim].push_back(dimCounter++);
        }
        for (int64_t i = 0; i < fusedMap.getNumDims(); ++i) {
          if (iterDimExpansion[i].empty())
            iterDimExpansion[i].push_back(dimCounter++);
        }

        SmallVector<ReassociationIndices> resultReassoc;
        dimCounter = 0;
        for (AffineExpr expr : resultMap.getResults()) {
          int64_t iterDim = cast<AffineDimExpr>(expr).getPosition();
          ReassociationIndices group;
          for (size_t i = 0; i < iterDimExpansion[iterDim].size(); ++i)
            group.push_back(dimCounter++);
          resultReassoc.push_back(group);
        }
        for (int64_t i = 0; i < innerRank; ++i)
          resultReassoc.push_back({dimCounter++});

        results.push_back(tensor::CollapseShapeOp::create(
            rewriter, op.getLoc(), resultType, result, resultReassoc));
      }
      rewriter.replaceOp(op, results);
      return success();
    }
    return failure();
  }

private:
  linalg::ControlFusionFn controlFn;
};

struct FoldConsumerExpandShapeWithInnerTiled
    : public OpRewritePattern<tensor::ExpandShapeOp> {
  FoldConsumerExpandShapeWithInnerTiled(MLIRContext *context,
                                        linalg::ControlFusionFn controlFn,
                                        PatternBenefit benefit = 1)
      : OpRewritePattern<tensor::ExpandShapeOp>(context, benefit),
        controlFn(std::move(controlFn)) {}

  LogicalResult matchAndRewrite(tensor::ExpandShapeOp expandOp,
                                PatternRewriter &rewriter) const override {
    auto producerResult = dyn_cast<OpResult>(expandOp.getSrc());
    if (!producerResult)
      return failure();

    auto innerTiledOp = dyn_cast<InnerTiledOp>(producerResult.getOwner());
    if (!innerTiledOp || !controlFn(&expandOp.getSrcMutable()))
      return failure();

    int64_t resultIdx = producerResult.getResultNumber();
    OpOperand *outputOperand = innerTiledOp.getDpsInitOperand(resultIdx);

    // Create expanded output init.
    auto expandedInit = tensor::ExpandShapeOp::create(
        rewriter, expandOp.getLoc(), expandOp.getResultType(),
        outputOperand->get(), expandOp.getReassociationIndices());

    FailureOr<InnerTiledOp> expandedOp = expandInnerTiledOp(
        innerTiledOp, outputOperand, expandOp.getReassociationIndices(),
        expandOp.getResultType().getShape(), expandedInit, rewriter);
    if (failed(expandedOp))
      return failure();

    rewriter.replaceOp(expandOp, expandedOp->getResult(resultIdx));
    return success();
  }

private:
  linalg::ControlFusionFn controlFn;
};

} // namespace

//===----------------------------------------------------------------------===//
// Populate Functions
//===----------------------------------------------------------------------===//

void populateFoldReshapeOpsByExpansionPatterns(
    RewritePatternSet &patterns,
    const linalg::ControlFusionFn &controlFoldingReshapes) {
  patterns.add<FoldProducerCollapseShapeWithInnerTiled,
               FoldConsumerExpandShapeWithInnerTiled>(patterns.getContext(),
                                                      controlFoldingReshapes);
}

} // namespace mlir::iree_compiler::IREE::Codegen