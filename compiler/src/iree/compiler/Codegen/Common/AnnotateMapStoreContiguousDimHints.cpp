// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/Util/Analysis/AffineSeedDependencyAnalysis.h"
#include "iree/compiler/Utils/AffineExprUtils.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_ANNOTATEMAPSTORECONTIGUOUSDIMHINTSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

static bool hasOnlyKnownCurrentInputIndexExpression(
    const IREE::Util::AffineSeedDependency &dependency,
    ArrayRef<Value> inputIndices, MLIRContext *ctx) {
  if (!dependency.isKnown() || dependency.hasInvalidatedSeeds()) {
    return false;
  }
  for (auto [seed, position] : dependency.getSeedPositions()) {
    if (position >= inputIndices.size() ||
        !llvm::is_contained(inputIndices, seed)) {
      return false;
    }
  }
  AffineExpr expression = dependency.getExpression(ctx);
  for (int64_t inputDim = 0, e = inputIndices.size(); inputDim < e;
       ++inputDim) {
    if (!getAffineDimCoefficient(expression, inputDim)) {
      return false;
    }
  }
  return true;
}

/// Infers a transfer_scatter base indexing map from affine seed dependencies.
/// Each known output dim is represented by its affine expression in the input
/// index seed dims. Unknown, unsupported, or invalidated output dims are
/// represented as fresh symbols that the vectorizer will materialize with the
/// full vectorized output index.
static AffineMap
inferTransferScatterIndexingMap(IREE::LinalgExt::MapStoreOp mapStoreOp,
                                DataFlowSolver &solver) {
  SmallVector<Value> inputIndices;
  inputIndices.reserve(mapStoreOp.getInputRank());
  for (int64_t inputDim = 0, e = mapStoreOp.getInputRank(); inputDim < e;
       ++inputDim) {
    inputIndices.push_back(mapStoreOp.getInputIndex(inputDim));
  }

  MLIRContext *ctx = mapStoreOp.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.reserve(mapStoreOp.getOutputRank());
  int64_t numSymbols = 0;
  DenseMap<Value, IREE::Util::AffineSeedDependency> localMemo;
  DenseSet<Value> localInFlight;
  for (int64_t outputDim = 0, e = mapStoreOp.getOutputRank(); outputDim < e;
       ++outputDim) {
    Value outputIndex = mapStoreOp.getOutputIndex(outputDim);
    const IREE::Util::AffineSeedDependencyLattice *lattice =
        solver.lookupState<IREE::Util::AffineSeedDependencyLattice>(
            outputIndex);
    IREE::Util::AffineSeedDependency dependency =
        lattice ? lattice->getValue()
                : IREE::Util::AffineSeedDependency::getUnknown();
    if (!hasOnlyKnownCurrentInputIndexExpression(dependency, inputIndices,
                                                 ctx)) {
      dependency = IREE::Util::inferLocalAffineSeedDependency(
          outputIndex, mapStoreOp.getTransformationRegion(), inputIndices,
          localMemo, localInFlight);
    }
    if (!hasOnlyKnownCurrentInputIndexExpression(dependency, inputIndices,
                                                 ctx)) {
      exprs.push_back(getAffineSymbolExpr(numSymbols++, ctx));
      continue;
    }

    exprs.push_back(dependency.getExpression(ctx));
  }
  return AffineMap::get(mapStoreOp.getInputRank(), numSymbols, exprs, ctx);
}

struct AnnotateMapStoreContiguousDimHintsPass final
    : impl::AnnotateMapStoreContiguousDimHintsPassBase<
          AnnotateMapStoreContiguousDimHintsPass> {
  using Base::Base;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    SmallVector<IREE::LinalgExt::MapStoreOp> mapStoreOps;

    funcOp.walk([&](IREE::LinalgExt::MapStoreOp mapStoreOp) {
      if (mapStoreOp.getContiguousDimHintsAttr() ||
          mapStoreOp.getTransferScatterIndexingMapAttr()) {
        return;
      }
      mapStoreOps.push_back(mapStoreOp);
    });
    if (mapStoreOps.empty()) {
      return;
    }

    for (IREE::LinalgExt::MapStoreOp mapStoreOp : mapStoreOps) {
      // The seed-to-dim mapping is local to each map_store: its region block
      // arguments are the input vector dims for exactly this op. Keep the solver
      // scoped per op for now so the analysis state can use those dim positions
      // directly.
      llvm::DenseMap<Value, unsigned> seedPositions;
      for (int64_t inputDim = 0, e = mapStoreOp.getInputRank(); inputDim < e;
           ++inputDim) {
        seedPositions[mapStoreOp.getInputIndex(inputDim)] = inputDim;
      }

      DataFlowSolver solver;
      dataflow::loadBaselineAnalyses(solver);
      solver.load<IREE::Util::AffineSeedDependencyAnalysis>(
          [&](Value value) -> std::optional<unsigned> {
            auto it = seedPositions.find(value);
            if (it == seedPositions.end()) {
              return std::nullopt;
            }
            return it->second;
          });
      if (failed(solver.initializeAndRun(funcOp))) {
        mapStoreOp->emitRemark()
            << "failed affine seed-dependency analysis; skipping "
               "transfer_scatter metadata";
        continue;
      }

      AffineMap indexingMap =
          inferTransferScatterIndexingMap(mapStoreOp, solver);
      mapStoreOp.setTransferScatterIndexingMapAttr(
          AffineMapAttr::get(indexingMap));
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
