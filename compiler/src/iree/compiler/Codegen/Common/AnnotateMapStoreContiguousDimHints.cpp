// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/Util/Analysis/AffineSeedDependencyAnalysis.h"
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

static bool hasOnlyKnownCurrentInputIndexCoefficients(
    const IREE::Util::AffineSeedDependency &dependency,
    ArrayRef<Value> inputIndices) {
  if (!dependency.isKnown()) {
    return false;
  }
  for (Value inputIndex : inputIndices) {
    if (!dependency.getCoefficient(inputIndex)) {
      return false;
    }
  }
  for (auto [seed, coefficient] : dependency.getCoefficients()) {
    if (!coefficient || !llvm::is_contained(inputIndices, seed)) {
      return false;
    }
  }
  return true;
}

/// Infers a transfer_scatter base indexing map from affine seed dependencies.
/// Each known output dim is represented as the sum of its input-index
/// coefficients. Unknown, unsupported, or seed-specific-unknown output dims are
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
    if (!hasOnlyKnownCurrentInputIndexCoefficients(dependency, inputIndices)) {
      dependency = IREE::Util::inferLocalAffineSeedDependency(
          outputIndex, mapStoreOp.getTransformationRegion(), inputIndices,
          localMemo, localInFlight);
    }
    if (!hasOnlyKnownCurrentInputIndexCoefficients(dependency, inputIndices)) {
      exprs.push_back(getAffineSymbolExpr(numSymbols++, ctx));
      continue;
    }

    AffineExpr expr = getAffineConstantExpr(0, ctx);
    for (int64_t inputDim = 0, f = mapStoreOp.getInputRank(); inputDim < f;
         ++inputDim) {
      int64_t coefficient = *dependency.getCoefficient(inputIndices[inputDim]);
      if (coefficient == 0) {
        continue;
      }
      AffineExpr dimExpr = getAffineDimExpr(inputDim, ctx);
      expr = expr + (coefficient == 1 ? dimExpr : dimExpr * coefficient);
    }
    exprs.push_back(expr);
  }
  return AffineMap::get(mapStoreOp.getInputRank(), numSymbols, exprs, ctx);
}

struct AnnotateMapStoreContiguousDimHintsPass final
    : impl::AnnotateMapStoreContiguousDimHintsPassBase<
          AnnotateMapStoreContiguousDimHintsPass> {
  using Base::Base;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    llvm::DenseSet<Value> seeds;
    SmallVector<IREE::LinalgExt::MapStoreOp> mapStoreOps;

    funcOp.walk([&](IREE::LinalgExt::MapStoreOp mapStoreOp) {
      if (mapStoreOp.getContiguousDimHintsAttr() ||
          mapStoreOp.getTransferScatterIndexingMapAttr()) {
        return;
      }
      mapStoreOps.push_back(mapStoreOp);
      for (int64_t inputDim = 0, e = mapStoreOp.getInputRank(); inputDim < e;
           ++inputDim) {
        seeds.insert(mapStoreOp.getInputIndex(inputDim));
      }
    });
    if (mapStoreOps.empty()) {
      return;
    }

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<IREE::Util::AffineSeedDependencyAnalysis>(
        [&](Value value) { return seeds.contains(value); });
    if (failed(solver.initializeAndRun(funcOp))) {
      funcOp->emitRemark() << "failed affine seed-dependency analysis; "
                              "skipping map_store transfer_scatter metadata";
      return;
    }

    for (IREE::LinalgExt::MapStoreOp mapStoreOp : mapStoreOps) {
      AffineMap indexingMap =
          inferTransferScatterIndexingMap(mapStoreOp, solver);
      mapStoreOp.setTransferScatterIndexingMapAttr(
          AffineMapAttr::get(indexingMap));
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
