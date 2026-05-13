// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"

#include "iree/compiler/Codegen/Utils/MapStoreVectorization.h"
#include "iree/compiler/Dialect/Util/Analysis/AffineSeedDependencyAnalysis.h"
#include "iree/compiler/Utils/Indexing.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/BuiltinAttributes.h"

#define DEBUG_TYPE "iree-codegen-annotate-map-store-contiguous-dim-hints"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_ANNOTATEMAPSTORECONTIGUOUSDIMHINTSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

static bool isKnownInputIndexExpression(
    const IREE::Util::AffineSeedDependency &dependency,
    ArrayRef<Value> inputIndices, MLIRContext *ctx) {
  if (!dependency.isKnown() || dependency.hasInvalidatedSeeds()) {
    return false;
  }

  for (auto [seed, position] : dependency.getSeedPositions()) {
    if (position >= inputIndices.size() || inputIndices[position] != seed) {
      return false;
    }
  }

  AffineExpr expression = dependency.getExpression(ctx);
  for (int64_t inputDim = 0, e = inputIndices.size(); inputDim < e;
       ++inputDim) {
    if (!getCoefficient(expression, inputDim)) {
      return false;
    }
  }
  return true;
}

static AffineExpr getInputIndexContributionExpression(
    const IREE::Util::AffineSeedDependency &dependency, int64_t inputRank,
    MLIRContext *ctx) {
  // The transfer_scatter indexing map captures only the vector-dimension
  // contribution. Seed-independent offsets, including affine constants and
  // dynamic SSA values, stay implicit and are materialized later from the full
  // yielded index with the input index block arguments replaced by zero.
  AffineExpr expression = dependency.getExpression(ctx);
  AffineExpr contribution = getAffineConstantExpr(0, ctx);
  for (int64_t inputDim = 0; inputDim < inputRank; ++inputDim) {
    int64_t coefficient = *getCoefficient(expression, inputDim);
    if (coefficient == 0) {
      continue;
    }
    AffineExpr dimExpr = getAffineDimExpr(inputDim, ctx);
    contribution = contribution +
                   (coefficient == 1 ? dimExpr : dimExpr * coefficient);
  }
  return contribution;
}

static AffineMap inferTransferScatterIndexingMap(
    IREE::LinalgExt::MapStoreOp mapStoreOp, DataFlowSolver &solver) {
  SmallVector<Value> inputIndices;
  inputIndices.reserve(mapStoreOp.getInputRank());
  for (int64_t inputDim = 0, e = mapStoreOp.getInputRank(); inputDim < e;
       ++inputDim) {
    inputIndices.push_back(mapStoreOp.getInputIndex(inputDim));
  }

  MLIRContext *ctx = mapStoreOp.getContext();
  SmallVector<AffineExpr> resultExprs;
  resultExprs.reserve(mapStoreOp.getOutputRank());
  int64_t numSymbols = 0;
  for (int64_t outputDim = 0, e = mapStoreOp.getOutputRank(); outputDim < e;
       ++outputDim) {
    Value outputIndex = mapStoreOp.getOutputIndex(outputDim);
    const IREE::Util::AffineSeedDependencyLattice *lattice =
        solver.lookupState<IREE::Util::AffineSeedDependencyLattice>(
            outputIndex);
    IREE::Util::AffineSeedDependency dependency =
        lattice ? lattice->getValue()
                : IREE::Util::AffineSeedDependency::getUnknown();
    if (!isKnownInputIndexExpression(dependency, inputIndices, ctx)) {
      resultExprs.push_back(getAffineSymbolExpr(numSymbols++, ctx));
      continue;
    }

    resultExprs.push_back(getInputIndexContributionExpression(
        dependency, mapStoreOp.getInputRank(), ctx));
  }
  return AffineMap::get(mapStoreOp.getInputRank(), numSymbols, resultExprs,
                        ctx);
}

struct AnnotateMapStoreContiguousDimHintsPass final
    : impl::AnnotateMapStoreContiguousDimHintsPassBase<
          AnnotateMapStoreContiguousDimHintsPass> {
  using Base::Base;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    SmallVector<IREE::LinalgExt::MapStoreOp> mapStoreOps;
    funcOp.walk([&](IREE::LinalgExt::MapStoreOp mapStoreOp) {
      if (hasMapStoreVectorizationHints(mapStoreOp)) {
        return;
      }
      mapStoreOps.push_back(mapStoreOp);
    });
    if (mapStoreOps.empty()) {
      return;
    }

    for (IREE::LinalgExt::MapStoreOp mapStoreOp : mapStoreOps) {
      // The affine seed positions are local to one map_store's input index
      // block arguments, so keep solver state scoped to a single op. This can
      // be batched later if compile-time profiling shows it matters.
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
        LDBG() << "failed affine seed-dependency analysis for " << mapStoreOp
               << "; skipping transfer_scatter metadata\n";
        continue;
      }

      setMapStoreTransferScatterIndexingMap(
          mapStoreOp,
          AffineMapAttr::get(inferTransferScatterIndexingMap(mapStoreOp,
                                                             solver)));
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
