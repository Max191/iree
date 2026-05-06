// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/Util/Analysis/AffineSeedDependencyAnalysis.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_ANNOTATEMAPSTORECONTIGUOUSDIMHINTSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

/// Returns true when `dependency` is known with coefficient 1 on `seed`,
/// coefficient 0 on every other current map_store input index seed, and no
/// explicit non-zero or unknown coefficient-map entries for seeds from other
/// map_store ops in the function.
static bool isUnitDependencyOnOnlySeed(
    const IREE::Util::AffineSeedDependency &dependency, Value seed,
    ArrayRef<Value> inputIndices) {
  if (!dependency.isKnown()) {
    return false;
  }
  for (Value inputIndex : inputIndices) {
    std::optional<int64_t> coefficient = dependency.getCoefficient(inputIndex);
    if (!coefficient) {
      return false;
    }
    int64_t expectedCoefficient = inputIndex == seed ? 1 : 0;
    if (*coefficient != expectedCoefficient) {
      return false;
    }
  }
  // The seed set is function-wide, but map_store region scoping should prevent
  // one map_store body from depending on another map_store's block-argument
  // seeds. Reject foreign explicit coefficient keys defensively if that ever
  // changes.
  for (Value coefficientSeed :
       llvm::make_first_range(dependency.getCoefficients())) {
    if (!llvm::is_contained(inputIndices, coefficientSeed)) {
      return false;
    }
  }
  return true;
}

/// Infers only exact unit-contiguous dimensions that the affine seed-dependency
/// analysis can certify. Ambiguous input dims that match zero or multiple output
/// dims are intentionally left unhinted; vectorization may still use its legacy
/// innermost fallback.
static SmallVector<int64_t>
inferContiguousDimHints(IREE::LinalgExt::MapStoreOp mapStoreOp,
                        DataFlowSolver &solver) {
  SmallVector<Value> inputIndices;
  inputIndices.reserve(mapStoreOp.getInputRank());
  for (int64_t inputDim = 0, e = mapStoreOp.getInputRank(); inputDim < e;
       ++inputDim) {
    inputIndices.push_back(mapStoreOp.getInputIndex(inputDim));
  }

  SmallVector<SmallVector<int64_t>> candidateOutputDims(
      mapStoreOp.getInputRank());
  for (int64_t outputDim = 0, e = mapStoreOp.getOutputRank(); outputDim < e;
       ++outputDim) {
    Value outputIndex = mapStoreOp.getOutputIndex(outputDim);
    const IREE::Util::AffineSeedDependencyLattice *lattice =
        solver.lookupState<IREE::Util::AffineSeedDependencyLattice>(
            outputIndex);
    if (!lattice) {
      continue;
    }
    const IREE::Util::AffineSeedDependency &dependency = lattice->getValue();
    for (int64_t inputDim = 0, f = mapStoreOp.getInputRank(); inputDim < f;
         ++inputDim) {
      if (isUnitDependencyOnOnlySeed(dependency, inputIndices[inputDim],
                                     inputIndices)) {
        candidateOutputDims[inputDim].push_back(outputDim);
      }
    }
  }

  SmallVector<int64_t> hints;
  for (auto [inputDim, outputDims] : llvm::enumerate(candidateOutputDims)) {
    if (outputDims.size() != 1) {
      continue;
    }
    hints.push_back(inputDim);
    hints.push_back(outputDims.front());
  }
  return hints;
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
      if (mapStoreOp.getContiguousDimHintsAttr()) {
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
                              "skipping map_store contiguous dimension hints";
      return;
    }

    for (IREE::LinalgExt::MapStoreOp mapStoreOp : mapStoreOps) {
      SmallVector<int64_t> hints = inferContiguousDimHints(mapStoreOp, solver);
      if (hints.empty()) {
        continue;
      }
      mapStoreOp.setContiguousDimHintsAttr(
          DenseI64ArrayAttr::get(mapStoreOp.getContext(), hints));
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
