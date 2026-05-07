// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/AffineSeedDependencyAnalysis.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

#define DEBUG_TYPE "iree-util-affine-seed-dependency-analysis"

namespace mlir::iree_compiler::IREE::Util {

AffineSeedDependencyAnalysis::AffineSeedDependencyAnalysis(
    DataFlowSolver &solver, SeedPredicate seedPredicate)
    : SparseForwardDataFlowAnalysis(solver),
      seedPredicate(std::move(seedPredicate)) {}

bool AffineSeedDependencyAnalysis::isSeed(Value value) const {
  return seedPredicate && seedPredicate(value);
}

void AffineSeedDependencyAnalysis::setLattice(
    AffineSeedDependencyLattice *lattice,
    const AffineSeedDependency &dependency) {
  propagateIfChanged(lattice, lattice->join(dependency));
}

void AffineSeedDependencyAnalysis::setToEntryState(
    AffineSeedDependencyLattice *lattice) {
  Value value = lattice->getAnchor();
  setLattice(lattice, isSeed(value) ? AffineSeedDependency::getSeed(value)
                                    : AffineSeedDependency::getIndependent());
}

void AffineSeedDependencyAnalysis::setToUnknown(
    AffineSeedDependencyLattice *lattice) {
  setLattice(lattice, AffineSeedDependency::getUnknown());
}

void AffineSeedDependencyAnalysis::setToSeedOrUnknown(
    AffineSeedDependencyLattice *lattice) {
  Value value = lattice->getAnchor();
  setLattice(lattice, isSeed(value) ? AffineSeedDependency::getSeed(value)
                                    : AffineSeedDependency::getUnknown());
}

LogicalResult AffineSeedDependencyAnalysis::visitOperation(
    Operation *op, ArrayRef<const AffineSeedDependencyLattice *> operands,
    ArrayRef<AffineSeedDependencyLattice *> results) {
  llvm::SmallDenseSet<Value> seedResults;
  for (AffineSeedDependencyLattice *result : results) {
    Value value = result->getAnchor();
    if (isSeed(value)) {
      seedResults.insert(value);
      setLattice(result, AffineSeedDependency::getSeed(value));
    }
  }

  if (seedResults.size() == results.size()) {
    return success();
  }

  if (llvm::any_of(operands, [](const AffineSeedDependencyLattice *lattice) {
        return lattice->getValue().isUninitialized();
      })) {
    return success();
  }

  InferAffineSeedDependencyOpInterface inferrable =
      dyn_cast<InferAffineSeedDependencyOpInterface>(op);
  if (!inferrable) {
    for (AffineSeedDependencyLattice *result : results) {
      if (!seedResults.contains(result->getAnchor())) {
        setToUnknown(result);
      }
    }
    return success();
  }

  LDBG() << "Inferring affine seed dependencies for " << op->getName()
         << " at " << op->getLoc() << "\n";
  auto argDependencies = llvm::map_to_vector(
      operands, [](const AffineSeedDependencyLattice *lattice) {
        return lattice->getValue();
      });

  llvm::SmallDenseSet<Value> updatedResults;
  auto setResultDependencies =
      [&](Value value, const AffineSeedDependency &dependency) {
        auto result = dyn_cast<OpResult>(value);
        if (!result) {
          return;
        }
        assert(result.getOwner() == op && "expected a result of this op");
        if (seedResults.contains(value)) {
          return;
        }
        updatedResults.insert(value);
        setLattice(results[result.getResultNumber()], dependency);
      };

  inferrable.inferResultAffineSeedDependencies(argDependencies,
                                               setResultDependencies);

  for (AffineSeedDependencyLattice *result : results) {
    Value value = result->getAnchor();
    if (!seedResults.contains(value) && !updatedResults.contains(value)) {
      setToUnknown(result);
    }
  }
  return success();
}

void AffineSeedDependencyAnalysis::visitExternalCall(
    CallOpInterface call,
    ArrayRef<const AffineSeedDependencyLattice *> argumentLattices,
    ArrayRef<AffineSeedDependencyLattice *> resultLattices) {
  // External calls are conservatively unknown until call effect modeling is
  // precise enough to propagate affine dependencies through callee boundaries.
  for (AffineSeedDependencyLattice *result : resultLattices) {
    setToSeedOrUnknown(result);
  }
}

void AffineSeedDependencyAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor, ValueRange successorInputs,
    ArrayRef<AffineSeedDependencyLattice *> argLattices) {
  // Get the dependency state for an OpFoldResult. Static values are independent
  // of all seeds; dynamic values are queried before the loop body and may
  // require a later solver revisit if their producer has not been initialized.
  auto getDependencyFromOfr =
      [&](std::optional<OpFoldResult> ofr,
          Block *block) -> std::optional<AffineSeedDependency> {
    if (!ofr) {
      return AffineSeedDependency::getIndependent();
    }
    if (getConstantIntValue(*ofr)) {
      return AffineSeedDependency::getIndependent();
    }
    if (dyn_cast<Attribute>(*ofr)) {
      return AffineSeedDependency::getIndependent();
    }
    Value value = cast<Value>(*ofr);
    const AffineSeedDependencyLattice *lattice =
        getLatticeElementFor(getProgramPointBefore(block), value);
    if (!lattice || lattice->getValue().isUninitialized()) {
      return std::nullopt;
    }
    return lattice->getValue();
  };

  // Infer LoopLike induction variables as `iv = lowerBound + i * step`.
  // The iteration counter `i` is not a seed-derived value. Therefore an
  // independent step contributes no seed coefficient, while a seed-dependent
  // step makes exactly those dependent seed coefficients unknown.
  if (auto loop = dyn_cast<LoopLikeOpInterface>(op)) {
    std::optional<SmallVector<Value>> ivs = loop.getLoopInductionVars();
    std::optional<SmallVector<OpFoldResult>> lbs = loop.getLoopLowerBounds();
    std::optional<SmallVector<OpFoldResult>> steps = loop.getLoopSteps();
    if (!ivs || !lbs || !steps) {
      return SparseForwardDataFlowAnalysis::visitNonControlFlowArguments(
          op, successor, successorInputs, argLattices);
    }

    llvm::SmallDenseSet<Value> loopIvs;
    for (auto [iv, lb, step] : llvm::zip_equal(*ivs, *lbs, *steps)) {
      loopIvs.insert(iv);
      AffineSeedDependencyLattice *ivLattice = getLatticeElement(iv);
      if (isSeed(iv)) {
        setLattice(ivLattice, AffineSeedDependency::getSeed(iv));
        continue;
      }

      Block *block = iv.getParentBlock();
      std::optional<AffineSeedDependency> lbDependency =
          getDependencyFromOfr(lb, block);
      std::optional<AffineSeedDependency> stepDependency =
          getDependencyFromOfr(step, block);
      if (!lbDependency || !stepDependency) {
        // Do not pessimize this to unknown: joins are monotonic and unknown
        // cannot refine back to a precise coefficient map. The lattice query
        // above records a dependency and the solver will revisit this transfer
        // once the bound state is initialized.
        continue;
      }

      AffineSeedDependency ivDependency = *lbDependency;
      if (!stepDependency->isIndependent()) {
        ivDependency = AffineSeedDependency::add(
            ivDependency, AffineSeedDependency::getUnknownForDependentSeeds(
                              {*stepDependency}));
      }
      setLattice(ivLattice, ivDependency);
    }

    // Preserve conservative handling for any other non-control-flow region
    // arguments not modeled as loop induction variables.
    for (AffineSeedDependencyLattice *arg : argLattices) {
      Value value = arg->getAnchor();
      if (!loopIvs.contains(value)) {
        setToSeedOrUnknown(arg);
      }
    }
    return;
  }

  // Region argument propagation is conservative by default. Seeds are
  // initialized explicitly and all other non-control-flow arguments are unknown.
  for (AffineSeedDependencyLattice *arg : argLattices) {
    setToSeedOrUnknown(arg);
  }
}

} // namespace mlir::iree_compiler::IREE::Util
