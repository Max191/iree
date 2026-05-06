// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/AffineSeedDependencyAnalysis.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/DebugLog.h"

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
  // Region argument propagation is conservative in this first version. Seeds are
  // initialized explicitly and all other non-control-flow arguments are unknown.
  for (AffineSeedDependencyLattice *arg : argLattices) {
    setToSeedOrUnknown(arg);
  }
}

} // namespace mlir::iree_compiler::IREE::Util
