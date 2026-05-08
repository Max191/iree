// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_UTIL_ANALYSIS_AFFINE_SEED_DEPENDENCY_ANALYSIS_H_
#define IREE_COMPILER_DIALECT_UTIL_ANALYSIS_AFFINE_SEED_DEPENDENCY_ANALYSIS_H_

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <functional>

namespace mlir::iree_compiler::IREE::Util {

class AffineSeedDependencyLattice
    : public dataflow::Lattice<AffineSeedDependency> {
public:
  using Lattice::Lattice;
};

class AffineSeedDependencyAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<
          AffineSeedDependencyLattice> {
public:
  using SeedPredicate = std::function<bool(Value)>;

  explicit AffineSeedDependencyAnalysis(
      DataFlowSolver &solver, SeedPredicate seedPredicate = {});

  void setToEntryState(AffineSeedDependencyLattice *lattice) override;

  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const AffineSeedDependencyLattice *> operands,
                 ArrayRef<AffineSeedDependencyLattice *> results) override;

  void visitExternalCall(
      CallOpInterface call,
      ArrayRef<const AffineSeedDependencyLattice *> argumentLattices,
      ArrayRef<AffineSeedDependencyLattice *> resultLattices) override;

  void visitNonControlFlowArguments(
      Operation *op, const RegionSuccessor &successor,
      ValueRange successorInputs,
      ArrayRef<AffineSeedDependencyLattice *> argLattices) override;

private:
  bool isSeed(Value value) const;
  void setLattice(AffineSeedDependencyLattice *lattice,
                  const AffineSeedDependency &dependency);
  void setToUnknown(AffineSeedDependencyLattice *lattice);
  void setToSeedOrUnknown(AffineSeedDependencyLattice *lattice);

  SeedPredicate seedPredicate;
};

/// Infers seed coefficients from a single-result affine map expression.
AffineSeedDependency
inferAffineMapSeedDependency(AffineMap map,
                             ArrayRef<AffineSeedDependency> argDeps);

/// Infers affine seed dependencies by walking local SSA producers within
/// `scopeRegion`. Values defined outside `scopeRegion` are treated as
/// seed-independent captures.
AffineSeedDependency inferLocalAffineSeedDependency(
    Value value, Region &scopeRegion, ArrayRef<Value> seeds,
    llvm::DenseMap<Value, AffineSeedDependency> &memo,
    llvm::DenseSet<Value> &inFlight);

AffineSeedDependency inferLocalAffineSeedDependency(Value value,
                                                    Region &scopeRegion,
                                                    ArrayRef<Value> seeds);

} // namespace mlir::iree_compiler::IREE::Util

#endif // IREE_COMPILER_DIALECT_UTIL_ANALYSIS_AFFINE_SEED_DEPENDENCY_ANALYSIS_H_
