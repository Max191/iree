// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/AffineSeedDependencyAnalysis.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/IR/AffineExprVisitor.h"

#define DEBUG_TYPE "iree-util-affine-seed-dependency-analysis"

namespace mlir::iree_compiler::IREE::Util {

namespace {

class LocalAffineExprSeedDependencyFinder final
    : public AffineExprVisitor<LocalAffineExprSeedDependencyFinder,
                               AffineSeedDependency> {
public:
  LocalAffineExprSeedDependencyFinder(
      AffineMap map, ArrayRef<AffineSeedDependency> argDeps)
      : map(map), argDeps(argDeps) {}

  AffineSeedDependency visitConstantExpr(AffineConstantExpr expr) {
    return AffineSeedDependency::getKnown(
        getAffineConstantExpr(expr.getValue(), expr.getContext()));
  }

  AffineSeedDependency visitDimExpr(AffineDimExpr expr) {
    if (expr.getPosition() >= map.getNumDims() ||
        expr.getPosition() >= argDeps.size()) {
      return AffineSeedDependency::getUnknown();
    }
    return argDeps[expr.getPosition()];
  }

  AffineSeedDependency visitSymbolExpr(AffineSymbolExpr expr) {
    int64_t argIndex = static_cast<int64_t>(map.getNumDims()) +
                       static_cast<int64_t>(expr.getPosition());
    if (expr.getPosition() >= map.getNumSymbols() ||
        argIndex >= static_cast<int64_t>(argDeps.size())) {
      return AffineSeedDependency::getUnknown();
    }
    return argDeps[argIndex];
  }

  AffineSeedDependency visitAddExpr(AffineBinaryOpExpr expr) {
    return AffineSeedDependency::add(visit(expr.getLHS()),
                                     visit(expr.getRHS()));
  }

  AffineSeedDependency visitMulExpr(AffineBinaryOpExpr expr) {
    AffineSeedDependency lhs = visit(expr.getLHS());
    AffineSeedDependency rhs = visit(expr.getRHS());
    if (auto rhsConstant = dyn_cast<AffineConstantExpr>(expr.getRHS())) {
      return AffineSeedDependency::scale(lhs, rhsConstant.getValue());
    }
    if (auto lhsConstant = dyn_cast<AffineConstantExpr>(expr.getLHS())) {
      return AffineSeedDependency::scale(rhs, lhsConstant.getValue());
    }
    if (lhs.isIndependent() && rhs.isIndependent()) {
      return AffineSeedDependency::getIndependent();
    }
    return AffineSeedDependency::getUnknownForDependentSeeds({lhs, rhs});
  }

  AffineSeedDependency visitFloorDivExpr(AffineBinaryOpExpr expr) {
    return visitDivOrModExpr(expr, [](const AffineSeedDependency &dependency,
                                      int64_t divisor) {
      return AffineSeedDependency::floorDiv(dependency, divisor);
    });
  }
  AffineSeedDependency visitCeilDivExpr(AffineBinaryOpExpr expr) {
    return visitDivOrModExpr(expr, [](const AffineSeedDependency &dependency,
                                      int64_t divisor) {
      return AffineSeedDependency::ceilDiv(dependency, divisor);
    });
  }
  AffineSeedDependency visitModExpr(AffineBinaryOpExpr expr) {
    return visitDivOrModExpr(expr, [](const AffineSeedDependency &dependency,
                                      int64_t divisor) {
      return AffineSeedDependency::mod(dependency, divisor);
    });
  }

private:
  AffineSeedDependency visitInvalidExpr(AffineBinaryOpExpr expr) {
    return AffineSeedDependency::getUnknown();
  }

  AffineSeedDependency visitNonLinearExpr(AffineBinaryOpExpr expr) {
    AffineSeedDependency lhs = visit(expr.getLHS());
    AffineSeedDependency rhs = visit(expr.getRHS());
    if (lhs.isIndependent() && rhs.isIndependent()) {
      return AffineSeedDependency::getIndependent();
    }
    return AffineSeedDependency::getUnknownForDependentSeeds({lhs, rhs});
  }

  AffineSeedDependency visitDivOrModExpr(
      AffineBinaryOpExpr expr,
      llvm::function_ref<AffineSeedDependency(const AffineSeedDependency &,
                                              int64_t)>
          apply) {
    AffineSeedDependency lhs = visit(expr.getLHS());
    auto rhsConstant = dyn_cast<AffineConstantExpr>(expr.getRHS());
    if (!rhsConstant) {
      return visitNonLinearExpr(expr);
    }
    return apply(lhs, rhsConstant.getValue());
  }

  AffineMap map;
  ArrayRef<AffineSeedDependency> argDeps;
};

} // namespace

AffineSeedDependency
inferAffineMapSeedDependency(AffineMap map,
                             ArrayRef<AffineSeedDependency> argDeps) {
  LocalAffineExprSeedDependencyFinder finder(map, argDeps);
  if (map.getNumResults() != 1) {
    return AffineSeedDependency::getUnknown();
  }
  return finder.visit(map.getResult(0));
}

AffineSeedDependencyAnalysis::AffineSeedDependencyAnalysis(
    DataFlowSolver &solver, SeedPositionFn seedPositionFn)
    : SparseForwardDataFlowAnalysis(solver),
      seedPositionFn(std::move(seedPositionFn)) {}

std::optional<unsigned>
AffineSeedDependencyAnalysis::getSeedPosition(Value value) const {
  if (!seedPositionFn) {
    return std::nullopt;
  }
  return seedPositionFn(value);
}

bool AffineSeedDependencyAnalysis::isSeed(Value value) const {
  return getSeedPosition(value).has_value();
}

AffineSeedDependency
AffineSeedDependencyAnalysis::getSeedDependency(Value value) const {
  std::optional<unsigned> position = getSeedPosition(value);
  assert(position && "expected seed value");
  return AffineSeedDependency::getSeed(value, *position);
}

void AffineSeedDependencyAnalysis::setLattice(
    AffineSeedDependencyLattice *lattice,
    const AffineSeedDependency &dependency) {
  propagateIfChanged(lattice, lattice->join(dependency));
}

void AffineSeedDependencyAnalysis::setToEntryState(
    AffineSeedDependencyLattice *lattice) {
  Value value = lattice->getAnchor();
  setLattice(lattice, isSeed(value) ? getSeedDependency(value)
                                    : AffineSeedDependency::getIndependent());
}

void AffineSeedDependencyAnalysis::setToUnknown(
    AffineSeedDependencyLattice *lattice) {
  setLattice(lattice, AffineSeedDependency::getUnknown());
}

void AffineSeedDependencyAnalysis::setToSeedOrUnknown(
    AffineSeedDependencyLattice *lattice) {
  Value value = lattice->getAnchor();
  setLattice(lattice, isSeed(value) ? getSeedDependency(value)
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
      setLattice(result, getSeedDependency(value));
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
  SmallVector<AffineSeedDependency> argDependencies = llvm::map_to_vector(
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
  // Region argument propagation is conservative by default. Seeds are
  // initialized explicitly and all other non-control-flow arguments are unknown.
  for (AffineSeedDependencyLattice *arg : argLattices) {
    setToSeedOrUnknown(arg);
  }
}

} // namespace mlir::iree_compiler::IREE::Util
