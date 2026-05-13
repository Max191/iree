// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/AffineSeedDependencyAnalysis.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

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

// Returns true if a known dependency has no invalidated seeds and every tracked
// seed has affine coefficient zero. Dynamic seed-independent offsets are
// allowed because they do not affect the seed relationship.
static bool hasKnownZeroSeedCoefficients(
    const AffineSeedDependency &dependency) {
  if (!dependency.isKnown() || dependency.hasInvalidatedSeeds()) {
    return false;
  }
  for (const auto &seedPosition : dependency.getSeedPositions()) {
    Value seed = seedPosition.first;
    std::optional<int64_t> coefficient = dependency.getCoefficient(seed);
    if (!coefficient || *coefficient != 0) {
      return false;
    }
  }
  return true;
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
  // Static loop bounds and steps are independent of all seeds. Dynamic values
  // are queried before the loop body and may require a later solver revisit if
  // their producer has not been initialized yet.
  auto getDependencyFromOfr =
      [&](std::optional<OpFoldResult> ofr,
          Block *block) -> std::optional<AffineSeedDependency> {
    if (!ofr) {
      return AffineSeedDependency::getIndependent();
    }
    if (getConstantIntValue(*ofr)) {
      return AffineSeedDependency::getIndependent();
    }
    if (isa<Attribute>(*ofr)) {
      return AffineSeedDependency::getUnknown();
    }
    Value value = cast<Value>(*ofr);
    const AffineSeedDependencyLattice *lattice =
        getLatticeElementFor(getProgramPointBefore(block), value);
    if (!lattice || lattice->getValue().isUninitialized()) {
      return std::nullopt;
    }
    return lattice->getValue();
  };

  if (auto loop = dyn_cast<LoopLikeOpInterface>(op)) {
    // This path is intended for scf.for/scf.forall-style LoopLike ops that
    // expose complete induction variable metadata with
    // `iv = lowerBound + iteration * step` semantics. For those ops,
    // `visitNonControlFlowArguments` receives loop non-successor inputs such as
    // induction variables. Loop-carried values are still handled by the sparse
    // dataflow framework through successor operand joins.
    std::optional<SmallVector<Value>> ivs = loop.getLoopInductionVars();
    std::optional<SmallVector<OpFoldResult>> lbs = loop.getLoopLowerBounds();
    std::optional<SmallVector<OpFoldResult>> ubs = loop.getLoopUpperBounds();
    std::optional<SmallVector<OpFoldResult>> steps = loop.getLoopSteps();
    if (!ivs || !lbs || !ubs || !steps) {
      // Non-standard LoopLike ops without complete IV metadata use the default
      // sparse dataflow handling rather than guessing loop semantics.
      return SparseForwardDataFlowAnalysis::visitNonControlFlowArguments(
          op, successor, successorInputs, argLattices);
    }

    llvm::SmallDenseSet<Value> loopIvs;
    for (auto [iv, lb, ub, step] : llvm::zip_equal(*ivs, *lbs, *ubs, *steps)) {
      loopIvs.insert(iv);
      AffineSeedDependencyLattice *ivLattice = getLatticeElement(iv);
      if (isSeed(iv)) {
        setLattice(ivLattice, getSeedDependency(iv));
        continue;
      }

      Block *block = iv.getParentBlock();
      std::optional<AffineSeedDependency> lbDependency =
          getDependencyFromOfr(lb, block);
      std::optional<AffineSeedDependency> ubDependency =
          getDependencyFromOfr(ub, block);
      std::optional<AffineSeedDependency> stepDependency =
          getDependencyFromOfr(step, block);
      if (!lbDependency || !ubDependency || !stepDependency) {
        // Do not pessimize this to unknown. `getLatticeElementFor` records a
        // sparse dataflow dependency on the producer state at the loop-body
        // entry program point, and SparseForwardDataFlowAnalysis revisits this
        // transfer once that state is initialized. A direct lit reproducer for
        // the first-visit uninitialized state would depend on solver
        // scheduling, so the test suite checks final fixpoint behavior instead.
        LDBG() << "Deferring LoopLike induction variable dependency for "
               << iv << " until bound states are initialized\n";
        continue;
      }

      AffineSeedDependency ivDependency = *lbDependency;
      AffineSeedDependency extentDependency =
          AffineSeedDependency::add(*ubDependency, *lbDependency, 1, -1);
      if (!hasKnownZeroSeedCoefficients(extentDependency)) {
        // Upper bounds do not directly contribute to the scalar recurrence, but
        // the loop extent controls which IV values exist. Treat seed-dependent
        // extents as control dependence on the IV.
        ivDependency = AffineSeedDependency::add(
            ivDependency, AffineSeedDependency::getUnknownForDependentSeeds(
                              {extentDependency}));
      }
      if (!stepDependency->isIndependent()) {
        // The iteration count is not represented in this analysis. A
        // seed-dependent step therefore conservatively invalidates separable
        // affine coefficients for any seeds that may affect the step while
        // preserving any lower-bound relationship that is still known.
        ivDependency = AffineSeedDependency::add(
            ivDependency, AffineSeedDependency::getUnknownForDependentSeeds(
                              {*stepDependency}));
      }
      setLattice(ivLattice, ivDependency);
    }

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
