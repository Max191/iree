// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/AffineSeedDependencyAnalysis.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
    return AffineSeedDependency::getIndependent();
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

AffineSeedDependency inferLocalAffineSeedDependency(
    Value value, Region &scopeRegion, ArrayRef<Value> seeds,
    llvm::DenseMap<Value, AffineSeedDependency> &memo,
    llvm::DenseSet<Value> &inFlight) {
  if (auto it = memo.find(value); it != memo.end()) {
    return it->second;
  }

  for (auto [position, seed] : llvm::enumerate(seeds)) {
    if (value == seed) {
      AffineSeedDependency dependency =
          AffineSeedDependency::getSeed(value, position);
      memo.try_emplace(value, dependency);
      return dependency;
    }
  }

  Operation *definingOp = value.getDefiningOp();
  if (!definingOp || definingOp->getParentRegion() != &scopeRegion) {
    AffineSeedDependency dependency = AffineSeedDependency::getIndependent();
    memo.try_emplace(value, dependency);
    return dependency;
  }
  if (!inFlight.insert(value).second) {
    return AffineSeedDependency::getUnknown();
  }
  llvm::scope_exit eraseInFlight([&]() { inFlight.erase(value); });

  auto getOperandDeps = [&]() {
    return llvm::map_to_vector(definingOp->getOperands(), [&](Value operand) {
      return inferLocalAffineSeedDependency(operand, scopeRegion, seeds, memo,
                                            inFlight);
    });
  };

  AffineSeedDependency dependency = AffineSeedDependency::getUnknown();
  if (isa<arith::ConstantOp>(definingOp)) {
    dependency = AffineSeedDependency::getIndependent();
  } else if (auto applyOp = dyn_cast<affine::AffineApplyOp>(definingOp)) {
    dependency =
        inferAffineMapSeedDependency(applyOp.getAffineMap(), getOperandDeps());
  } else if (auto linearizeOp =
                 dyn_cast<affine::AffineLinearizeIndexOp>(definingOp)) {
    if (linearizeOp.getDynamicBasis().empty()) {
      SmallVector<OpFoldResult> paddedBasis = linearizeOp.getPaddedBasis();
      SmallVector<AffineSeedDependency> argDeps = getOperandDeps();
      dependency = AffineSeedDependency::getIndependent();
      for (auto [index, indexDep] :
           llvm::enumerate(ArrayRef(argDeps).take_front(
               linearizeOp.getMultiIndex().size()))) {
        int64_t stride = 1;
        for (OpFoldResult basis : ArrayRef(paddedBasis).drop_front(index + 1)) {
          std::optional<int64_t> constantBasis = getConstantIntValue(basis);
          if (!constantBasis ||
              llvm::MulOverflow(stride, *constantBasis, stride)) {
            dependency =
                AffineSeedDependency::getUnknownForDependentSeeds(argDeps);
            memo.try_emplace(value, dependency);
            return dependency;
          }
        }
        dependency = AffineSeedDependency::add(
            dependency, AffineSeedDependency::scale(indexDep, stride));
      }
    }
  } else if (auto addOp = dyn_cast<arith::AddIOp>(definingOp)) {
    SmallVector<AffineSeedDependency> argDeps = getOperandDeps();
    dependency = AffineSeedDependency::add(argDeps[0], argDeps[1]);
  } else if (auto subOp = dyn_cast<arith::SubIOp>(definingOp)) {
    SmallVector<AffineSeedDependency> argDeps = getOperandDeps();
    dependency = AffineSeedDependency::add(argDeps[0], argDeps[1], 1, -1);
  } else if (auto mulOp = dyn_cast<arith::MulIOp>(definingOp)) {
    SmallVector<AffineSeedDependency> argDeps = getOperandDeps();
    if (std::optional<int64_t> rhsConstant =
            getConstantIntValue(mulOp.getRhs())) {
      dependency = AffineSeedDependency::scale(argDeps[0], *rhsConstant);
    } else if (std::optional<int64_t> lhsConstant =
                   getConstantIntValue(mulOp.getLhs())) {
      dependency = AffineSeedDependency::scale(argDeps[1], *lhsConstant);
    } else if (argDeps[0].isIndependent() && argDeps[1].isIndependent()) {
      dependency = AffineSeedDependency::getIndependent();
    } else {
      dependency = AffineSeedDependency::getUnknownForDependentSeeds(argDeps);
    }
  } else if (auto indexCastOp = dyn_cast<arith::IndexCastOp>(definingOp)) {
    dependency = inferLocalAffineSeedDependency(indexCastOp.getIn(),
                                                scopeRegion, seeds, memo,
                                                inFlight);
  } else {
    // Local inference is interface/known-op based. Unsupported operations are
    // full unknown even when their operands are independent, because the
    // operation semantics may manufacture a seed-dependent result.
    dependency = AffineSeedDependency::getUnknown();
  }

  memo.try_emplace(value, dependency);
  return dependency;
}

AffineSeedDependency inferLocalAffineSeedDependency(Value value,
                                                    Region &scopeRegion,
                                                    ArrayRef<Value> seeds) {
  DenseMap<Value, AffineSeedDependency> memo;
  DenseSet<Value> inFlight;
  return inferLocalAffineSeedDependency(value, scopeRegion, seeds, memo,
                                        inFlight);
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
  // independent step contributes no seed expression, while a seed-dependent
  // step invalidates the dependent seed expressions.
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
        setLattice(ivLattice, getSeedDependency(iv));
        continue;
      }

      Block *block = iv.getParentBlock();
      std::optional<AffineSeedDependency> lbDependency =
          getDependencyFromOfr(lb, block);
      std::optional<AffineSeedDependency> stepDependency =
          getDependencyFromOfr(step, block);
      if (!lbDependency || !stepDependency) {
        // Do not pessimize this to unknown: joins are monotonic and unknown
        // cannot refine back to a precise affine expression. The lattice query
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
