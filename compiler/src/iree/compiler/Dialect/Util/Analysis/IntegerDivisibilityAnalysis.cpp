// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/IntegerDivisibilityAnalysis.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

#define DEBUG_TYPE "iree-util-int-divisibility-analysis"

using llvm::dbgs;

namespace mlir::iree_compiler::IREE::Util {

void IntegerDivisibilityAnalysis::setToEntryState(
    IntegerDivisibilityLattice *lattice) {
  propagateIfChanged(lattice,
                     lattice->join(IntegerDivisibility::getMinDivisibility()));
}

LogicalResult IntegerDivisibilityAnalysis::visitOperation(
    Operation *op, ArrayRef<const IntegerDivisibilityLattice *> operands,
    ArrayRef<IntegerDivisibilityLattice *> results) {
  auto inferrable = dyn_cast<InferIntDivisibilityOpInterface>(op);
  if (!inferrable) {
    setAllToEntryStates(results);
    return success();
  }

  LLVM_DEBUG(dbgs() << "Inferring divisibility for " << *op << "\n");
  auto argDivs = llvm::map_to_vector(
      operands, [](const IntegerDivisibilityLattice *lattice) {
        return lattice->getValue();
      });
  auto joinCallback = [&](Value v, const IntegerDivisibility &newDiv) {
    auto result = dyn_cast<OpResult>(v);
    if (!result)
      return;
    assert(llvm::is_contained(op->getResults(), result));

    LLVM_DEBUG(dbgs() << "Inferred divisibility " << newDiv << "\n");
    IntegerDivisibilityLattice *lattice = results[result.getResultNumber()];
    IntegerDivisibility oldDiv = lattice->getValue();

    ChangeResult changed = lattice->join(newDiv);

    // Catch loop results with loop variant bounds and conservatively make
    // them [-inf, inf] so we don't circle around infinitely often (because
    // the dataflow analysis in MLIR doesn't attempt to work out trip counts
    // and often can't).
    bool isYieldedResult = llvm::any_of(v.getUsers(), [](Operation *op) {
      return op->hasTrait<OpTrait::IsTerminator>();
    });
    if (isYieldedResult && !oldDiv.isUninitialized() &&
        !(lattice->getValue() == oldDiv)) {
      changed |= lattice->join(IntegerDivisibility::getMinDivisibility());
    }
    propagateIfChanged(lattice, changed);
  };

  inferrable.inferResultDivisibility(argDivs, joinCallback);
  return success();
}

void IntegerDivisibilityAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor,
    ArrayRef<IntegerDivisibilityLattice *> argLattices, unsigned firstIndex) {
  auto getDivFromConstants = [&](std::optional<OpFoldResult> loopBound,
                                 Block *block, bool isUnsigned) -> uint64_t {
    if (loopBound.has_value()) {
      if (auto constBound = getConstantIntValue(*loopBound)) {
        return constBound.value();
      }
      auto value = cast<Value>(loopBound.value());
      const IntegerDivisibilityLattice *lattice =
          getLatticeElementFor(getProgramPointBefore(block), value);
      if (lattice != nullptr && !lattice->getValue().isUninitialized())
        return isUnsigned ? lattice->getValue().getValue().udiv()
                          : lattice->getValue().getValue().sdiv();
    }
    return isUnsigned
               ? IntegerDivisibility::getMinDivisibility().getValue().udiv()
               : IntegerDivisibility::getMinDivisibility().getValue().sdiv();
  };

  // Infer bounds for loop arguments that have static bounds
  if (auto loop = dyn_cast<LoopLikeOpInterface>(op)) {
    std::optional<Value> iv = loop.getSingleInductionVar();
    if (!iv) {
      return SparseForwardDataFlowAnalysis ::visitNonControlFlowArguments(
          op, successor, argLattices, firstIndex);
    }
    Block *block = iv->getParentBlock();
    std::optional<OpFoldResult> lb = loop.getSingleLowerBound();
    std::optional<OpFoldResult> step = loop.getSingleStep();

    IntegerDivisibilityLattice *ivEntry = getLatticeElement(*iv);
    uint64_t stepUDiv = getDivFromConstants(step, block, /*unsigned=*/true);
    uint64_t stepSDiv = getDivFromConstants(step, block, /*unsigned=*/false);
    // if (lb.has_value() && isConstantIntValue(*lb, 0)) {
    //   ConstantIntDivisibility ivDiv(stepUDiv, stepSDiv);
    //   propagateIfChanged(ivEntry, ivEntry->join(ivDiv));
    //   return;
    // }

    uint64_t lbUDiv = getDivFromConstants(lb, block, /*unsigned=*/true);
    uint64_t lbSDiv = getDivFromConstants(lb, block, /*unsigned=*/false);
    ConstantIntDivisibility lbDiv(lbUDiv, lbSDiv);
    ConstantIntDivisibility stepDiv(stepUDiv, stepSDiv);
    ConstantIntDivisibility ivDiv = stepDiv.getUnion(lbDiv);
    propagateIfChanged(ivEntry, ivEntry->join(ivDiv));
    return;
  }

  return SparseForwardDataFlowAnalysis::visitNonControlFlowArguments(
      op, successor, argLattices, firstIndex);
}

} // namespace mlir::iree_compiler::IREE::Util
