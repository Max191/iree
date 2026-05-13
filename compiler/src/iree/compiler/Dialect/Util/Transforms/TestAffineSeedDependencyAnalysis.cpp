// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/AffineSeedDependencyAnalysis.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"

namespace mlir::iree_compiler::IREE::Util {

#define GEN_PASS_DEF_TESTAFFINESEEDDEPENDENCYANALYSISPASS
#include "iree/compiler/Dialect/Util/Transforms/Passes.h.inc"

namespace {

static constexpr StringLiteral kQueryOpName =
    "iree_unregistered.test_affine_seed_dependency";

class TestAffineSeedDependencyAnalysisPass final
    : public impl::TestAffineSeedDependencyAnalysisPassBase<
          TestAffineSeedDependencyAnalysisPass> {
public:
  void runOnOperation() override {
    Operation *rootOp = getOperation();
    MLIRContext *context = &getContext();

    SmallVector<Operation *> queryOps;
    llvm::DenseMap<Value, unsigned> seedPositions;
    rootOp->walk([&](Operation *op) {
      if (op->getName().getStringRef() != kQueryOpName ||
          op->getNumOperands() < 1) {
        return;
      }
      queryOps.push_back(op);
      for (Value seed : op->getOperands().drop_front()) {
        seedPositions.try_emplace(seed,
                                  static_cast<unsigned>(seedPositions.size()));
      }
    });

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<AffineSeedDependencyAnalysis>(
        [&](Value value) -> std::optional<unsigned> {
          auto it = seedPositions.find(value);
          if (it == seedPositions.end()) {
            return std::nullopt;
          }
          return it->second;
        });
    if (failed(solver.initializeAndRun(rootOp))) {
      return signalPassFailure();
    }

    for (Operation *queryOp : queryOps) {
      Value queryValue = queryOp->getOperand(0);
      const auto *lattice =
          solver.lookupState<AffineSeedDependencyLattice>(queryValue);
      queryOp->setAttr(
          "affine_seed_dependency",
          StringAttr::get(context, formatDependency(lattice, queryOp)));
      queryOp->setAttr(
          "affine_seed_expression",
          StringAttr::get(context, formatExpression(lattice, queryOp)));
    }
  }

private:
  static std::string formatExpression(
      const AffineSeedDependencyLattice *lattice, Operation *queryOp) {
    if (!lattice || lattice->getValue().isUninitialized()) {
      return "uninitialized";
    }
    const AffineSeedDependency &dependency = lattice->getValue();
    if (dependency.isUnknown()) {
      return "unknown";
    }

    std::string result;
    llvm::raw_string_ostream os(result);
    os << dependency.getExpression(queryOp->getContext());
    if (dependency.hasInvalidatedSeeds()) {
      os << " invalidated";
    }
    return os.str();
  }

  static std::string formatDependency(
      const AffineSeedDependencyLattice *lattice, Operation *queryOp) {
    if (!lattice || lattice->getValue().isUninitialized()) {
      return "uninitialized";
    }
    const AffineSeedDependency &dependency = lattice->getValue();
    if (dependency.isUnknown()) {
      return "unknown";
    }

    std::string result;
    llvm::raw_string_ostream os(result);
    int64_t index = 0;
    StringRef separator;
    for (Value seed : queryOp->getOperands().drop_front()) {
      os << separator << "seed" << index << " = ";
      std::optional<int64_t> coefficient = dependency.getCoefficient(seed);
      if (coefficient) {
        os << *coefficient;
      } else {
        os << "?";
      }
      separator = ", ";
      ++index;
    }
    return os.str();
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Util
