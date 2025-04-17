// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtAttrs.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"

#define DEBUG_TYPE "iree-linalg-ext-attrs"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtAttrs.cpp.inc"

namespace mlir::iree_compiler::IREE::LinalgExt {

//===----------------------------------------------------------------------===//
// Index Transformation Attributes
//===----------------------------------------------------------------------===//

int64_t TransposeIndicesAttr::getNumResultIndices() const {
  return getPermutation().size();
}

int64_t TransposeIndicesAttr::getNumInputIndices() const {
  return getPermutation().size();
}

int64_t TransposeIndicesAttr::getNumDynamicIndices() const {
  return 0;
}

LogicalResult
TransposeIndicesAttr::verify(function_ref<mlir::InFlightDiagnostic()> emitError,
                             ArrayRef<int64_t> permutation) {
  if (!isPermutationVector(permutation)) {
    return emitError() << "expected valid permutation";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Attribute Registration
//===----------------------------------------------------------------------===//

void IREELinalgExtDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtAttrs.cpp.inc" // IWYU pragma: keep
      >();
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
