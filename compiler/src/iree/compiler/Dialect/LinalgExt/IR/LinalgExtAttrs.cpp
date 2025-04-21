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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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
// TransposeIndicesAttr
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

SmallVector<OpFoldResult>
TransposeIndicesAttr::transformIndices(OpBuilder &b, Location loc,
                                       ArrayRef<OpFoldResult> indices,
                                       ArrayRef<Value> dynamicValues) const {
  return applyPermutation(indices, getPermutation());
}

//===----------------------------------------------------------------------===//
// LinearizeIndicesAttr
//===----------------------------------------------------------------------===//

int64_t LinearizeIndicesAttr::getNumResultIndices() const {
  return 1;
}

int64_t LinearizeIndicesAttr::getNumInputIndices() const {
  return getBasis().size();
}

int64_t LinearizeIndicesAttr::getNumDynamicIndices() const {
  return llvm::count_if(getBasis(),
      [](int64_t dim) { return ShapedType::isDynamic(dim); });
}

SmallVector<OpFoldResult>
LinearizeIndicesAttr::transformIndices(OpBuilder &b, Location loc,
                                       ArrayRef<OpFoldResult> indices,
                                       ArrayRef<Value> dynamicValues) const {
  SmallVector<Value> indexValues =
      getValueOrCreateConstantIndexOp(b, loc, indices);
  SmallVector<OpFoldResult> mixedBasis =
      getMixedValues(getBasis(), dynamicValues, getContext());
  auto linearizeOp =
      b.create<affine::AffineLinearizeIndexOp>(loc, indexValues, mixedBasis);
  return linearizeOp->getResults();
}

//===----------------------------------------------------------------------===//
// DelinearizeIndicesAttr
//===----------------------------------------------------------------------===//

int64_t DelinearizeIndicesAttr::getNumResultIndices() const {
  return getBasis().size();
}

int64_t DelinearizeIndicesAttr::getNumInputIndices() const {
  return 1;
}

int64_t DelinearizeIndicesAttr::getNumDynamicIndices() const {
  return llvm::count_if(getBasis(),
      [](int64_t dim) { return ShapedType::isDynamic(dim); });
}

SmallVector<OpFoldResult>
DelinearizeIndicesAttr::transformIndices(OpBuilder &b, Location loc,
                                         ArrayRef<OpFoldResult> indices,
                                         ArrayRef<Value> dynamicValues) const {
  assert(indices.size() == 1 && "expected only a single index");
  Value index = getValueOrCreateConstantIndexOp(b, loc, indices[0]);
  SmallVector<OpFoldResult> mixedBasis =
      getMixedValues(getBasis(), dynamicValues, getContext());
  auto delinearizeOp =
      b.create<affine::AffineDelinearizeIndexOp>(loc, index, mixedBasis);
  return delinearizeOp->getResults();
}

//===----------------------------------------------------------------------===//
// ClampIndicesAttr
//===----------------------------------------------------------------------===//

int64_t ClampIndicesAttr::getNumResultIndices() const {
  return getBounds().size();
}

int64_t ClampIndicesAttr::getNumInputIndices() const {
  return getBounds().size();
}

int64_t ClampIndicesAttr::getNumDynamicIndices() const {
  return llvm::count_if(getBounds(),
      [](int64_t dim) { return ShapedType::isDynamic(dim); });
}

SmallVector<OpFoldResult>
ClampIndicesAttr::transformIndices(OpBuilder &b, Location loc,
                                   ArrayRef<OpFoldResult> indices,
                                   ArrayRef<Value> dynamicValues) const {
  return SmallVector<OpFoldResult>(indices);
}

//===----------------------------------------------------------------------===//
// AddIndicesAttr
//===----------------------------------------------------------------------===//

int64_t AddIndicesAttr::getNumResultIndices() const {
  return getValues().size();
}

int64_t AddIndicesAttr::getNumInputIndices() const {
  return getValues().size();
}

int64_t AddIndicesAttr::getNumDynamicIndices() const {
  return llvm::count_if(getValues(),
                        [](int64_t dim) { return ShapedType::isDynamic(dim); });
}

SmallVector<OpFoldResult>
AddIndicesAttr::transformIndices(OpBuilder &b, Location loc,
                                 ArrayRef<OpFoldResult> indices,
                                 ArrayRef<Value> dynamicValues) const {
  SmallVector<OpFoldResult> mixedValues =
      getMixedValues(getValues(), dynamicValues, getContext());
  AffineMap addMap = AffineMap::get(
      2, 0, {b.getAffineDimExpr(0) + b.getAffineDimExpr(1)}, getContext());
  SmallVector<OpFoldResult> transformedIndices;
  for (auto [index, value] : llvm::zip_equal(indices, mixedValues)) {
    transformedIndices.push_back(
        affine::makeComposedFoldedAffineApply(b, loc, addMap, {index, value}));
  }
  return transformedIndices;
}

//===----------------------------------------------------------------------===//
// custom<DynamicI64ArrayAttr>
//===----------------------------------------------------------------------===//

ParseResult parseDynamicI64ArrayAttr(AsmParser &p,
                                     SmallVector<int64_t> &array) {
  if (failed(p.parseLSquare()))
    return failure();
  if (failed(p.parseCommaSeparatedList([&] {
        int64_t value = ShapedType::kDynamic;
        if (failed(p.parseOptionalQuestion()) &&
            failed(p.parseInteger(value))) {
          return failure();
        }
        array.push_back(value);
        return success();
      }))) {
    return failure();
  }
  if (failed(p.parseRSquare()))
    return failure();
  return success();
}

void printDynamicI64ArrayAttr(AsmPrinter &p, ArrayRef<int64_t> attrs) {
  p << "[";
  llvm::interleaveComma(attrs, p, [&](int64_t value) {
    if (ShapedType::isDynamic(value)) {
      p << "?";
    } else {
      p << value;
    }
  });
  p << "]";
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
