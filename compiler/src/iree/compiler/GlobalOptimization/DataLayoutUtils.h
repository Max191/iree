// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef IREE_GLOBALOPTIMIZATION_DATALAYOUTUTILS_H_
#define IREE_GLOBALOPTIMIZATION_DATALAYOUTUTILS_H_

#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class Location;
class OpBuilder;
class Operation;
class RewriterBase;
class Value;
} // namespace mlir

namespace mlir::iree_compiler::GlobalOptimization {

/// INTERMEDIATE: It is possible to propagate through this node.
/// BARRIER: It is not possible to propagate through this node.
enum class DataLayoutNodeType {
  UNINITIALIZED,
  INTERMEDIATE,
  BARRIER,
};

// struct DataLayoutNodeInfo {
//   int64_t layoutId;
//   DataLayoutNodeType nodeType;
// };

/// TODO: Abstractify DataLayoutTransformation to decouple from specific types
/// of transformations.
class DataLayoutTransformation {
public:
  DataLayoutTransformation(ShapedType orig, ShapedType transformed)
      : originalType(orig), transformedType(transformed){};
  DataLayoutTransformation(ShapedType orig) : originalType(orig){};
  DataLayoutTransformation(DataLayoutTransformation &other) {
    originalType = other.originalType;
    transformedType = other.transformedType;
    reshapeReInd = other.reshapeReInd;
    innerDimsPos = other.innerDimsPos;
    innerTileSizes = other.innerTileSizes;
    outerDimsPerm = other.outerDimsPerm;
    correspondingTransformedIndices = other.correspondingTransformedIndices;
  };
  DataLayoutTransformation(){};
  /// Get the transformed layout at the `newValue`, given the `currentValue`
  /// with `this` layout.
  static DataLayoutTransformation *getIdentityTransformation(ShapedType type) {
    auto *tf = new DataLayoutTransformation(type, type);
    tf->setCorrespondingTransformedIndices(
        llvm::to_vector(llvm::seq<int64_t>(0, type.getRank())));
    return tf;
  }
  bool transformLayout(Value currentValue, Value newValue);
  bool combineLayout(DataLayoutTransformation other);
  const bool hasValidTransform();
  ShapedType getOriginalType() const { return originalType; };
  ShapedType getTransformedType() const { return transformedType; };
  SmallVector<ReassociationIndices> getReshapeReInd() const {
    return reshapeReInd;
  };
  SmallVector<int64_t> getInnerDimsPos() const { return innerDimsPos; };
  SmallVector<int64_t> getInnerTileSizes() const { return innerTileSizes; };
  SmallVector<int64_t> getOuterDimsPerm() const { return outerDimsPerm; };
  SmallVector<int64_t> getCorrespondingTransformedIndices() const {
    return correspondingTransformedIndices;
  };
  void setOriginalType(ShapedType type) { originalType = type; };
  void setTransformedType(ShapedType type) { transformedType = type; };
  void setReshapeReInd(SmallVector<ReassociationIndices> inds) {
    reshapeReInd = inds;
  };
  void setInnerDimsPos(SmallVector<int64_t> pos) { innerDimsPos = pos; };
  void setInnerTileSizes(SmallVector<int64_t> tiles) {
    innerTileSizes = tiles;
  };
  void setOuterDimsPerm(SmallVector<int64_t> perm) { outerDimsPerm = perm; };
  void setCorrespondingTransformedIndices(SmallVector<int64_t> inds) {
    correspondingTransformedIndices = inds;
  };
  bool isIntersecting(DataLayoutTransformation other);
  ArrayAttr makeTransformArrayAttr(MLIRContext *ctx);

private:
  /// Transform the layout as if propagating through an operation, from the
  /// `currentValue` to `newValue`, and return the new layout.
  bool transform(Operation *op, Value currentValue, Value newValue);

  ShapedType originalType;
  ShapedType transformedType;
  /// Transformation from `originalType`->`transformedType`.
  SmallVector<ReassociationIndices> reshapeReInd;
  SmallVector<int64_t> innerDimsPos;
  SmallVector<int64_t> innerTileSizes;
  SmallVector<int64_t> outerDimsPerm;
  /// Indices in the `originalType` corresponding to each index of the value
  /// associated with the DataLayoutTransformation.
  SmallVector<int64_t> correspondingTransformedIndices;
};
RankedTensorType getPackedSliceType(
    RankedTensorType packedSourceType, SmallVector<OpFoldResult> sliceSizes,
    llvm::SmallDenseSet<unsigned> rankReductionMask,
    ArrayRef<int64_t> outerDimsPerm, ArrayRef<int64_t> innerDimsPos);
FailureOr<Value>
packSliceOfTensor(PatternRewriter &rewriter, Value slice,
                  SmallVector<OpFoldResult> sliceSizes,
                  llvm::SmallDenseSet<unsigned> rankReductionMask,
                  tensor::PackOp packOp);
FailureOr<Value>
unPackSliceOfTensor(PatternRewriter &rewriter, Value slice,
                    SmallVector<OpFoldResult> sliceSizes,
                    llvm::SmallDenseSet<unsigned> rankReductionMask,
                    tensor::UnPackOp unpackOp,
                    SmallVector<OpFoldResult> destMixedSizes,
                    RankedTensorType originalSliceType);
void getCollapsedSliceMetadata(
    ArrayRef<int64_t> tensorShape, ArrayRef<int64_t> sliceShape,
    SmallVector<OpFoldResult> mixedOffsets,
    SmallVector<OpFoldResult> mixedSizes,
    SmallVector<OpFoldResult> mixedStrides,
    SmallVector<OpFoldResult> &collapsedOffsets,
    SmallVector<OpFoldResult> &collapsedSizes,
    SmallVector<OpFoldResult> &collapsedStrides,
    std::optional<SmallVector<ReassociationIndices>> &collapsedTensorRe,
    std::optional<SmallVector<ReassociationIndices>> &collapsedSliceRe,
    SmallVector<int64_t> &collapsedSliceShape);
DataLayoutNodeType getNodeTypeForValue(Value value);
SmallVector<StringRef> getTerminalNodeIDs(Value value);
// Padding values can get in the way of unpack(pack(x)) foldings, so we
// explicitly allow this pack to fold despite the padding_value with this
// function.
void setFoldablePackUnPackAttribute(Operation *op);
bool hasFoldablePackUnPackAttribute(Operation *op);
std::optional<DataLayoutNodeType> getNodeTypeFromAttr(Operation *op);
void setNodeTypeAttribute(Operation *op, DataLayoutNodeType nodeType);
void setDataLayoutTransformationAttributes(Operation *op,
                                           DataLayoutTransformation *transform,
                                           StringRef transformID);
LogicalResult transformGlobalsToNewLayout(IRRewriter &rewriter,
                                          SmallVector<Value> edgeNodes,
                                          DataLayoutTransformation *transform,
                                          IREE::Util::GlobalOp global,
                                          SymbolTable moduleSymbols);

} // namespace mlir::iree_compiler::GlobalOptimization

#endif // IREE_GLOBALOPTIMIZATION_DATALAYOUTUTILS_H_
