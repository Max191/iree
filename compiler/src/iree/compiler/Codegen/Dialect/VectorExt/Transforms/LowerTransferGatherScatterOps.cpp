// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Transforms.h"
#include "iree/compiler/Utils/AffineExprUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/Support/CheckedArithmetic.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::VectorExt;

namespace {

/// Remove dim 0 from an AffineMap by:
/// 1. Replacing AffineDimExpr(0) with AffineConstantExpr(0)
/// 2. Renumbering AffineDimExpr(k) where k > 0 to AffineDimExpr(k-1)
/// 3. Reducing numDims by 1
static AffineMap removeDim0FromMap(AffineMap map) {
  MLIRContext *ctx = map.getContext();
  SmallVector<AffineExpr> dimReplacements;
  for (unsigned i = 0, e = map.getNumDims(); i < e; ++i) {
    if (i == 0) {
      dimReplacements.push_back(getAffineConstantExpr(0, ctx));
    } else {
      dimReplacements.push_back(getAffineDimExpr(i - 1, ctx));
    }
  }
  return map.replaceDimsAndSymbols(dimReplacements, /*symReplacements=*/{},
                                   map.getNumDims() - 1,
                                   map.getNumSymbols());
}

/// Remove dim 0 references from an index vec map. Returns the new map with
/// results that referenced dim 0 dropped, and the axis positions in the index
/// vec that need to be sliced.
static AffineMap removeDim0FromIndexVecMap(AffineMap map,
                                           SmallVectorImpl<int64_t> &axes) {
  MLIRContext *ctx = map.getContext();
  SmallVector<AffineExpr> newResults;
  for (auto [resultIdx, expr] : llvm::enumerate(map.getResults())) {
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
      unsigned pos = dimExpr.getPosition();
      if (pos == 0) {
        axes.push_back(resultIdx);
        continue;
      }
      newResults.push_back(getAffineDimExpr(pos - 1, ctx));
    } else {
      newResults.push_back(expr);
    }
  }
  return AffineMap::get(map.getNumDims() - 1, map.getNumSymbols(), newResults,
                        ctx);
}

/// Extract a slice from a vector at position `idx` along the given `axis`.
/// For a vector<4x8xindex>, extracting axis=0, idx=2 gives vector<8xindex>.
static Value extractVecSlice(OpBuilder &b, Location loc, Value vec,
                             int64_t axis, int64_t idx) {
  auto vecType = cast<VectorType>(vec.getType());
  int64_t rank = vecType.getRank();

  if (axis == 0) {
    // Extracting from rank-1 along axis 0 gives a scalar.
    return vector::ExtractOp::create(b, loc, vec, int64_t{idx});
  }

  // General case: use extract_strided_slice.
  SmallVector<int64_t> offsets(rank, 0);
  SmallVector<int64_t> sizes(vecType.getShape());
  SmallVector<int64_t> strides(rank, 1);
  offsets[axis] = idx;
  sizes[axis] = 1;
  Value slice = vector::ExtractStridedSliceOp::create(b, loc, vec, offsets,
                                                      sizes, strides);
  // Drop the unit dim.
  SmallVector<int64_t> newShape;
  for (int64_t i = 0; i < rank; ++i) {
    if (i != axis) {
      newShape.push_back(vecType.getShape()[i]);
    }
  }
  auto newType = VectorType::get(newShape, vecType.getElementType());
  return vector::ShapeCastOp::create(b, loc, newType, slice);
}

//===----------------------------------------------------------------------===//
// Shared unroll helpers
//===----------------------------------------------------------------------===//

struct BaseDim0Coefficient {
  int64_t baseResultIndex;
  int64_t dim0Coefficient;
};

/// Compute dim-0-removed indexing maps for unrolling. Populates the new base
/// map, per-index-vec maps and axes, mask map and axes, linear dim-0
/// coefficients for base-map results, and the combined new indexing maps array.
/// Fails if any base-map result has a non-linear or otherwise unsupported dim-0
/// contribution.
static LogicalResult
computeUnrollDim0Maps(ArrayRef<AffineMap> indexingMaps, int64_t numIndexVecs,
                      bool hasMask, AffineMap baseMap,
                      SmallVectorImpl<AffineMap> &newAllMaps,
                      SmallVectorImpl<SmallVector<int64_t>> &indexVecAxes,
                      SmallVectorImpl<int64_t> &maskAxes,
                      SmallVectorImpl<BaseDim0Coefficient>
                          &baseDim0Coefficients) {
  AffineMap newBaseMap = removeDim0FromMap(baseMap);
  newAllMaps.push_back(newBaseMap);

  for (int64_t i = 0; i < numIndexVecs; ++i) {
    SmallVector<int64_t> axes;
    AffineMap newMap = removeDim0FromIndexVecMap(indexingMaps[1 + i], axes);
    newAllMaps.push_back(newMap);
    indexVecAxes.push_back(std::move(axes));
  }

  if (hasMask) {
    AffineMap newMaskMap =
        removeDim0FromIndexVecMap(indexingMaps.back(), maskAxes);
    newAllMaps.push_back(newMaskMap);
  }

  for (auto [j, expr] : llvm::enumerate(baseMap.getResults())) {
    std::optional<int64_t> coefficient =
        mlir::iree_compiler::getAffineDimCoefficient(expr, 0);
    if (!coefficient) {
      return failure();
    }
    if (*coefficient != 0) {
      baseDim0Coefficients.push_back({static_cast<int64_t>(j),
                                      *coefficient});
    }
  }
  return success();
}

/// Extract sliced index vecs and mask for iteration `i` of dim-0 unrolling.
/// Returns the sliced mask (or nullptr if no mask).
static Value
extractSlicesForIteration(OpBuilder &rewriter, Location loc, int64_t i,
                          OperandRange indexVecs, int64_t numIndexVecs,
                          ArrayRef<SmallVector<int64_t>> indexVecAxes,
                          Value mask, ArrayRef<int64_t> maskAxes,
                          SmallVectorImpl<Value> &newIndexVecs) {
  for (int64_t k = 0; k < numIndexVecs; ++k) {
    Value idxVec = indexVecs[k];
    if (!indexVecAxes[k].empty()) {
      for (int64_t axis : indexVecAxes[k]) {
        idxVec = extractVecSlice(rewriter, loc, idxVec, axis, i);
      }
    }
    newIndexVecs.push_back(idxVec);
  }

  if (!mask) {
    return nullptr;
  }
  Value m = mask;
  for (int64_t axis : maskAxes) {
    m = extractVecSlice(rewriter, loc, m, axis, i);
  }
  return m;
}

/// Update base offsets for dim-0 iteration `i`.
static FailureOr<SmallVector<Value>>
computeNewOffsets(OpBuilder &rewriter, Location loc, ValueRange offsets,
                  int64_t i,
                  ArrayRef<BaseDim0Coefficient> baseDim0Coefficients) {
  SmallVector<Value> newOffsets(offsets);
  for (BaseDim0Coefficient coefficient : baseDim0Coefficients) {
    Value offset = newOffsets[coefficient.baseResultIndex];
    std::optional<int64_t> scaledDim0Offset =
        llvm::checkedMul(i, coefficient.dim0Coefficient);
    if (!scaledDim0Offset) {
      return failure();
    }
    Value iVal =
        arith::ConstantIndexOp::create(rewriter, loc, *scaledDim0Offset);
    newOffsets[coefficient.baseResultIndex] =
        arith::AddIOp::create(rewriter, loc, offset, iVal);
  }
  return newOffsets;
}

static OpFoldResult computeProduct(OpBuilder &builder, Location loc,
                                   ArrayRef<OpFoldResult> values) {
  int64_t staticProduct = 1;
  Value dynamicProduct;
  auto flushStaticProduct = [&]() {
    if (staticProduct == 1) {
      return;
    }
    Value staticProductValue =
        arith::ConstantIndexOp::create(builder, loc, staticProduct);
    dynamicProduct =
        dynamicProduct
            ? arith::MulIOp::create(builder, loc, dynamicProduct,
                                    staticProductValue)
                  .getResult()
            : staticProductValue;
    staticProduct = 1;
  };
  for (OpFoldResult value : values) {
    if (std::optional<int64_t> constant = getConstantIntValue(value)) {
      if (std::optional<int64_t> product =
              llvm::checkedMul(staticProduct, *constant)) {
        staticProduct = *product;
        continue;
      }
      flushStaticProduct();
      Value constantValue =
          arith::ConstantIndexOp::create(builder, loc, *constant);
      dynamicProduct =
          dynamicProduct
              ? arith::MulIOp::create(builder, loc, dynamicProduct,
                                      constantValue)
                    .getResult()
              : constantValue;
      continue;
    }
    Value dynamicValue = dyn_cast<Value>(value);
    assert(dynamicValue && "expected dynamic size to be a value");
    flushStaticProduct();
    dynamicProduct =
        dynamicProduct
            ? arith::MulIOp::create(builder, loc, dynamicProduct, dynamicValue)
                  .getResult()
            : dynamicValue;
  }
  if (!dynamicProduct) {
    return builder.getIndexAttr(staticProduct);
  }
  if (staticProduct == 1) {
    return dynamicProduct;
  }
  Value staticProductValue =
      arith::ConstantIndexOp::create(builder, loc, staticProduct);
  return arith::MulIOp::create(builder, loc, dynamicProduct, staticProductValue)
      .getResult();
}

/// Flatten the scatter base. If the memref cannot be collapsed directly,
/// reinterpret it as a 1D strided view and return the original strides.
static Value createFlatScatterBase(RewriterBase &rewriter, Location loc,
                                   Value base,
                                   SmallVectorImpl<Value> &strides) {
  auto baseType = cast<MemRefType>(base.getType());
  if (baseType.getRank() == 1) {
    return base;
  }
  SmallVector<ReassociationIndices> reassociations;
  reassociations.push_back(
      llvm::to_vector(llvm::seq<int64_t>(baseType.getRank())));
  if (memref::CollapseShapeOp::isGuaranteedCollapsible(baseType,
                                                       reassociations)) {
    return memref::CollapseShapeOp::create(rewriter, loc, base,
                                           reassociations);
  }

  SmallVector<OpFoldResult> sizes = memref::getMixedSizes(rewriter, loc, base);
  auto stridedMetadataOp =
      memref::ExtractStridedMetadataOp::create(rewriter, loc, base);
  strides.append(stridedMetadataOp.getStrides().begin(),
                 stridedMetadataOp.getStrides().end());
  OpFoldResult collapsedSize = computeProduct(rewriter, loc, sizes);
  SmallVector<OpFoldResult> collapsedShape = {collapsedSize};
  SmallVector<OpFoldResult> collapsedStrides = {rewriter.getIndexAttr(1)};
  return memref::ReinterpretCastOp::create(rewriter, loc, base,
                                           stridedMetadataOp.getOffset(),
                                           collapsedShape, collapsedStrides);
}

static Value broadcastIndexValue(OpBuilder &builder, Location loc, Value value,
                                 VectorType vectorType) {
  return vector::BroadcastOp::create(builder, loc, vectorType, value);
}

static Value materializeRank1IndexSymbol(PatternRewriter &rewriter,
                                         Location loc, Value indexValue,
                                         VectorType indexVectorType) {
  if (isa<IndexType>(indexValue.getType())) {
    return broadcastIndexValue(rewriter, loc, indexValue, indexVectorType);
  }
  auto vectorType = cast<VectorType>(indexValue.getType());
  if (vectorType == indexVectorType) {
    return indexValue;
  }
  if (vectorType.getNumElements() != 1) {
    return nullptr;
  }
  Value scalarIndex = vector::ExtractOp::create(
      rewriter, loc, indexValue, SmallVector<int64_t>(vectorType.getRank(), 0));
  return broadcastIndexValue(rewriter, loc, scalarIndex, indexVectorType);
}

static Value materializeRank1IndexExpr(PatternRewriter &rewriter, Location loc,
                                       TransferScatterOp op, AffineExpr expr,
                                       VectorType indexVectorType) {
  if (auto constantExpr = dyn_cast<AffineConstantExpr>(expr)) {
    Value constant =
        arith::ConstantIndexOp::create(rewriter, loc, constantExpr.getValue());
    return broadcastIndexValue(rewriter, loc, constant, indexVectorType);
  }
  if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
    if (dimExpr.getPosition() != 0) {
      return nullptr;
    }
    return vector::StepOp::create(rewriter, loc, indexVectorType);
  }
  if (auto symbolExpr = dyn_cast<AffineSymbolExpr>(expr)) {
    return materializeRank1IndexSymbol(
        rewriter, loc, op.getIndexVecs()[symbolExpr.getPosition()],
        indexVectorType);
  }
  auto binaryExpr = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!binaryExpr) {
    return nullptr;
  }

  switch (binaryExpr.getKind()) {
  case AffineExprKind::Add: {
    Value lhs = materializeRank1IndexExpr(rewriter, loc, op,
                                          binaryExpr.getLHS(), indexVectorType);
    Value rhs = materializeRank1IndexExpr(rewriter, loc, op,
                                          binaryExpr.getRHS(), indexVectorType);
    if (!lhs || !rhs) {
      return nullptr;
    }
    return arith::AddIOp::create(rewriter, loc, lhs, rhs);
  }
  case AffineExprKind::Mul: {
    auto lhsConstant = dyn_cast<AffineConstantExpr>(binaryExpr.getLHS());
    auto rhsConstant = dyn_cast<AffineConstantExpr>(binaryExpr.getRHS());
    AffineExpr valueExpr;
    int64_t coefficient;
    if (lhsConstant) {
      coefficient = lhsConstant.getValue();
      valueExpr = binaryExpr.getRHS();
    } else if (rhsConstant) {
      coefficient = rhsConstant.getValue();
      valueExpr = binaryExpr.getLHS();
    } else {
      return nullptr;
    }
    Value value =
        materializeRank1IndexExpr(rewriter, loc, op, valueExpr, indexVectorType);
    if (!value) {
      return nullptr;
    }
    Value constant =
        arith::ConstantIndexOp::create(rewriter, loc, coefficient);
    Value constantVector =
        broadcastIndexValue(rewriter, loc, constant, indexVectorType);
    return arith::MulIOp::create(rewriter, loc, value, constantVector);
  }
  default:
    return nullptr;
  }
}

static Value materializeRank1IndexVector(PatternRewriter &rewriter,
                                         Location loc, TransferScatterOp op,
                                         AffineExpr expr,
                                         VectorType indexVectorType) {
  return materializeRank1IndexExpr(rewriter, loc, op, expr, indexVectorType);
}

static Value materializeRank1MaskVector(PatternRewriter &rewriter,
                                        Location loc, TransferScatterOp op,
                                        VectorType maskVectorType) {
  Value mask = op.getMask();
  if (!mask) {
    auto trueAttr = DenseElementsAttr::get(maskVectorType,
                                           rewriter.getBoolAttr(true));
    return arith::ConstantOp::create(rewriter, loc, maskVectorType, trueAttr);
  }
  auto vectorType = cast<VectorType>(mask.getType());
  if (vectorType == maskVectorType) {
    return mask;
  }
  if (vectorType.getNumElements() != 1) {
    return nullptr;
  }
  Value scalarMask = vector::ExtractOp::create(
      rewriter, loc, mask, SmallVector<int64_t>(vectorType.getRank(), 0));
  return vector::BroadcastOp::create(rewriter, loc, maskVectorType, scalarMask);
}

static Value getLinearizationStride(RewriterBase &rewriter, Location loc,
                                    ArrayRef<OpFoldResult> sizes,
                                    ArrayRef<Value> strides,
                                    int64_t dim) {
  if (!strides.empty()) {
    return strides[dim];
  }
  OpFoldResult mixedStride =
      computeProduct(rewriter, loc, sizes.drop_front(dim + 1));
  return getValueOrCreateConstantIndexOp(rewriter, loc, mixedStride);
}

/// Lower a rank-1 non-contiguous transfer_scatter to vector.scatter.
struct LowerRank1TransferScatterToVectorScatter final
    : OpRewritePattern<TransferScatterOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(TransferScatterOp op,
                                PatternRewriter &rewriter) const override {
    VectorType vectorType = op.getVectorType();
    if (vectorType.getRank() != 1) {
      return rewriter.notifyMatchFailure(op, "expected rank-1 vector");
    }
    if (op.getIndexVecs().empty()) {
      return rewriter.notifyMatchFailure(op, "contiguous scatter");
    }
    bool allIndexVecsAreSingleElement =
        llvm::all_of(op.getIndexVecs(), [](Value indexVec) {
          if (isa<IndexType>(indexVec.getType())) {
            return true;
          }
          return cast<VectorType>(indexVec.getType()).getNumElements() == 1;
        });
    if (allIndexVecsAreSingleElement) {
      return rewriter.notifyMatchFailure(
          op, "single-element index vecs should fold into offsets");
    }
    if (!isa<MemRefType>(op.getBase().getType())) {
      return rewriter.notifyMatchFailure(op, "expected memref base");
    }
    // vector.scatter cannot represent sub-byte element scatters. Leave those
    // cases for unrolling/folding paths that can preserve byte-aligned stores.
    if (vectorType.getElementTypeBitWidth() < 8) {
      return rewriter.notifyMatchFailure(op,
                                         "sub-byte scatter is unsupported");
    }

    Location loc = op.getLoc();
    auto baseType = cast<MemRefType>(op.getBase().getType());
    auto indexVectorType =
        VectorType::get(vectorType.getShape(), rewriter.getIndexType());
    SmallVector<OpFoldResult> sizes =
        memref::getMixedSizes(rewriter, loc, op.getBase());
    SmallVector<Value> strides;
    Value flatBase =
        createFlatScatterBase(rewriter, loc, op.getBase(), strides);

    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value linearIndexVector =
        broadcastIndexValue(rewriter, loc, zero, indexVectorType);
    AffineMap baseMap = op.getIndexingMapsArray().front();
    // Compute element indices into the flattened destination by summing each
    // destination dim index multiplied by that dim's logical element stride.
    // Non-contiguous memrefs use the strides extracted from memref metadata.
    for (int64_t dim = 0; dim < baseType.getRank(); ++dim) {
      Value dimIndexVector = materializeRank1IndexVector(
          rewriter, loc, op, baseMap.getResult(dim), indexVectorType);
      if (!dimIndexVector) {
        return rewriter.notifyMatchFailure(op,
                                           "unsupported base indexing map");
      }
      Value offsetVector =
          broadcastIndexValue(rewriter, loc, op.getOffsets()[dim],
                              indexVectorType);
      dimIndexVector =
          arith::AddIOp::create(rewriter, loc, dimIndexVector, offsetVector);

      Value stride =
          getLinearizationStride(rewriter, loc, sizes, strides, dim);
      Value strideVector =
          broadcastIndexValue(rewriter, loc, stride, indexVectorType);
      Value stridedIndexVector =
          arith::MulIOp::create(rewriter, loc, dimIndexVector, strideVector);
      linearIndexVector = arith::AddIOp::create(
          rewriter, loc, linearIndexVector, stridedIndexVector);
    }

    auto maskVectorType =
        VectorType::get(vectorType.getShape(), rewriter.getIntegerType(1));
    Value maskVector =
        materializeRank1MaskVector(rewriter, loc, op, maskVectorType);
    if (!maskVector) {
      return rewriter.notifyMatchFailure(op, "unsupported mask map");
    }

    SmallVector<Value> operands = {flatBase, zero, linearIndexVector,
                                   maskVector, op.getVector()};
    rewriter.replaceOpWithNewOp<vector::ScatterOp>(op, TypeRange{}, operands);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// UnrollTransferGatherDim / UnrollTransferScatterDim
//===----------------------------------------------------------------------===//

/// Unrolls dim 0 of a transfer_gather, reducing vector rank by 1 each
/// application. Sub-gathers are assembled into the result via
/// insert_strided_slice. Stops at rank 1.
struct UnrollTransferGatherDim final : OpRewritePattern<TransferGatherOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(TransferGatherOp op,
                                PatternRewriter &rewriter) const override {
    VectorType vectorType = op.getVector().getType();
    int64_t rank = vectorType.getRank();
    if (rank <= 1) {
      return rewriter.notifyMatchFailure(op, "already rank <= 1");
    }

    Location loc = op.getLoc();
    int64_t dim0Size = vectorType.getShape()[0];
    SmallVector<AffineMap> indexingMaps = op.getIndexingMapsArray();
    OperandRange indexVecs = op.getIndexVecs();
    int64_t numIndexVecs = indexVecs.size();
    Value mask = op.getMask();

    SmallVector<AffineMap> newAllMaps;
    SmallVector<SmallVector<int64_t>> indexVecAxes;
    SmallVector<int64_t> maskAxes;
    SmallVector<BaseDim0Coefficient> baseDim0Coefficients;
    if (failed(computeUnrollDim0Maps(indexingMaps, numIndexVecs, !!mask,
                                     indexingMaps[0], newAllMaps, indexVecAxes,
                                     maskAxes, baseDim0Coefficients))) {
      return rewriter.notifyMatchFailure(op, "unsupported base indexing map");
    }

    SmallVector<int64_t> newShape(vectorType.getShape().drop_front());
    auto newVectorType = VectorType::get(newShape, vectorType.getElementType());

    Value acc = ub::PoisonOp::create(rewriter, loc, vectorType);

    for (int64_t i = 0; i < dim0Size; ++i) {
      FailureOr<SmallVector<Value>> newOffsets = computeNewOffsets(
          rewriter, loc, op.getOffsets(), i, baseDim0Coefficients);
      if (failed(newOffsets)) {
        return rewriter.notifyMatchFailure(op, "offset overflow while unrolling");
      }

      SmallVector<Value> newIndexVecs;
      Value newMask =
          extractSlicesForIteration(rewriter, loc, i, indexVecs, numIndexVecs,
                                    indexVecAxes, mask, maskAxes, newIndexVecs);

      auto subGather = TransferGatherOp::create(
          rewriter, loc, newVectorType, op.getBase(), *newOffsets, newIndexVecs,
          rewriter.getAffineMapArrayAttr(newAllMaps), op.getPadding(), newMask);

      SmallVector<int64_t> offsets(rank, 0);
      offsets[0] = i;
      SmallVector<int64_t> strides(newShape.size(), 1);
      acc = vector::InsertStridedSliceOp::create(
          rewriter, loc, subGather.getResult(), acc, offsets, strides);
    }

    rewriter.replaceOp(op, acc);
    return success();
  }
};

/// Unrolls dim 0 of a transfer_scatter, reducing vector rank by 1 each
/// application. For tensor semantics, sub-scatters are chained via SSA
/// results. For memref semantics, sub-scatters write in-place. Stops at
/// rank 1.
struct UnrollTransferScatterDim final : OpRewritePattern<TransferScatterOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(TransferScatterOp op,
                                PatternRewriter &rewriter) const override {
    VectorType vectorType = op.getVectorType();
    int64_t rank = vectorType.getRank();
    if (rank <= 1) {
      return rewriter.notifyMatchFailure(op, "already rank <= 1");
    }

    Location loc = op.getLoc();
    int64_t dim0Size = vectorType.getShape()[0];
    SmallVector<AffineMap> indexingMaps = op.getIndexingMapsArray();
    OperandRange indexVecs = op.getIndexVecs();
    int64_t numIndexVecs = indexVecs.size();
    Value mask = op.getMask();

    SmallVector<AffineMap> newAllMaps;
    SmallVector<SmallVector<int64_t>> indexVecAxes;
    SmallVector<int64_t> maskAxes;
    SmallVector<BaseDim0Coefficient> baseDim0Coefficients;
    if (failed(computeUnrollDim0Maps(indexingMaps, numIndexVecs, !!mask,
                                     indexingMaps[0], newAllMaps, indexVecAxes,
                                     maskAxes, baseDim0Coefficients))) {
      return rewriter.notifyMatchFailure(op, "unsupported base indexing map");
    }

    Value dest = op.getBase();

    for (int64_t i = 0; i < dim0Size; ++i) {
      FailureOr<SmallVector<Value>> newOffsets = computeNewOffsets(
          rewriter, loc, op.getOffsets(), i, baseDim0Coefficients);
      if (failed(newOffsets)) {
        return rewriter.notifyMatchFailure(op, "offset overflow while unrolling");
      }

      SmallVector<Value> newIndexVecs;
      Value newMask =
          extractSlicesForIteration(rewriter, loc, i, indexVecs, numIndexVecs,
                                    indexVecAxes, mask, maskAxes, newIndexVecs);

      Value vecSlice =
          vector::ExtractOp::create(rewriter, loc, op.getVector(), int64_t{i});

      if (op.hasTensorSemantics()) {
        auto subScatter = TransferScatterOp::create(
            rewriter, loc, dest.getType(), dest, vecSlice, *newOffsets,
            newIndexVecs, rewriter.getAffineMapArrayAttr(newAllMaps), newMask);
        dest = subScatter.getResult();
        continue;
      }
      TransferScatterOp::create(rewriter, loc, /*resultTypes=*/TypeRange{},
                                dest, vecSlice, *newOffsets, newIndexVecs,
                                rewriter.getAffineMapArrayAttr(newAllMaps),
                                newMask);
    }

    if (op.hasTensorSemantics()) {
      rewriter.replaceOp(op, dest);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

} // namespace

namespace mlir::iree_compiler::IREE::VectorExt {

void populateVectorTransferGatherScatterLoweringPatterns(
    RewritePatternSet &patterns) {
  patterns.add<UnrollTransferGatherDim, UnrollTransferScatterDim,
               LowerRank1TransferScatterToVectorScatter>(
      patterns.getContext());
}

} // namespace mlir::iree_compiler::IREE::VectorExt
