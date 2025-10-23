// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

/// Command line options used purely for development purposes. Not to be relied
/// on in any way.
static llvm::cl::opt<float> clAttentionSoftmaxMax(
    "iree-linalgext-attention-softmax-max",
    llvm::cl::desc("maximum expected value from attention softmax"),
    llvm::cl::init(1.0));

template <typename T>
static Value elementwiseValueInPlace(OpBuilder &builder, Location loc,
                                     AffineMap inputMap, AffineMap scaleMap,
                                     Value value, Value scale) {
  SmallVector<AffineMap> compressedMaps =
      compressUnusedDims(SmallVector<AffineMap>{inputMap, scaleMap});
  inputMap = compressedMaps[0];
  scaleMap = compressedMaps[1];

  SmallVector<utils::IteratorType> iteratorTypes(inputMap.getNumDims(),
                                                 utils::IteratorType::parallel);

  auto genericOp = linalg::GenericOp::create(
      builder, loc, value.getType(), scale, value,
      SmallVector<AffineMap>{scaleMap, inputMap}, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        // Convert scale to the same datatype as input.
        Value scale = convertScalarToDtype(b, loc, args[0], args[1].getType(),
                                           /*isUnsignedCast=*/false);
        Value result = T::create(b, loc, scale, args[1]);
        linalg::YieldOp::create(b, loc, result);
      });
  return genericOp.getResult(0);
}

static Value reciprocalValue(OpBuilder &b, Location loc, Value input,
                             Value output) {
  int64_t rank = cast<ShapedType>(input.getType()).getRank();
  SmallVector<AffineMap> maps = {b.getMultiDimIdentityMap(rank),
                                 b.getMultiDimIdentityMap(rank)};

  SmallVector<utils::IteratorType> iteratorTypes(rank,
                                                 utils::IteratorType::parallel);
  auto genericOp = linalg::GenericOp::create(
      b, loc, output.getType(), ValueRange{input}, output, maps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value in = convertScalarToDtype(b, loc, args[0], args[1].getType(),
                                        /*isUnsignedCast=*/false);
        // Convert scale to the same datatype as input.
        Value one = arith::ConstantOp::create(
            b, loc, b.getFloatAttr(in.getType(), 1.0));
        Value result = arith::DivFOp::create(b, loc, one, in);
        linalg::YieldOp::create(b, loc, result);
      });
  return genericOp.getResult(0);
}

static Value truncateFloat(OpBuilder &builder, Location loc, AffineMap inputMap,
                           AffineMap outputMap, Value value, Value output,
                           bool clampToFPRange) {
  SmallVector<AffineMap> compressedMaps =
      compressUnusedDims(SmallVector<AffineMap>{inputMap, outputMap});
  inputMap = compressedMaps[0];
  outputMap = compressedMaps[1];

  SmallVector<utils::IteratorType> iteratorTypes(inputMap.getNumDims(),
                                                 utils::IteratorType::parallel);
  auto genericOp = linalg::GenericOp::create(
      builder, loc, output.getType(), value, output,
      SmallVector<AffineMap>{inputMap, outputMap}, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        auto srcTy = cast<FloatType>(args[0].getType());
        auto dstTy = cast<FloatType>(args[1].getType());

        Value input = args[0];

        if (clampToFPRange) {
          double mxDbl =
              APFloat::getLargest(dstTy.getFloatSemantics(), /*Negative=*/false)
                  .convertToDouble();

          // Clamp input to dstTy(usually `fp8`) MAX value to prevent NaNs.
          // We do not clamp for `-MAX` because this function meant to only be
          // used by attention's exp2 who's value is always > 0.
          Value mx = arith::ConstantOp::create(
              builder, loc, builder.getFloatAttr(srcTy, mxDbl));
          input = arith::MinimumFOp::create(b, loc, mx, input);
        }

        // Convert scale to the same datatype as input.
        Value trunc = convertScalarToDtype(b, loc, input, dstTy,
                                           /*isUnsignedCast=*/false);
        linalg::YieldOp::create(b, loc, trunc);
      });
  return genericOp.getResult(0);
}

template <typename T>
static Value reduce(OpBuilder &builder, Location loc, AffineMap inputMap,
                    AffineMap outputMap, Value input, Value output) {
  SmallVector<AffineMap> compressedMaps =
      compressUnusedDims(SmallVector<AffineMap>{inputMap, outputMap});
  inputMap = compressedMaps[0];
  outputMap = compressedMaps[1];

  // Dims not present in outputMap are reductionDims.
  SmallVector<utils::IteratorType> iteratorTypes(
      inputMap.getNumDims(), utils::IteratorType::reduction);
  for (AffineExpr dim : outputMap.getResults()) {
    int pos = cast<AffineDimExpr>(dim).getPosition();
    iteratorTypes[pos] = utils::IteratorType::parallel;
  }

  auto genericOp = linalg::GenericOp::create(
      builder, loc, output.getType(), input, output,
      SmallVector<AffineMap>{inputMap, outputMap}, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        // Convert input to the same datatype as acc.
        Value in = convertScalarToDtype(b, loc, args[0], args[1].getType(),
                                        /*isUnsignedCast=*/false);
        Value result = T::create(b, loc, in, args[1]);
        linalg::YieldOp::create(b, loc, result);
      });

  return genericOp.getResult(0);
}

static Value computeMatmul(OpBuilder &builder, Location loc, AffineMap lhsMap,
                           AffineMap rhsMap, AffineMap accMap, Value lhs,
                           Value rhs, Value acc) {

  SmallVector<AffineMap> compressedMaps =
      compressUnusedDims(SmallVector<AffineMap>{lhsMap, rhsMap, accMap});
  lhsMap = compressedMaps[0];
  rhsMap = compressedMaps[1];
  accMap = compressedMaps[2];

  // Dims not present in accMap are reduction dims.
  SmallVector<utils::IteratorType> iteratorTypes(
      accMap.getNumDims(), utils::IteratorType::reduction);
  for (AffineExpr dim : accMap.getResults()) {
    int pos = cast<AffineDimExpr>(dim).getPosition();
    iteratorTypes[pos] = utils::IteratorType::parallel;
  }

  auto genericOp = linalg::GenericOp::create(
      builder, loc, acc.getType(), SmallVector<Value>{lhs, rhs}, acc,
      SmallVector<AffineMap>{lhsMap, rhsMap, accMap}, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        // Cast inputs to match output datatype.
        Value lhs = convertScalarToDtype(b, loc, args[0], args[2].getType(),
                                         /*isUnsignedCast=*/false);
        Value rhs = convertScalarToDtype(b, loc, args[1], args[2].getType(),
                                         /*isUnsignedCast=*/false);
        Value mul = arith::MulFOp::create(b, loc, lhs, rhs);
        Value add = arith::AddFOp::create(b, loc, mul, args[2]);
        linalg::YieldOp::create(b, loc, add);
      });

  return genericOp.getResult(0);
}

static Value applyPostQKMatmulElementwise(OpBuilder &builder, Location loc,
                                          Region &region, Value value) {
  auto rank = cast<RankedTensorType>(value.getType()).getRank();
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(rank, builder.getContext());
  SmallVector<AffineMap> indexingMaps{identityMap};
  SmallVector<utils::IteratorType> iteratorTypes(rank,
                                                 utils::IteratorType::parallel);
  auto genericOp =
      linalg::GenericOp::create(builder, loc, value.getType(), ValueRange{},
                                value, indexingMaps, iteratorTypes);
  auto &dstRegion = genericOp.getRegion();
  builder.cloneRegionBefore(region, dstRegion, dstRegion.end());
  {
    OpBuilder::InsertionGuard withinRegion(builder);
    builder.setInsertionPoint(dstRegion.back().getTerminator());
    linalg::YieldOp::create(builder, loc,
                            dstRegion.back().getTerminator()->getOperands());
    dstRegion.back().getTerminator()->erase();
  }
  return genericOp.getResult(0);
}

static Value applyMask(OpBuilder &builder, Location loc, AffineMap qkMap,
                       AffineMap maskMap, Value qk, Value mask) {

  SmallVector<AffineMap> compressedMaps =
      compressUnusedDims(SmallVector<AffineMap>{qkMap, maskMap});
  qkMap = compressedMaps[0];
  maskMap = compressedMaps[1];

  SmallVector<utils::IteratorType> iteratorTypes(qkMap.getNumDims(),
                                                 utils::IteratorType::parallel);

  Value zero = arith::ConstantOp::create(
      builder, loc,
      builder.getFloatAttr(getElementTypeOrSelf(qk.getType()), 0.0));
  Value negInf = arith::ConstantOp::create(
      builder, loc,
      builder.getFloatAttr(getElementTypeOrSelf(qk.getType()),
                           -std::numeric_limits<double>::infinity()));
  auto genericOp = linalg::GenericOp::create(
      builder, loc, qk.getType(), SmallVector<Value>{mask}, qk,
      SmallVector<AffineMap>{maskMap, qkMap}, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value qkVal = args[1];
        Value maskVal = args[0];

        // TODO: Replace bool mask condition once treated as i1 (instead of i8)
        auto maskValType = maskVal.getType();
        if (maskValType.isInteger()) {
          if (maskValType.getIntOrFloatBitWidth() != 1) {
            maskVal =
                arith::TruncIOp::create(b, loc, builder.getI1Type(), maskVal);
          }
          maskVal = arith::SelectOp::create(b, loc, maskVal, zero, negInf);
        } else {
          maskVal = convertScalarToDtype(b, loc, maskVal, qkVal.getType(),
                                         /*isUnsignedCast=*/false);
          // Scaling to compensate for base-2 softmax
          Value log2e = arith::ConstantOp::create(
              b, loc, b.getFloatAttr(qkVal.getType(), M_LOG2E));
          maskVal = arith::MulFOp::create(b, loc, maskVal, log2e);
        }
        // Finally, set the returned value to the qk element plus the mask
        // element (or 0/-infinity if bool mask). We opt for a AddFOp (instead
        // of a SelectFOp to stay consistent with the additive definition of
        // attention masking)
        Value add = arith::AddFOp::create(b, loc, qkVal, maskVal);
        linalg::YieldOp::create(b, loc, add);
      });

  return genericOp.getResult(0);
}

// Compute output = exp2(output - input)
static Value computeSubAndExp2(OpBuilder &builder, Location loc,
                               AffineMap inputMap, AffineMap outputMap,
                               Value input, Value output) {
  SmallVector<AffineMap> compressedMaps =
      compressUnusedDims(SmallVector<AffineMap>{inputMap, outputMap});
  inputMap = compressedMaps[0];
  outputMap = compressedMaps[1];

  SmallVector<utils::IteratorType> iteratorTypes(inputMap.getNumDims(),
                                                 utils::IteratorType::parallel);
  auto genericOp = linalg::GenericOp::create(
      builder, loc, output.getType(), input, output,
      SmallVector<AffineMap>{inputMap, outputMap}, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        // Convert input to the same datatype as output.
        Value in = convertScalarToDtype(b, loc, args[0], args[1].getType(),
                                        /*isUnsignedCast=*/false);
        Value diff = arith::SubFOp::create(b, loc, args[1], in);
        Value weight = math::Exp2Op::create(b, loc, diff);
        linalg::YieldOp::create(b, loc, weight);
      });
  return genericOp.getResult(0);
}

// Helper method to check if a slice will be contiguous given the offset,
// slice size. This checks that `inputSize` and `offset` are both evenly
// divisible by `tileSize`.
static bool willBeContiguousSlice(OpFoldResult inputSize, OpFoldResult tileSize,
                                  OpFoldResult offset) {
  auto constInputSize = getConstantIntValue(inputSize);
  auto constTileSize = getConstantIntValue(tileSize);
  if (!constTileSize.has_value() || !constInputSize.has_value() ||
      constInputSize.value() % constTileSize.value() != 0) {
    return false;
  }
  auto constOffset = getConstantIntValue(offset);
  if (constOffset.has_value() &&
      constOffset.value() % constTileSize.value() == 0) {
    return true;
  }
  auto affineOp = cast<Value>(offset).getDefiningOp<affine::AffineApplyOp>();
  return affineOp &&
         affineOp.getMap().getResult(0).isMultipleOf(constTileSize.value());
}

//===----------------------------------------------------------------------===//
// Attention Helpers
//===----------------------------------------------------------------------===//

Value computeQKAndElementwise(Location loc, OpBuilder &b, Value query,
                              Value key, Value scale, std::optional<Value> mask,
                              AffineMap qMap, AffineMap kMap, AffineMap sMap,
                              std::optional<AffineMap> maskMap,
                              SmallVector<OpFoldResult> iterationDomain,
                              Type sElementType, Region &elementwiseRegion,
                              DictionaryAttr qkAttrs, bool lowPrecision) {
  MLIRContext *ctx = b.getContext();
  // Since we use exp2 for attention instead of the original exp, we have to
  // multiply the scale by log2(e). We use exp2 instead of exp as most platforms
  // have better support for exp2 (we verified that we gain some speedup on
  // some GPUs).
  Value log2e = arith::ConstantOp::create(
      b, loc, b.getFloatAttr(scale.getType(), M_LOG2E));
  scale = arith::MulFOp::create(b, loc, scale, log2e);

  auto qETy = getElementTypeOrSelf(query.getType());

  AffineMap scaleMap = AffineMap::get(/*dimCount=*/qMap.getNumInputs(),
                                      /*symbolCount=*/0, ctx);

  // In the original algorithm, the scaling is done after the softmax:
  //        softmax(Q @ K.T * scale) @ V
  //
  // But, it is mathematically equivalent to do it on Q first and then multiply
  // it by K.T. This just allows us to do the scaling once, instead of each
  // iteration of the loop. This is only valid for f16 or f32 types as f8
  // is extremely limited on its dynamic range therefore this would
  // significantly affect numerics.
  if (!lowPrecision) {
    query = elementwiseValueInPlace<arith::MulFOp>(b, loc, qMap, scaleMap,
                                                   query, scale);
  }

  // ---- QK Matmul ----

  // Get sizes for S.
  SmallVector<OpFoldResult> sSizes;
  for (AffineExpr dimExpr : sMap.getResults()) {
    int dim = cast<AffineDimExpr>(dimExpr).getPosition();
    sSizes.push_back(iterationDomain[dim]);
  }

  // S = Q @ K
  // SMap = QMap @ KMap
  Value emptyS = tensor::EmptyOp::create(b, loc, sSizes, sElementType);
  Value sZero = arith::ConstantOp::create(b, loc, b.getZeroAttr(sElementType));
  Value s = linalg::FillOp::create(b, loc, sZero, emptyS).getResult(0);

  s = computeMatmul(b, loc, qMap, kMap, sMap, query, key, s);
  if (qkAttrs) {
    s.getDefiningOp()->setAttrs(qkAttrs);
  }

  s = applyPostQKMatmulElementwise(b, loc, elementwiseRegion, s);

  if (lowPrecision) {
    // For low bit-depth types we perform post Q @ K scaling. This is to avoid
    // losing numerical precision due to the low dynamic range of fp8 types when
    // pre applying the sclaing.
    AffineMap sMap = b.getMultiDimIdentityMap(sSizes.size());
    AffineMap scaleMap = AffineMap::get(/*dimCount=*/sMap.getNumInputs(),
                                        /*symbolCount=*/0, ctx);
    s = elementwiseValueInPlace<arith::MulFOp>(b, loc, sMap, scaleMap, s,
                                               scale);

    // If we need to truncate to fp8 post softmax we apply a scaling to use the
    // full fp8 range. We can do this with a offset as post `exp2` this equates
    // to multiplying by a static value. We are able to do this as `max` and
    // `sum` are scaled by the same value so the end result is the same.
    auto fpTy = cast<FloatType>(qETy);
    double mx =
        APFloat::getLargest(fpTy.getFloatSemantics(), /*Negative=*/false)
            .convertToDouble();
    Value offset = arith::ConstantOp::create(
        b, loc, b.getFloatAttr(sElementType, clAttentionSoftmaxMax / mx));
    s = elementwiseValueInPlace<arith::AddFOp>(b, loc, sMap, scaleMap, s,
                                               offset);
  }

  // S += mask
  if (mask != nullptr) {
    s = applyMask(b, loc, sMap, *maskMap, s, mask.value());
  }

  return s;
}

//===----------------------------------------------------------------------===//
// AttentionOp
//===----------------------------------------------------------------------===//

FailureOr<SmallVector<Value>> AttentionOp::decomposeOperation(OpBuilder &b) {
  Location loc = getLoc();
  Value query = getQuery();
  Value key = getKey();
  Value value = getValue();
  std::optional<Value> mask = getMask();
  DictionaryAttr config = getDecompositionConfigAttr();

  DictionaryAttr qkAttrs, pvAttrs;
  if (config) {
    qkAttrs = config.getAs<DictionaryAttr>(getQKAttrStr());
    pvAttrs = config.getAs<DictionaryAttr>(getPVAttrStr());
  }
  Value output = getOutput();

  FailureOr<AttentionOpDetail> maybeOpInfo = AttentionOpDetail::get(
      getQueryMap(), getKeyMap(), getValueMap(), getOutputMap());
  assert(succeeded(maybeOpInfo) && "Invalid attention indexing maps");
  AttentionOpDetail opInfo = maybeOpInfo.value();

  SmallVector<OpFoldResult> sizes = llvm::map_to_vector(
      getIterationDomain(b), [](Range x) { return x.size; });

  AffineMap qMap = getQueryMap();
  AffineMap kMap = getKeyMap();
  AffineMap sMap = opInfo.getSMap();

  auto qETy = getElementTypeOrSelf(query.getType());
  bool lowPrecision = qETy.getIntOrFloatBitWidth() <= 8;

  // We compute output of first matmul in f32.
  Type f32Type = b.getF32Type();

  // ---- QK Matmul + elementwise math ----
  Value s = computeQKAndElementwise(loc, b, query, key, getScale(), mask, qMap,
                                    kMap, sMap, getMaskMap(), sizes, f32Type,
                                    getRegion(), qkAttrs, lowPrecision);

  // ---- Softmax ----

  AffineMap accMap = getOutputMap();

  llvm::SmallBitVector projectedK2Dims(opInfo.getDomainRank(), false);
  for (auto dim : opInfo.getK2Dims()) {
    projectedK2Dims.set(dim);
  }

  AffineMap maxMap = projectDims(sMap, projectedK2Dims).dropZeroResults();
  AffineMap sumMap = maxMap;

  SmallVector<OpFoldResult> rowRedSize =
      applyPermutationMap<OpFoldResult>(maxMap, sizes);

  Value rowRedEmpty = tensor::EmptyOp::create(b, loc, rowRedSize, f32Type);

  Value accInit = arith::getIdentityValue(arith::AtomicRMWKind::addf,
                                          getElementTypeOrSelf(output), b, loc,
                                          /*useOnlyFiniteValue=*/true);
  Value maxInit =
      arith::getIdentityValue(arith::AtomicRMWKind::maximumf, f32Type, b, loc,
                              /*useOnlyFiniteValue=*/true);
  Value sumInit =
      arith::getIdentityValue(arith::AtomicRMWKind::addf, f32Type, b, loc);

  Value accFill =
      linalg::FillOp::create(b, loc, ValueRange{accInit}, output).getResult(0);
  Value maxFill =
      linalg::FillOp::create(b, loc, ValueRange{maxInit}, rowRedEmpty)
          .getResult(0);
  Value sumFill =
      linalg::FillOp::create(b, loc, ValueRange{sumInit}, rowRedEmpty)
          .getResult(0);

  // max = rowMax(S)
  Value max = reduce<arith::MaximumFOp>(b, loc, sMap, maxMap, s, maxFill);

  // P = exp2(S - max)
  AffineMap pMap = sMap;
  Value p = computeSubAndExp2(b, loc, maxMap, sMap, max, s);

  // sum = rowSum(P)
  Value sum = reduce<arith::AddFOp>(b, loc, pMap, sumMap, p, sumFill);

  // P = P / sum
  p = elementwiseValueInPlace<arith::DivFOp>(b, loc, pMap, sumMap, p, sum);

  // ---- Scale and truncate LHS to match RHS ----
  SmallVector<OpFoldResult> sSizes;
  for (AffineExpr dimExpr : sMap.getResults()) {
    int dim = cast<AffineDimExpr>(dimExpr).getPosition();
    sSizes.push_back(sizes[dim]);
  }

  auto pETy = getElementTypeOrSelf(p.getType());
  auto vETy = getElementTypeOrSelf(value.getType());
  if (pETy != vETy && isa<FloatType>(vETy)) {
    Value convertP = tensor::EmptyOp::create(b, loc, sSizes, vETy);
    p = truncateFloat(b, loc, pMap, pMap, p, convertP, lowPrecision);
  }

  // result = P @ V + acc
  Value result =
      computeMatmul(b, loc, pMap, getValueMap(), accMap, p, value, accFill);
  if (pvAttrs) {
    result.getDefiningOp()->setAttrs(pvAttrs);
  }

  return SmallVector<Value>{result};
}

//===----------------------------------------------------------------------===//
// OnlineAttentionOp
//===----------------------------------------------------------------------===//

FailureOr<SmallVector<Value>>
OnlineAttentionOp::decomposeOperation(OpBuilder &b) {
  Location loc = getLoc();
  Value query = getQuery();
  Value key = getKey();
  Value value = getValue();
  std::optional<Value> mask = getMask();
  Value oldAcc = getOutput();
  Value oldMax = getMax();
  Value oldSum = getSum();
  Type elementType = getElementTypeOrSelf(getOutput().getType());
  DictionaryAttr config = getDecompositionConfigAttr();

  DictionaryAttr qkAttrs, pvAttrs;
  if (config) {
    qkAttrs = config.getAs<DictionaryAttr>(getQKAttrStr());
    pvAttrs = config.getAs<DictionaryAttr>(getPVAttrStr());
  }

  FailureOr<AttentionOpDetail> maybeOpInfo = AttentionOpDetail::get(
      getQueryMap(), getKeyMap(), getValueMap(), getOutputMap());
  assert(succeeded(maybeOpInfo) && "Invalid attention indexing maps");
  AttentionOpDetail opInfo = maybeOpInfo.value();

  SmallVector<OpFoldResult> sizes = llvm::map_to_vector(
      getIterationDomain(b), [](Range x) { return x.size; });

  AffineMap qMap = getQueryMap();
  AffineMap kMap = getKeyMap();
  AffineMap sMap = opInfo.getSMap();

  auto qETy = getElementTypeOrSelf(query.getType());
  bool lowPrecision = qETy.getIntOrFloatBitWidth() <= 8;

  // ---- QK Matmul + elementwise math ----
  Value s = computeQKAndElementwise(
      loc, b, query, key, getScale(), mask, qMap, kMap, sMap, getMaskMap(),
      sizes, elementType, getRegion(), qkAttrs, lowPrecision);

  // TODO: This decomposition should be in a seperate op called
  // "online softmax".
  // ---- Online Softmax ----

  // newMax = max(oldMax, rowMax(S))
  AffineMap maxMap = getMaxMap();
  Value newMax = reduce<arith::MaximumFOp>(b, loc, sMap, maxMap, s, oldMax);

  // norm = exp2(oldMax - newMax)
  // normMap = maxMap
  AffineMap normMap = getMaxMap();
  Value norm = computeSubAndExp2(b, loc, maxMap, normMap, newMax, oldMax);

  // normSum = norm * oldSum
  AffineMap sumMap = getSumMap();
  Value normSum = elementwiseValueInPlace<arith::MulFOp>(b, loc, sumMap,
                                                         normMap, oldSum, norm);

  // P = exp2(S - newMax)
  // PMap = SMap
  AffineMap pMap = sMap;
  Value p = computeSubAndExp2(b, loc, maxMap, sMap, newMax, s);

  // newSum = normSum + rowSum(P)
  Value newSum = reduce<arith::AddFOp>(b, loc, pMap, sumMap, p, normSum);

  // newAcc = norm * oldAcc
  AffineMap accMap = getOutputMap();

  // ---- Scale and truncate LHS to match RHS ----
  SmallVector<OpFoldResult> sSizes;
  for (AffineExpr dimExpr : sMap.getResults()) {
    int dim = cast<AffineDimExpr>(dimExpr).getPosition();
    sSizes.push_back(sizes[dim]);
  }

  auto pETy = getElementTypeOrSelf(p.getType());
  auto vETy = getElementTypeOrSelf(value.getType());
  if (pETy != vETy && isa<FloatType>(vETy)) {
    Value convertP = tensor::EmptyOp::create(b, loc, sSizes, vETy);
    p = truncateFloat(b, loc, pMap, pMap, p, convertP, lowPrecision);
  }

  Value newAcc = elementwiseValueInPlace<arith::MulFOp>(b, loc, accMap, normMap,
                                                        oldAcc, norm);

  // ---- Matmul 2 ----

  // newAcc = P @ V + newAcc
  newAcc = computeMatmul(b, loc, pMap, getValueMap(), accMap, p, value, newAcc);
  if (pvAttrs) {
    newAcc.getDefiningOp()->setDiscardableAttrs(pvAttrs);
  }

  return SmallVector<Value>{newAcc, newMax, newSum};
}

//===----------------------------------------------------------------------===//
// Im2colOp
//===----------------------------------------------------------------------===//

static SmallVector<int64_t>
chooseDimsToVectorize(OpBuilder &b, Location loc, Im2colOp im2colOp,
                      SmallVector<Range> iterationDomain,
                      SmallVector<OpFoldResult> inputSizes,
                      OpFoldResult kOffset) {
  std::optional<ArrayRef<int64_t>> vectorizationHint =
      im2colOp.getVectorizationHint();
  if (vectorizationHint.has_value()) {
    return SmallVector<int64_t>(vectorizationHint.value());
  }
  int64_t innerInputDim = im2colOp.getInputRank() - 1;
  SmallVector<SmallVector<int64_t>> vectorizationMap =
      im2colOp.getInputToOutputDimVectorizationMap();
  SmallVector<int64_t> vectorizableOutputDims = vectorizationMap[innerInputDim];
  if (vectorizableOutputDims.empty()) {
    return {};
  }
  SetVector<int64_t> kDimSet(llvm::from_range, im2colOp.getKOutputDims());
  // There may be multiple output dims that we can vectorize, so prioritize the
  // innermost dims first.
  llvm::sort(vectorizableOutputDims);
  // Check each dim in order from innermost to outermost, and return the first
  // one that is vectorizable.
  while (!vectorizableOutputDims.empty()) {
    int64_t outputDimToVectorize = vectorizableOutputDims.pop_back_val();
    // If a K dim is being vectorized, then it is contiguous along either the
    // input channel dimension, or the filter kernel window. If it is contiguous
    // along the kernel window, then the actual inner slice size is equal to the
    // size of the corresponding kernel window dimension. Otherwise, the inner
    // slice size is just the size of the input tensor's inner dimension.
    OpFoldResult innerSliceSize = inputSizes[innerInputDim];
    if (kDimSet.contains(outputDimToVectorize)) {
      for (auto [kernelSize, mPos] :
           llvm::zip_equal(im2colOp.getMixedKernelSize(), im2colOp.getMPos())) {
        if (mPos == innerInputDim) {
          innerSliceSize = kernelSize;
        }
      }
    }

    // If the input slice is contiguous along the innermost dimension, then it
    // is vectorizable. If it is not, then move on to the next innermost dim.
    SetVector<int64_t> mDimSet(llvm::from_range, im2colOp.getMOutputDims());
    OpFoldResult offset = b.getIndexAttr(0);
    if (kDimSet.contains(outputDimToVectorize)) {
      offset = kOffset;
    } else if (mDimSet.contains(outputDimToVectorize)) {
      // TODO(Max191): Support vectorization along the M dimension.
      continue;
    }
    OpFoldResult outputDimSize = iterationDomain[outputDimToVectorize].size;
    if (!willBeContiguousSlice(innerSliceSize, outputDimSize, offset)) {
      continue;
    }
    return {outputDimToVectorize};
  }
  return {};
}

/// Given offsets and sizes into the input and output, generate the
/// decomposition slice for an Im2colOp, expecting that the input and output
/// dimension order matches, and return the decomposed result. At most one
/// dimension of the im2col op's output can have an upper bound greater than 1.
///
/// The im2col decomposition will be done by expanding the input and output, and
/// then generating a single tensor.extract_slice op. The reassociation for the
/// expansion is determined by the `outputToInputDimVectorizationMap` mapping,
/// and may be different for the input vs the output. The expansion will only
/// add additional unit dimensions in order to create a 1 to 1 mapping between
/// input and output dimensions. The slice will then be collapsed back to the
/// original output shape.
///
/// This is possible because of the ordering and bound constraints imposed by
/// this function. If dimensions are out of order, then it is not possible to
/// decompose the im2col as a simple extract_slice. If more than a single output
/// dimension has an upper bound greater than 1, then it might not valid to add
/// only unit dimensions in the expansion.
///
/// Example:
///
/// Consider the decomposition for an NCHW layout im2col op (some metadata is
/// ommitted for brevity, and shapes are shown with symbols to help show how
/// dims map to each other):
///
///   %im2col = iree_linalg_ext.im2col
///       strides = [1, 1] dilations = [1, 1] kernel_size = [P, Q]
///       batch_pos = [0] m_pos = [2, 3] k_pos = [1]
///       input_k_perm = [0, 1, 2] output_perm = [0, 1, 2, 3]
///       ins(%in : tensor<NxCxHxWxf32>)
///       outs(%out : tensor<BxM0xM1xKxf32>) -> tensor<BxM0xM1xKxf32>
///
/// For this example, let's assume that we vectorized along `K`, so the inner
/// dim of the output is bounded by > 1, and the other output dims are bounded
/// by 1. The mapping of output to input dimensions would look like the
/// following:
///   [ B, M0, M1, K ]
///     |    \   \ |
///     |     \   \|
///   [ N, C,  H,  W ]
/// There is not a 1 to 1 mapping from output to input dimensions, so the
/// operands will be expanded to create a 1 to 1 mapping, as follows:
///   [ B, M0, M1, K ]  ->  [ B, 1, M0, M1, K ]
///     |    \   \ |          |  |   |  |   |
///     |     \   \|          |  |   |  |   |
///   [ N, C,  H,  W ]  ->  [ N, C,  H, 1,  W ]
/// The K dimension is the vectorized dim, so it must map to the innermost input
/// dimension after the expansion. The other dimensions are bounded by 1, so the
/// resulting slice will be valid even if some of the output dimensions now map
/// to unit dimensions. The resulting IR will look like:
///
///   %expanded_in = tensor.expand_shape %in [[0], [1], [2, 3], [4]]
///       : tensor<NxCxHxWxf32> into tensor<NxCxHx1xWxf32>
///   %slice = tensor.extract_slice %expanded_in
///       [%oN, %oC, %oH, 0, %oW][%B, 1, %M0, %M1, %K][1, 1, 1, 1, 1]
///       : to tensor<NxCxHx1xWxf32> into tensor<Bx1xM0xM1xKxf32>
///   %collapsed_out = tensor.collapse_shape %slice [[0], [1, 2], [3], [4]]
///       : tensor<Bx1xM0xM1xKxf32> into tensor<BxM0xM1xKxf32>
static Value generateInOrderIm2colSlice(
    OpBuilder &b, Im2colOp im2colOp, ArrayRef<OpFoldResult> inputOffsets,
    ArrayRef<OpFoldResult> outputSizes,
    ArrayRef<std::optional<int64_t>> outputToInputDimVectorizationMap) {
  // TODO: Compute the reassociations for the input and output based on
  // `outputToInputDimVectorizationMap`.
  SmallVector<ReassociationIndices> inputReassociations, outputReassociations;
  int64_t prevInputDim = 0, expandedInputDim = 0, expandedOutputDim = 0;
  for (auto [outputDim, inputDim] :
       llvm::enumerate(outputToInputDimVectorizationMap)) {
    // Case 1: Output dim doesn't map to any input dim. Just add the output dim
    // in its own reassociation group, since we don't have any information about
    // how many unit dims to add if we have no mapping.
    if (!inputDim.has_value()) {
      outputReassociations.push_back({expandedOutputDim++});
      continue;
    }
    // Case 2: Output dim maps to the same input dim as the previous output dim.
    // This means we need to expand out a new input dim to avoid this collision.
    if (*inputDim == prevInputDim) {
      // TODO: May need to account for which dim is vectorized if the final
      // dims map to the same input dim.
      if (inputReassociations.empty()) {
        inputReassociations.push_back({expandedInputDim++});
      } else {
        inputReassociations.back().push_back(expandedInputDim++);
      }
      outputReassociations.push_back({expandedOutputDim++});
      prevInputDim = *inputDim;
      continue;
    }
    // Case 3: Output dim maps to the immediate next input dim. This obeys the
    // 1 to 1 mapping, so just add both dims to their own groups.
    if (*inputDim == prevInputDim + 1) {
      inputReassociations.back().push_back(expandedInputDim++);
      outputReassociations.push_back({expandedOutputDim++});
      prevInputDim = *inputDim;
      continue;
    }
    // Case 4: Output dim maps to a new input dim, skipping one or more input
    // dims. This means we need to expand out new output dims to map to the
    // skipped input dims.
    assert(*inputDim > prevInputDim + 1 &&
           "Expected input and output dims to be in order");
    int64_t numSkippedInputDims = *inputDim - prevInputDim - 1;
    for (int64_t i = 0; i < numSkippedInputDims; ++i) {
      inputReassociations.push_back({expandedInputDim++});
      outputReassociations.back().push_back(expandedOutputDim++);
    }
    inputReassociations.push_back({expandedInputDim++});
    outputReassociations.push_back({expandedOutputDim++});
    prevInputDim = *inputDim;
  }

  int64_t sliceRank = expandedInputDim;
  SmallVector<int64_t> expandedInputShape(sliceRank, 1);
  SmallVector<OpFoldResult> expandedInputOffsets(sliceRank, b.getIndexAttr(0));
  SmallVector<OpFoldResult> expandedInputSizes(sliceRank, b.getIndexAttr(1));
  Location loc = im2colOp.getLoc();
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfterValue(im2colOp.getInput());
  SmallVector<OpFoldResult> inputSizes =
      tensor::getMixedSizes(b, loc, im2colOp.getInput());
  auto inputType = cast<RankedTensorType>(im2colOp.getInputType());
  for (auto [idx, group] : llvm::enumerate(inputReassociations)) {
    // TODO: Will also need to account for which dim is vectorized here if the
    // final dims map to the same input dim.
    expandedInputShape[group.back()] = inputType.getDimSize(idx);
    expandedInputSizes[group.back()] = inputSizes[idx];
    expandedInputOffsets[group.back()] = inputOffsets[idx];
  }
  SmallVector<OpFoldResult> expandedOutputSizes(sliceRank, b.getIndexAttr(1));
  for (auto [idx, group] : llvm::enumerate(outputReassociations)) {
    expandedOutputSizes[idx] = outputSizes[idx];
  }

  RankedTensorType expandedInputType = inputType.clone(expandedInputShape);
  Value expandedInput = tensor::ExpandShapeOp::create(
      b, loc, expandedInputType, im2colOp.getInput(), inputReassociations,
      expandedInputSizes);

  ArrayRef<OpFoldResult> sliceOffsets = expandedInputOffsets;
  ArrayRef<OpFoldResult> sliceSizes = expandedOutputSizes;
  SmallVector<OpFoldResult> sliceStrides(sliceRank, b.getIndexAttr(1));
  Value slice = tensor::ExtractSliceOp::create(
      b, loc, expandedInput, sliceOffsets, sliceSizes, sliceStrides);

  return tensor::CollapseShapeOp::create(b, loc, slice, outputReassociations);
}

static Value generateOutOfOrderIm2colSlice(
    OpBuilder &b, Im2colOp im2colOp, ArrayRef<OpFoldResult> inputOffsets,
    ArrayRef<OpFoldResult> inputSizes, ArrayRef<OpFoldResult> outputOffsets,
    ArrayRef<OpFoldResult> outputSizes,
    ArrayRef<std::optional<int64_t>> outputToInputDimVectorizationMap) {
  assert(false && "Not implemented");
  return Value();
}

static Value generateIm2colSlice(
    OpBuilder &b, Im2colOp im2colOp, ArrayRef<OpFoldResult> inputOffsets,
    ArrayRef<OpFoldResult> inputSizes, ArrayRef<OpFoldResult> outputOffsets,
    ArrayRef<OpFoldResult> outputSizes,
    ArrayRef<std::optional<int64_t>> outputToInputDimVectorizationMap) {
  bool inputOrderMatchesOutputOrder = true;
  int64_t prevInputDim = 0;
  for (std::optional<int64_t> inputDim : outputToInputDimVectorizationMap) {
    if (!inputDim.has_value()) {
      continue;
    }
    if (*inputDim < prevInputDim) {
      inputOrderMatchesOutputOrder = false;
      break;
    }
    prevInputDim = *inputDim;
  }

  if (inputOrderMatchesOutputOrder) {
    return generateInOrderIm2colSlice(b, im2colOp, inputOffsets, outputSizes,
                                      outputToInputDimVectorizationMap);
  }
  return generateOutOfOrderIm2colSlice(b, im2colOp, inputOffsets, inputSizes,
                                       outputOffsets, outputSizes,
                                       outputToInputDimVectorizationMap);
}

/// Decomposition implementation for iree_linalg_ext.im2col op.
/// The im2col op is decomposed into serial loops of `insert->extract->copy`.
/// The decomposition supports leaving either the `batch` or `K` dimension
/// untiled when the corresponding slice in the input tensor is contiguous.
/// If the entire `K` dimension maps to a contiguous slice, the loop over `K`
/// is left untiled to enable more efficient data transfer. Likewise, if the
/// `batch` dimension is contiguous, it is left untiled instead. All other
/// dimensions, including any non-contiguous `batch` or `K`, are tiled to 1.
/// TODO(Max191): Fallback to larger tile sizes instead of immediately tiling K
///               dimension to 1 when non-contiguous.
///
/// The simple decomposition (with K tiled to 1) will look like:
/// ```
///   %im2col = iree_linalg_ext.im2col
///       strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
///       m_offset = [%m_off] * [1] k_offset = [%k_off] * [1]
///       batch_pos = [0] m_pos = [1, 2] k_pos = [3]
///       input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
///       ins(%in : tensor<2x34x34x640xf32>)
///       outs(%out : tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
/// ```
/// Decomposes to:
/// ```
/// scf.for %B = %c0 to %c2 step %c1
///   scf.for %M = %c0 to %c4 step %c1
///     scf.for %K = %c0 to %c8 step %c1
///       %slice = tensor.extract_slice %in[%B, %h, %w, %k] ... to tensor<1xf32>
///       %copy = linalg.copy ins(%slice) outs(%out)
///       %insert = tensor.insert_slice %copy into %loop_arg
/// ```
/// Where the offsets are computed as:
///   `%h` = `(%m_off + %M) / 32 + ((%k_off + %K) / 640) / 3`
///   `%w` = `(%m_off + %M) mod 32 + ((%k_off + %K) / 640) mod 3`
///   `%k` = `(%k_off + %K) mod 640`
///
FailureOr<SmallVector<Value>> Im2colOp::decomposeOperation(OpBuilder &b) {
  // This is part of the im2col verifier, but check here in case this changes.
  assert(getConstantIntValue(getMixedMStrides().back()).value() == 1 &&
         getConstantIntValue(getMixedKStrides().back()).value() == 1 &&
         "Expected inner m_offset and k_offset to be 1");

  // Get the linearized mOffset and kOffset.
  Location loc = getLoc();
  auto linearizeIndex = [&](ArrayRef<OpFoldResult> inds,
                            ArrayRef<OpFoldResult> basis) {
    MLIRContext *ctx = b.getContext();
    SmallVector<AffineExpr> dims(inds.size()), symbols(basis.size());
    bindDimsList<AffineExpr>(ctx, dims);
    bindSymbolsList<AffineExpr>(ctx, symbols);
    AffineExpr linearExpr = mlir::linearize(ctx, dims, symbols);
    SmallVector<OpFoldResult> mapOperands(inds);
    mapOperands.append(basis.begin(), basis.end());
    auto linearMap = AffineMap::get(
        /*dimCount=*/inds.size(), /*symbolCount=*/basis.size(), linearExpr);
    OpFoldResult linearIdx =
        affine::makeComposedFoldedAffineApply(b, loc, linearMap, mapOperands);
    return linearIdx;
  };
  OpFoldResult mOffset = linearizeIndex(getMixedMOffset(), getMixedMStrides());
  OpFoldResult kOffset = linearizeIndex(getMixedKOffset(), getMixedKStrides());

  // Step 1: Tile the im2col op to loops with contiguous slices in the
  // innermost loop.
  //
  // If the innermost dim of the input tensor contains a full contiguous slice,
  // then don't tile the corresponding loop of the im2col op and maintain a
  // larger contiguous slice. Note that if the im2col input tensor has the batch
  // dim at last, im2col output tensor has an implicit transpose to move the
  // batch dim in front, and tiling should be along the batch dim.
  SmallVector<Range> iterationDomain(getIterationDomain(b));
  SmallVector<OpFoldResult> inputSizes =
      tensor::getMixedSizes(b, loc, getInput());
  SmallVector<int64_t> outputDimsToVectorize = chooseDimsToVectorize(
      b, loc, *this, iterationDomain, inputSizes, kOffset);

  SmallVector<OpFoldResult> inputTileSizes(getInputRank(), b.getIndexAttr(1));
  SmallVector<OpFoldResult> outputTileSizes(getOutputRank(), b.getIndexAttr(1));
  SmallVector<std::optional<int64_t>> outputToInputDimVectorizationMap(
      getOutputRank(), {});
  SmallVector<SmallVector<int64_t>> inputToOutputDimVectorizationMap =
      getInputToOutputDimVectorizationMap();
  for (auto [inputDim, outputDims] :
       llvm::enumerate(inputToOutputDimVectorizationMap)) {
    for (int64_t outputDim : outputDims) {
      // The `inputToOutputDimVectorizationMap` already contains a mapping where
      // corresponding dims are strided together, meaning an increment in the
      // output dim will cause an increment in the input dim. There are no dims
      // in the output that can have this relationship with more than one input
      // dimension, so we expect only one input dim per output dim.
      assert(!outputToInputDimVectorizationMap[outputDim].has_value() &&
             "Expected only one vectorizable input dim per output dim");
      outputToInputDimVectorizationMap[outputDim] = inputDim;
    }
  }
  // Erase vectorizable dims from the iteration domain in reverse order. We do
  // this partly to preserve the relative order of the remaining dims, but also
  // because we want to prioritize vectorizing the innermost output dims first.
  // There could be multiple output dims mapping to the same input dim, since
  // convolution input images have convolved dimensions (indexing map results
  // that take the form of `d0 + d1`).
  SetVector<int64_t> vectorizedOutputDims, vectorizedInputDims;
  std::sort(outputDimsToVectorize.begin(), outputDimsToVectorize.end());
  for (int64_t outputDimToVectorize : llvm::reverse(outputDimsToVectorize)) {
    OpFoldResult dimSize = iterationDomain[outputDimToVectorize].size;
    std::optional<int64_t> inputDimToVectorize =
        outputToInputDimVectorizationMap[outputDimToVectorize];
    if (!inputDimToVectorize.has_value()) {
      continue;
    }
    // There may be multiple output dims mapping to the same input dim. For
    // example, in 2D convolutions with a CHW layout, the K output dim can
    // vectorize with the W input dim, but the innermost M output dim can also
    // vectorize with the M input dim. In such cases, we do not expect both of
    // the output dims to be vectorized at the same time unless one or both of
    // them has an upper bound dim size of 1. If this is the case, then we might
    // vectorize both dims, and the vector size along the corresponding input
    // dim can be computed as the product of the 2 output dim sizes. This works
    // because one of the output dims must be either 0 or 1, so we are loading
    // either 0 elements, or the full vector size of the other output dim.
    OpFoldResult inputDimSize =
        vectorizedInputDims.contains(*inputDimToVectorize)
            ? LinalgExt::mulOfrs(b, loc, inputTileSizes[*inputDimToVectorize],
                                 dimSize)
            : dimSize;
    inputTileSizes[*inputDimToVectorize] = inputDimSize;
    outputTileSizes[outputDimToVectorize] = dimSize;
    iterationDomain.erase(iterationDomain.begin() + outputDimToVectorize);
    vectorizedOutputDims.insert(outputDimToVectorize);
    vectorizedInputDims.insert(*inputDimToVectorize);
  }

  // Build loop nest.
  SmallVector<Value> lbs, ubs, steps;
  for (auto range : iterationDomain) {
    lbs.push_back(getValueOrCreateConstantIndexOp(b, loc, range.offset));
    ubs.push_back(getValueOrCreateConstantIndexOp(b, loc, range.size));
    steps.push_back(getValueOrCreateConstantIndexOp(b, loc, range.stride));
  }
  scf::LoopNest loopNest = scf::buildLoopNest(
      b, loc, lbs, ubs, steps, getOutput(),
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange outputIvs,
          ValueRange iterArgs) -> scf::ValueVector { return iterArgs; });
  SmallVector<Value> ivs;
  for (scf::ForOp loop : loopNest.loops) {
    ivs.push_back(loop.getInductionVar());
  }
  if (!vectorizedOutputDims.empty()) {
    Value zero = arith::ConstantIndexOp::create(b, loc, 0);
    // `vectorizedOutputDims` was constructed in reverse order, but we want to
    // iterate in forward order to insert at the correct positions.
    for (int64_t outputDim : llvm::reverse(vectorizedOutputDims)) {
      ivs.insert(ivs.begin() + outputDim, zero);
    }
  }

  // Step 2: Compute indices into the input tensor for extract_slice.
  OpBuilder::InsertionGuard guard(b);
  if (!loopNest.loops.empty()) {
    b.setInsertionPoint(loopNest.loops.front());
  }
  SetVector<int64_t> mPosSet(getMPos().begin(), getMPos().end());

  // Compute the basis for the iteration space of the convolution window
  // (i.e., the H and W dims of the convolution output).
  SmallVector<Value> mBasis;
  ArrayRef<int64_t> strides = getStrides();
  ArrayRef<int64_t> dilations = getDilations();
  SmallVector<OpFoldResult> kernelSize = getMixedKernelSize();
  for (auto [idx, pos] : llvm::enumerate(getMPos())) {
    AffineExpr x, k;
    bindDims(getContext(), x, k);
    AffineExpr mapExpr =
        (x - 1 - (k - 1) * dilations[idx]).floorDiv(strides[idx]) + 1;
    OpFoldResult size = affine::makeComposedFoldedAffineApply(
        b, loc, AffineMap::get(2, 0, {mapExpr}, getContext()),
        {inputSizes[pos], kernelSize[idx]});
    mBasis.push_back(getValueOrCreateConstantIndexOp(b, loc, size));
  }

  // Delinearize the k_offset into an offset into the convolution window and
  // any reduced channels. For an NHWC conv2d, the basis for delinearization
  // would be [P, Q, C] for a PxQ kernel with C channels.
  if (!loopNest.loops.empty()) {
    b.setInsertionPointToStart(loopNest.loops.back().getBody());
  }

  SmallVector<OpFoldResult> kBasis;
  SmallVector<int64_t> mKernelIdx(getInputRank(), -1);
  for (auto [idx, mPos] : enumerate(getMPos())) {
    mKernelIdx[mPos] = idx;
  }
  SetVector<int64_t> batchPosSet(getBatchPos().begin(), getBatchPos().end());
  for (auto [idx, size] : enumerate(inputSizes)) {
    if (batchPosSet.contains(idx))
      continue;
    if (mPosSet.contains(idx)) {
      kBasis.push_back(kernelSize[mKernelIdx[idx]]);
      continue;
    }
    kBasis.push_back(size);
  }

  // Transpose the order of (P, Q, C) according to `inputKPerm` encoded in
  // im2col metadata.
  ArrayRef<int64_t> inputKPerm = getInputKPerm();
  applyPermutationToVector(kBasis, inputKPerm);

  OpFoldResult kIndex = kOffset;
  for (auto [i, ivIdx, stride] :
       llvm::enumerate(getKOutputDims(), getMixedKStrides())) {
    if (isConstantIntValue(ivs[ivIdx], 0)) {
      continue;
    }
    OpFoldResult ivOffset = mulOfrs(b, loc, stride, ivs[ivIdx]);
    kIndex = addOfrs(b, loc, kIndex, ivOffset);
  }
  ValueRange delinKOffset =
      affine::AffineDelinearizeIndexOp::create(
          b, loc, getValueOrCreateConstantIndexOp(b, loc, kIndex), kBasis,
          /*hasOuterBound=*/true)
          .getResults();
  // Split the delinearized offsets into the window offsets (for M offsets)
  // and the K offsets for the input tensor based on the layout.
  SmallVector<Value> windowOffset, inputKOffset;
  int delinKIdx = 0;
  SmallVector<int64_t> invInputKPerm = invertPermutationVector(inputKPerm);
  for (int i = 0; i < getInputRank(); ++i) {
    if (batchPosSet.contains(i))
      continue;
    if (mPosSet.contains(i)) {
      windowOffset.push_back(delinKOffset[invInputKPerm[delinKIdx++]]);
      continue;
    }
    inputKOffset.push_back(delinKOffset[invInputKPerm[delinKIdx++]]);
  }

  // Compute offsets for extract. The linearized im2col result M offset is
  // computed as the m_offset * m_strides inner product plus the linearized
  // offset from the tiled m loops. The M offsets into the im2col input are then
  // computed as the delinearized im2col result M offset (in the convolution
  // result iteration space), plus the convolutional window offsets computed
  // above.
  SmallVector<OpFoldResult> mIvs, mOutStrides(getMixedMStrides());
  for (auto [idx, dim] : llvm::enumerate(getMOutputDims())) {
    mIvs.push_back(ivs[dim]);
  }
  OpFoldResult linearMIv = linearizeIndex(mIvs, mOutStrides);
  OpFoldResult linearMOffset = addOfrs(b, loc, linearMIv, mOffset);
  // Delinearize the m_offset * m_strides into the convolution output space.
  // `mBasis` contains the basis for the iteration space of result of the
  // convolution op (i.e., basis for result H and W dims).
  ValueRange delinMOffset =
      affine::AffineDelinearizeIndexOp::create(
          b, loc, getValueOrCreateConstantIndexOp(b, loc, linearMOffset),
          mBasis,
          /*hasOuterBound=*/true)
          .getResults();

  // Compute the final offsets into the input tensor.
  OpFoldResult zero = b.getIndexAttr(0);
  OpFoldResult one = b.getIndexAttr(1);
  SmallVector<OpFoldResult> inputSliceOffsets(getInputRank(), zero);
  SmallVector<OpFoldResult> inputSliceStrides(getInputRank(), one);
  SmallVector<OpFoldResult> inputSliceSizes = inputTileSizes;
  // Add the offset into the convolution window, and account for strides and
  // dilations.
  AffineExpr mOff, wOff;
  bindDims(b.getContext(), mOff, wOff);
  for (auto [idx, mPos] : llvm::enumerate(getMPos())) {
    auto map =
        AffineMap::get(2, 0, {mOff * strides[idx] + wOff * dilations[idx]});
    OpFoldResult offset = affine::makeComposedFoldedAffineApply(
        b, loc, map, {delinMOffset[idx], windowOffset[idx]});
    inputSliceOffsets[mPos] = offset;
  }

  // Set the batch and K offsets for the input tensor.
  const int64_t kPos = getKPos().front();
  inputSliceOffsets[kPos] = inputKOffset.front();
  int ivIdx = 0;
  SmallVector<int64_t> inverseOutputPerm =
      invertPermutationVector(getOutputPerm());
  for (auto bPos : getBatchPos()) {
    inputSliceOffsets[bPos] = ivs[inverseOutputPerm[ivIdx++]];
  }

  // Step 3. Decompose the im2col op.
  SmallVector<OpFoldResult> outputSliceOffsets(ivs.begin(), ivs.end());
  SmallVector<OpFoldResult> outputSliceSizes = outputTileSizes;
  Value sliceResult = generateIm2colSlice(
      b, *this, inputSliceOffsets, inputSliceSizes, outputSliceOffsets,
      outputSliceSizes, outputToInputDimVectorizationMap);
  if (loopNest.loops.empty()) {
    return SmallVector<Value>({sliceResult});
  }
  auto yieldOp =
      cast<scf::YieldOp>(loopNest.loops.back().getBody()->getTerminator());
  b.setInsertionPoint(yieldOp);
  SmallVector<OpFoldResult> outputSliceStrides(getOutputRank(), one);
  Value dest = loopNest.loops.back().getRegionIterArg(0);
  Value insert = tensor::InsertSliceOp::create(
      b, loc, sliceResult, dest, outputSliceOffsets, outputSliceSizes,
      outputSliceStrides);
  yieldOp->getOpOperands().front().assign(insert);
  return SmallVector<Value>({loopNest.results[0]});
}

//===----------------------------------------------------------------------===//
// CustomOp
//===----------------------------------------------------------------------===//

FailureOr<SmallVector<Value>> CustomOp::decomposeOperation(OpBuilder &builder) {
  CustomOp customOp = *this;

  IRRewriter rewriter(builder);
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(customOp);
  // Inline the body of the operation using the ins/outs as the arguments.
  SmallVector<Value> argReplacements;
  Location loc = getLoc();
  Block *body = customOp.getBody();
  for (auto [operand, argument] :
       llvm::zip_equal(customOp->getOperands(), body->getArguments())) {
    if (operand.getType() != argument.getType()) {
      assert(isa<RankedTensorType>(operand.getType()) &&
             isa<RankedTensorType>(argument.getType()) &&
             "expected operand and arguments to be `RankedTensorType`");
      Value cast =
          tensor::CastOp::create(builder, loc, argument.getType(), operand);
      argReplacements.push_back(cast);
    } else {
      argReplacements.push_back(operand);
    }
  }

  Block *oldBlock = customOp->getBlock();
  Block *newBlock = rewriter.splitBlock(oldBlock, Block::iterator(customOp));
  rewriter.mergeBlocks(body, oldBlock, argReplacements);

  // Get the operands of the `iree_linalg_ext.yield` which is the terminator of
  // `oldBlock` right now.
  auto yieldOp = cast<IREE::LinalgExt::YieldOp>(oldBlock->getTerminator());
  rewriter.setInsertionPointToEnd(oldBlock);
  SmallVector<Value> customOpReplacements;
  for (auto [yieldedVal, result] :
       llvm::zip_equal(yieldOp->getOperands(), customOp->getResults())) {
    if (yieldedVal.getType() != result.getType()) {
      assert(isa<RankedTensorType>(yieldedVal.getType()) &&
             isa<RankedTensorType>(result.getType()) &&
             "expected yielded value and result to be `RankedTensorType`");
      Value cast =
          tensor::CastOp::create(builder, loc, result.getType(), yieldedVal);
      customOpReplacements.push_back(cast);
    } else {
      customOpReplacements.push_back(yieldedVal);
    }
  }
  // Erase the yield op.
  rewriter.eraseOp(yieldOp);

  // Merge the block back.
  rewriter.mergeBlocks(newBlock, oldBlock);

  return customOpReplacements;
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
