// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/DataLayoutUtils.h"
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "iree-global-opt-propagate-data-layout"

static const char kDataLayoutNodeTypeAttr[] = "__node_type__";
static const char kFoldablePackUnPack[] = "__foldable_pack_unpack__";

namespace mlir::iree_compiler::GlobalOptimization {
using iree_compiler::IREE::Util::GlobalOp;

//===----------------------------------------------------------------------===//
// DataLayoutTransformation
//===----------------------------------------------------------------------===//

template <typename T>
static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const llvm::SmallVectorImpl<T> &vector) {
  os << "[ ";
  for (T element : vector) {
    os << element << " ";
  }
  os << "]";

  return os;
}

template <typename T>
static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const llvm::ArrayRef<T> &vector) {
  os << "[ ";
  for (T element : vector) {
    os << element << " ";
  }
  os << "]";

  return os;
}

static llvm::raw_ostream &
operator<<(llvm::raw_ostream &os, const DataLayoutTransformation &transform) {
  os << "originalType: " << transform.getOriginalType() << "\n";
  os << "transformedType: " << transform.getTransformedType() << "\n";
  os << "innerDimsPos: " << transform.getInnerDimsPos() << "\n";
  os << "innerTileSizes: " << transform.getInnerTileSizes() << "\n";
  os << "outerDimsPerm: " << transform.getOuterDimsPerm() << "\n";
  os << "correspondingTransformedIndices: "
     << transform.getCorrespondingTransformedIndices() << "\n";
  return os;
}

const bool DataLayoutTransformation::hasValidTransform() {
  return llvm::detail::isPresent(originalType) &&
         llvm::detail::isPresent(transformedType);
}

bool DataLayoutTransformation::transformLayout(Value currentValue,
                                               Value newValue) {
  auto curDefiningOp = currentValue.getDefiningOp();
  auto newDefiningOp = newValue.getDefiningOp();
  auto isInitOperand = [](Operation *op, Value val) -> bool {
    if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op)) {
      return llvm::any_of(dpsOp.getDpsInits(),
                          [&](Value v) { return v == val; });
    }
    return false;
  };
  auto valueConsumedByOp = [](Operation *op, Value val) -> bool {
    return op &&
           llvm::any_of(op->getOperands(), [&](Value v) { return v == val; });
  };
  // // If values are used by the same op, then they have the same layout.
  // if (llvm::any_of(currentValue.getUsers(), [&](Operation *op) {
  //       // If either value is a DPS init, then they will not have the same
  //       // layout. For now, we ignore this case here and let it get handled
  //       // later by a different transformation. This couples the
  //       // `transformLayout` function with the flow of the
  //       // `GlobalDataLayoutAnalysis` DFX analysis.
  //       /// TODO: decouple `transformLayout` and `GlobalDataLayoutAnalysis`.
  //       return !isInitOperand(op, currentValue) &&
  //              !isInitOperand(op, newValue) && valueConsumedByOp(op,
  //              newValue);
  //     })) {
  //   return true;
  // }
  // currentValue is a producer of newValue
  if (valueConsumedByOp(newDefiningOp, currentValue)) {
    // If currentValue is an init operand, then it is already transformed.
    if (isInitOperand(newDefiningOp, currentValue)) {
      return true;
    }
    // Otherwise, perform transformation down through `newDefiningOp`.
    if (auto newType = dyn_cast<ShapedType>(newValue.getType())) {
      return transform(newDefiningOp, currentValue, newValue);
    }
    // currentValue is a consumer of newValue
  } else if (valueConsumedByOp(curDefiningOp, newValue)) {
    // If newValue is an init operand, then no transformation is necessary.
    if (isInitOperand(curDefiningOp, newValue)) {
      return true;
    }
    // Otherwise, perform transformation up through `curDefiningOp`.
    if (auto newType = dyn_cast<ShapedType>(newValue.getType())) {
      return transform(curDefiningOp, currentValue, newValue);
    }
  }
  // Fail if no connecting op
  return false;
}

ArrayAttr DataLayoutTransformation::makeTransformArrayAttr(MLIRContext *ctx) {
  SmallVector<Attribute> attrs;
  attrs.push_back(TypeAttr::get(originalType));
  attrs.push_back(TypeAttr::get(transformedType));
  return ArrayAttr::get(ctx, attrs);
}

bool DataLayoutTransformation::isIntersecting(DataLayoutTransformation other) {
  return true;
}

//===----------------------------------------------------------------------===//
// DataLayoutTransformation transform implementations
//===----------------------------------------------------------------------===//

bool transformThroughOperation(tensor::InsertSliceOp insertOp,
                               DataLayoutTransformation &transform,
                               Value currentValue, Value newValue) {
  Value dest = insertOp.getDest();
  // If transform is dest->result or result->dest, no transform is needed.
  // dest->source or source->dest is not handled here..
  if (currentValue == dest || newValue == dest) {
    return true;
  }
  bool isRankExtending = (newValue == insertOp.getResult() &&
                          currentValue == insertOp.getSource());
  bool isRankReducing = (currentValue == insertOp.getResult() &&
                         newValue == insertOp.getSource());
  auto originalShape = insertOp.getStaticSizes();
  auto reducedShape = insertOp.getSourceType().getShape();
  std::optional<llvm::SmallDenseSet<unsigned>> maybeRankReducingMask =
      mlir::computeRankReductionMask(originalShape, reducedShape);
  if (!maybeRankReducingMask) {
    return false;
  }
  SmallVector<int64_t> currentCTf =
      transform.getCorrespondingTransformedIndices();
  SmallVector<int64_t> newCTf;
  int64_t cTfInd = 0;
  for (int64_t destIdx = 0; destIdx < originalShape.size(); destIdx++) {
    if (!maybeRankReducingMask->count(destIdx)) {
      newCTf.push_back(currentCTf[cTfInd++]);
    } else if (isRankExtending) {
      newCTf.push_back(-1);
    } else if (isRankReducing) {
      cTfInd++;
    }
  }
  transform.setCorrespondingTransformedIndices(newCTf);
  return true;
}

bool transformThroughOperation(tensor::ExtractSliceOp extractOp,
                               DataLayoutTransformation &transform,
                               Value currentValue, Value newValue) {
  bool isRankExtending = (newValue == extractOp.getSource() &&
                          currentValue == extractOp.getResult());
  bool isRankReducing = (currentValue == extractOp.getSource() &&
                         newValue == extractOp.getResult());
  auto originalShape = extractOp.getStaticSizes();
  auto reducedShape = extractOp.getResultType().getShape();
  std::optional<llvm::SmallDenseSet<unsigned>> maybeRankReducingMask =
      mlir::computeRankReductionMask(originalShape, reducedShape);
  if (!maybeRankReducingMask) {
    return false;
  }
  SmallVector<int64_t> currentCTf =
      transform.getCorrespondingTransformedIndices();
  SmallVector<int64_t> newCTf;
  int64_t cTfInd = 0;
  for (int64_t destIdx = 0; destIdx < originalShape.size(); destIdx++) {
    if (!maybeRankReducingMask->count(destIdx)) {
      newCTf.push_back(currentCTf[cTfInd++]);
    } else if (isRankExtending) {
      newCTf.push_back(-1);
    } else if (isRankReducing) {
      cTfInd++;
    }
  }
  transform.setCorrespondingTransformedIndices(newCTf);
  return true;
}

bool transformThroughOperation(tensor::PackOp packOp,
                               DataLayoutTransformation &transform,
                               Value currentValue, Value newValue) {
  if (!transform.getInnerTileSizes().empty() ||
      !transform.getInnerDimsPos().empty()) {
    return false;
  }
  if (currentValue != packOp.getSource() || newValue != packOp.getResult()) {
    return false;
  }
  auto origType = dyn_cast<RankedTensorType>(transform.getOriginalType());
  if (!origType) {
    return false;
  }
  SmallVector<int64_t> currentOuterDimsPerm = transform.getOuterDimsPerm();
  if (currentOuterDimsPerm.empty()) {
    currentOuterDimsPerm =
        llvm::to_vector(llvm::seq<int64_t>(0, origType.getRank()));
  }
  auto cTfInds = transform.getCorrespondingTransformedIndices();
  if (llvm::any_of(cTfInds, [](int64_t ind) { return ind == -1; })) {
    return false;
  }
  SmallVector<int64_t> packOuterDimsPerm(packOp.getOuterDimsPerm());
  SmallVector<int64_t> newOuterDimsPerm(currentOuterDimsPerm);
  for (auto [idx, permIdx] : llvm::enumerate(packOuterDimsPerm)) {
    newOuterDimsPerm[cTfInds[idx]] = cTfInds[permIdx];
  }

  SmallVector<int64_t> packInnerDimsPos(packOp.getInnerDimsPos());
  SmallVector<int64_t> newInnerDimsPos;
  auto inverseOuterDimsPos = invertPermutationVector(currentOuterDimsPerm);
  for (auto innerPos : packInnerDimsPos) {
    newInnerDimsPos.push_back(inverseOuterDimsPos[cTfInds[innerPos]]);
  }

  transform.setOuterDimsPerm(newOuterDimsPerm);
  transform.setInnerDimsPos(newInnerDimsPos);
  SmallVector<int64_t> innerTiles(packOp.getStaticInnerTiles());
  transform.setInnerTileSizes(innerTiles);

  auto newTransformedType = tensor::PackOp::inferPackedType(
      origType, transform.getInnerTileSizes(), transform.getInnerDimsPos(),
      transform.getOuterDimsPerm());
  transform.setTransformedType(newTransformedType);
  return true;
}

bool transformThroughOperation(tensor::UnPackOp unpackOp,
                               DataLayoutTransformation &transform,
                               Value currentValue, Value newValue) {
  return false;
}

bool transformThroughOperation(tensor::ReshapeOp reshapeOp,
                               DataLayoutTransformation &transform,
                               Value currentValue, Value newValue) {
  return false;
}

// TODO: Add this transform for unit_dim cases
bool transformThroughOperation(tensor::ExpandShapeOp expandOp,
                               DataLayoutTransformation &transform,
                               Value currentValue, Value newValue) {
  return false;
}

// TODO: Add this transform for unit_dim cases
bool transformThroughOperation(tensor::CollapseShapeOp collapseOp,
                               DataLayoutTransformation &transform,
                               Value currentValue, Value newValue) {
  return false;
}

// TODO: Add this transform
bool transformThroughOperation(IREE::Flow::TensorUpdateOp updateOp,
                               DataLayoutTransformation &transform,
                               Value currentValue, Value newValue) {
  return false;
}

// TODO: Add this transform
bool transformThroughOperation(IREE::Flow::TensorSliceOp sliceOp,
                               DataLayoutTransformation &transform,
                               Value currentValue, Value newValue) {
  return false;
}

bool DataLayoutTransformation::transform(Operation *op, Value currentValue,
                                         Value newValue) {
  return TypeSwitch<Operation *, bool>(op)
      .Case<tensor::InsertSliceOp>([&](tensor::InsertSliceOp insertOp) {
        return transformThroughOperation(insertOp, *this, currentValue,
                                         newValue);
      })
      .Case<tensor::ExtractSliceOp>([&](tensor::ExtractSliceOp extractOp) {
        return transformThroughOperation(extractOp, *this, currentValue,
                                         newValue);
      })
      .Case<tensor::PackOp>([&](tensor::PackOp packOp) {
        return transformThroughOperation(packOp, *this, currentValue, newValue);
      })
      .Default([](Operation *op) { return false; });
}

bool DataLayoutTransformation::combineLayout(DataLayoutTransformation other) {
  assert(correspondingTransformedIndices.size() ==
         other.correspondingTransformedIndices.size());
  bool changed = false;
  for (auto [idx, cTfInd] : llvm::enumerate(correspondingTransformedIndices)) {
    if (cTfInd == -1 && other.correspondingTransformedIndices[idx] != -1) {
      cTfInd = other.correspondingTransformedIndices[idx];
      changed = true;
    }
  }
  return changed;
}

//===----------------------------------------------------------------------===//
// Analysis helpers
//===----------------------------------------------------------------------===//

SmallVector<StringRef> getTerminalNodeIDs(Value value) {
  SmallVector<StringRef> IDs;
  DataLayoutTransformation newLayout(cast<ShapedType>(value.getType()));
  if (auto loadOp = value.getDefiningOp<IREE::Util::GlobalLoadOpInterface>()) {
    IDs.push_back(loadOp.getGlobalName());
  }
  for (Operation *op : value.getUsers()) {
    if (auto storeOp = dyn_cast<IREE::Util::GlobalStoreOpInterface>(op)) {
      IDs.push_back(storeOp.getGlobalName());
    }
  }
  return IDs;
}

//===----------------------------------------------------------------------===//
// Pass helpers
//===----------------------------------------------------------------------===//

LogicalResult transformGlobalsToNewLayout(IRRewriter &rewriter,
                                          SmallVector<Value> edgeNodes,
                                          DataLayoutTransformation *transform,
                                          GlobalOp global,
                                          SymbolTable moduleSymbols) {
  // Create a new transformed GlobalOp.
  std::string newGlobalName(global.getGlobalName().str());
  newGlobalName.append(".packed");
  auto transformedType = transform->getTransformedType();
  auto originalType = transform->getOriginalType();
  rewriter.setInsertionPoint(global);
  auto newGlobal =
      rewriter.create<GlobalOp>(global->getLoc(), StringRef(newGlobalName),
                                global.getIsMutable(), transformedType);
  moduleSymbols.insert(newGlobal);
  SymbolTable::setSymbolVisibility(newGlobal,
                                   SymbolTable::getSymbolVisibility(global));

  // Create an initializer to initialize the new global to the padding
  // value of the pack. This is necessary because we have folded the pack
  // op into the new global. Additional analysis could tell us whether the
  // padding is actually needed, but for now we always pad.
  Location globalLoc = newGlobal->getLoc();
  auto initializerOp = rewriter.create<IREE::Util::InitializerOp>(globalLoc);
  auto initializerBuilder =
      OpBuilder::atBlockBegin(initializerOp.addEntryBlock());
  // ***This is a placeholder. This needs to use the real padding value.
  auto padValue = initializerBuilder.create<arith::ConstantOp>(
      globalLoc, rewriter.getZeroAttr(transformedType.getElementType()));
  auto splatOp = initializerBuilder.create<IREE::Flow::TensorSplatOp>(
      globalLoc, transformedType, padValue, /*result_dims=*/ValueRange{});
  // util.global.store
  initializerBuilder.create<IREE::Util::GlobalStoreOp>(
      globalLoc, splatOp.getResult(), newGlobal.getName());
  initializerBuilder.create<IREE::Util::ReturnOp>(globalLoc);

  // Rewrite loads and stores to use the new global.
  SmallVector<OpFoldResult> innerTilesOfr;
  for (auto tile : transform->getInnerTileSizes()) {
    innerTilesOfr.push_back(rewriter.getIndexAttr(tile));
  }
  for (auto node : edgeNodes) {
    if (llvm::any_of(node.getUsers(), [](Operation *user) {
          return isa<IREE::Util::GlobalStoreOp>(user);
        })) {
      rewriter.setInsertionPointAfterValue(node);
      auto dest = rewriter.create<tensor::EmptyOp>(
          node.getLoc(), transformedType.getShape(),
          transformedType.getElementType());
      // // ***This is a placeholder. This needs to use the real padding value.
      auto nodePadValue = rewriter.create<arith::ConstantOp>(
          node.getLoc(),
          rewriter.getZeroAttr(transformedType.getElementType()));
      auto pack = rewriter.create<tensor::PackOp>(
          node.getLoc(), node, dest, transform->getInnerDimsPos(),
          innerTilesOfr, /*padding_value=*/nodePadValue,
          transform->getOuterDimsPerm());
      setFoldablePackUnPackAttribute(pack);
      for (auto user : node.getUsers()) {
        if (isa<IREE::Util::GlobalStoreOp>(user)) {
          rewriter.create<IREE::Util::GlobalStoreOp>(
              user->getLoc(), pack.getResult(), newGlobal);
          rewriter.eraseOp(user);
        }
      }
    }
    if (auto loadOp = node.getDefiningOp<IREE::Util::GlobalLoadOp>()) {
      rewriter.setInsertionPoint(loadOp);
      auto newLoad = rewriter.create<IREE::Util::GlobalLoadOp>(loadOp->getLoc(),
                                                               newGlobal);
      auto dest = rewriter.create<tensor::EmptyOp>(
          loadOp->getLoc(), originalType.getShape(),
          originalType.getElementType());
      auto unpack = rewriter.create<tensor::UnPackOp>(
          loadOp.getLoc(), newLoad, dest, transform->getInnerDimsPos(),
          innerTilesOfr, transform->getOuterDimsPerm());
      setFoldablePackUnPackAttribute(unpack);
      rewriter.replaceOp(loadOp, unpack.getResult());
    }
  }
  return success();
}

static SetVector<int64_t>
getRankReducedInnerTileIndices(llvm::SmallDenseSet<unsigned> rankReductionMask,
                               ArrayRef<int64_t> innerDimsPos) {
  SetVector<int64_t> rankReducedInnerTileIndices;
  for (auto [idx, dimPos] : llvm::enumerate(innerDimsPos)) {
    if (rankReductionMask.contains(dimPos)) {
      rankReducedInnerTileIndices.insert(idx);
    }
  }
  return rankReducedInnerTileIndices;
}

static void
getRankReducedPackInfo(llvm::SmallDenseSet<unsigned> rankReductionMask,
                       ArrayRef<OpFoldResult> innerTiles,
                       ArrayRef<int64_t> innerDimsPos,
                       ArrayRef<int64_t> outerDimsPerm,
                       SmallVector<OpFoldResult> &reducedInnerTiles,
                       SmallVector<int64_t> &reducedInnerDimsPos,
                       SmallVector<int64_t> &reducedOuterDimsPerm) {
  // Compute mapping from full-rank tensor indices to slice indices.
  SmallVector<int64_t> sliceIdxForFullRankIdx(outerDimsPerm.size(), -1);
  int64_t sliceIdx = 0;
  for (auto [i, idx] : llvm::enumerate(sliceIdxForFullRankIdx)) {
    if (!rankReductionMask.contains(i)) {
      idx = sliceIdx++;
    }
  }
  // Compute rank-reduced innerTiles and innerDimsPos.
  for (auto [idx, tile] : llvm::enumerate(innerTiles)) {
    if (!rankReductionMask.contains(innerDimsPos[idx])) {
      reducedInnerTiles.push_back(tile);
      reducedInnerDimsPos.push_back(sliceIdxForFullRankIdx[innerDimsPos[idx]]);
    }
  }
  // Compute rank-reduced outerDimsPerm.
  for (int64_t permIdx : outerDimsPerm) {
    if (!rankReductionMask.contains(permIdx)) {
      reducedOuterDimsPerm.push_back(sliceIdxForFullRankIdx[permIdx]);
    }
  }
}

int64_t getStaticSize(OpFoldResult ofr) {
  auto constSize = getConstantIntValue(ofr);
  if (!constSize)
    return ShapedType::kDynamic;
  return constSize.value();
}

RankedTensorType getPackedSliceType(
    RankedTensorType packedSourceType, SmallVector<OpFoldResult> sliceSizes,
    llvm::SmallDenseSet<unsigned> rankReductionMask,
    ArrayRef<int64_t> outerDimsPerm, ArrayRef<int64_t> innerDimsPos) {
  // Insert unit dims into rank-reduced dimensions so the inner dims of the
  // slice match the original full-rank tensor.

  SetVector<int64_t> rankReducedInnerTileIndices =
      getRankReducedInnerTileIndices(rankReductionMask, innerDimsPos);
  // auto inverseOuterDimsPerm = invertPermutationVector(outerDimsPerm);
  int64_t outerRank = outerDimsPerm.size();
  SmallVector<int64_t> expandedShape(packedSourceType.getShape());
  int64_t firstNonReducedIdx = -1;
  for (auto [idx, size] : llvm::enumerate(expandedShape)) {
    // Set rank-reduced outer dims and inner tiles to 1.
    if ((idx < outerRank && rankReductionMask.contains(outerDimsPerm[idx])) ||
        (idx >= outerRank &&
         rankReducedInnerTileIndices.contains(idx - outerRank))) {
      size = 1;
    } else {
      size = getStaticSize(sliceSizes[idx]);
      if (firstNonReducedIdx == -1)
        firstNonReducedIdx = idx;
    }
  }
  // Remove leading reduced dims
  SmallVector<int64_t> packedSliceShape(
      expandedShape.begin() + firstNonReducedIdx, expandedShape.end());
  return RankedTensorType::get(packedSliceShape,
                               packedSourceType.getElementType());
}

FailureOr<Value>
packSliceOfTensor(PatternRewriter &rewriter, Value slice,
                  SmallVector<OpFoldResult> sliceSizes,
                  llvm::SmallDenseSet<unsigned> rankReductionMask,
                  tensor::PackOp packOp) {
  SmallVector<OpFoldResult> sliceInnerTiles;
  SmallVector<int64_t> sliceInnerDimsPos;
  SmallVector<int64_t> sliceOuterDimsPerm;
  // The slice may not fill the full innerTiles, so use the actual packed slice
  // sizes instead of the packOp innerTiles.
  SmallVector<OpFoldResult> slicedTileSizes(
      sliceSizes.begin() + packOp.getSourceRank(), sliceSizes.end());
  getRankReducedPackInfo(rankReductionMask, slicedTileSizes,
                         packOp.getInnerDimsPos(), packOp.getOuterDimsPerm(),
                         sliceInnerTiles, sliceInnerDimsPos,
                         sliceOuterDimsPerm);

  // Pack the slice.
  Location loc = slice.getLoc();
  auto empty = tensor::PackOp::createDestinationTensor(
      rewriter, loc, slice, sliceInnerTiles, sliceInnerDimsPos,
      sliceOuterDimsPerm);
  auto pack = rewriter.create<tensor::PackOp>(
      loc, slice, empty, sliceInnerDimsPos, sliceInnerTiles,
      packOp.getPaddingValue(), sliceOuterDimsPerm);
  Value packedSlice = pack.getResult();
  setFoldablePackUnPackAttribute(pack);

  // Reshape the packed slice to fit into the full rank tensor.
  RankedTensorType expandedType =
      getPackedSliceType(packOp.getDestType(), sliceSizes, rankReductionMask,
                         packOp.getOuterDimsPerm(), packOp.getInnerDimsPos());
  auto packedSliceType = cast<ShapedType>(packedSlice.getType());
  if (llvm::equal(expandedType.getShape(), packedSliceType.getShape())) {
    return packedSlice;
  }
  auto reInds =
      getReassociationIndicesForReshape(packedSliceType, expandedType);
  if (!reInds) {
    return failure();
  }
  return rewriter
      .create<tensor::ExpandShapeOp>(loc, expandedType, packedSlice,
                                     reInds.value())
      .getResult();
}

FailureOr<Value>
unPackSliceOfTensor(PatternRewriter &rewriter, Value slice,
                    SmallVector<OpFoldResult> sliceSizes,
                    llvm::SmallDenseSet<unsigned> rankReductionMask,
                    tensor::UnPackOp unpackOp,
                    SmallVector<OpFoldResult> destMixedSizes,
                    RankedTensorType originalSliceType) {
  SmallVector<OpFoldResult> sliceInnerTiles;
  SmallVector<int64_t> sliceInnerDimsPos;
  SmallVector<int64_t> sliceOuterDimsPerm;
  SmallVector<OpFoldResult> slicedTileSizes(
      sliceSizes.begin() + unpackOp.getDestRank(), sliceSizes.end());
  getRankReducedPackInfo(rankReductionMask, slicedTileSizes,
                         unpackOp.getInnerDimsPos(),
                         unpackOp.getOuterDimsPerm(), sliceInnerTiles,
                         sliceInnerDimsPos, sliceOuterDimsPerm);

  SmallVector<int64_t> staticSliceInnerTiles =
      llvm::map_to_vector(sliceInnerTiles, getStaticSize);
  auto expectedPackedType =
      tensor::PackOp::inferPackedType(originalSliceType, staticSliceInnerTiles,
                                      sliceInnerDimsPos, sliceOuterDimsPerm);
  auto sliceType = cast<RankedTensorType>(slice.getType());
  Location loc = slice.getLoc();
  if (expectedPackedType.getRank() != sliceType.getRank()) {
    auto reInds =
        getReassociationIndicesForReshape(sliceType, expectedPackedType);
    if (!reInds.has_value()) {
      return failure();
    }
    slice = expectedPackedType.getRank() < sliceType.getRank()
                ? rewriter
                      .create<tensor::CollapseShapeOp>(loc, expectedPackedType,
                                                       slice, reInds.value())
                      .getResult()
                : rewriter
                      .create<tensor::ExpandShapeOp>(loc, expectedPackedType,
                                                     slice, reInds.value())
                      .getResult();
  }
  // Unpack the slice.
  Value empty = rewriter.create<tensor::EmptyOp>(loc, destMixedSizes,
                                                 sliceType.getElementType());
  auto unpack =
      rewriter.create<tensor::UnPackOp>(loc, slice, empty, sliceInnerDimsPos,
                                        sliceInnerTiles, sliceOuterDimsPerm);
  setFoldablePackUnPackAttribute(unpack);
  return unpack.getResult();
}

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
    SmallVector<int64_t> &collapsedSliceShape) {
  int rankDiff = tensorShape.size() - sliceShape.size();
  SmallVector<int64_t> tensorNonUnitDims, sliceNonUnitDims;
  bool foundNonUnitSliceDim = false;
  for (auto [tensorIdx, tensorSize] : enumerate(tensorShape)) {
    if (tensorSize != 1) {
      tensorNonUnitDims.push_back(tensorIdx);
      int sliceIdx = tensorIdx - rankDiff;
      if (sliceIdx >= 0) {
        if (sliceShape[sliceIdx] != 1 || foundNonUnitSliceDim) {
          collapsedSliceShape.push_back(sliceShape[sliceIdx]);
          sliceNonUnitDims.push_back(sliceIdx);
          foundNonUnitSliceDim = true;
        }
      }
      collapsedOffsets.push_back(mixedOffsets[tensorIdx]);
      collapsedSizes.push_back(mixedSizes[tensorIdx]);
      collapsedStrides.push_back(mixedStrides[tensorIdx]);
    }
  }
  auto getReassociationIndices = [](SmallVector<int64_t> nonUnitDims,
                                    ArrayRef<int64_t> shape)
      -> std::optional<SmallVector<ReassociationIndices>> {
    SetVector<int64_t> s(nonUnitDims.begin(), nonUnitDims.end());
    SmallVector<ReassociationIndices> reInd(nonUnitDims.size(),
                                            ReassociationIndices{});
    int64_t reListIdx = 0;
    for (int64_t idx = 0; idx < shape.size(); idx++) {
      reInd[reListIdx].push_back(idx);
      if (s.contains(idx)) {
        reListIdx++;
      }
    }
    if (llvm::all_of(reInd,
                     [](ReassociationIndices re) { return re.size() == 1; })) {
      return std::nullopt;
    }
    return reInd;
  };
  collapsedTensorRe = getReassociationIndices(tensorNonUnitDims, tensorShape);
  collapsedSliceRe = getReassociationIndices(sliceNonUnitDims, sliceShape);
}

static bool isLayoutFlexibleOp(Operation *op) {
  // if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
  //   if (linalg::isElementwise(linalgOp)) {
  //     return true;
  //   }
  // }
  // TODO: Add flow.tensor.slice and flow.tensor.update
  // TODO: Add unit dim tensor.collapse_shape and tensor.expand_shape
  return isa<tensor::InsertSliceOp, tensor::ExtractSliceOp,
             IREE::Util::GlobalStoreOp>(op);
}

static bool isLayoutDefiningOp(Operation *op) {
  return isa<tensor::PackOp>(op);
}

static bool isOpOnShapedTypes(Operation *op) {
  auto isShaped = [](Type t) -> bool { return isa<ShapedType>(t); };
  return any_of(op->getOperandTypes(), isShaped) ||
         any_of(op->getResultTypes(), isShaped);
}

static bool isIntermediateNode(Value value) {
  if (auto definingOp = value.getDefiningOp()) {
    if (!isLayoutFlexibleOp(definingOp) || isLayoutDefiningOp(definingOp)) {
      return false;
    }
  }
  for (auto op : value.getUsers()) {
    if (!isOpOnShapedTypes(op))
      continue;
    if (!isLayoutFlexibleOp(op) && !isLayoutDefiningOp(op))
      return false;
  }
  return true;
}

DataLayoutNodeType getNodeTypeForValue(Value value) {
  if (isIntermediateNode(value))
    return DataLayoutNodeType::INTERMEDIATE;
  return DataLayoutNodeType::BARRIER;
}

static StringAttr getNodeTypeStringAttr(MLIRContext *ctx,
                                        DataLayoutNodeType type) {
  switch (type) {
  case DataLayoutNodeType::UNINITIALIZED:
    return StringAttr::get(ctx, "UNINITIALIZED");
  case DataLayoutNodeType::INTERMEDIATE:
    return StringAttr::get(ctx, "INTERMEDIATE");
  case DataLayoutNodeType::BARRIER:
    return StringAttr::get(ctx, "BARRIER");
  default:
    assert(false && "invalid DataLayoutNodeType");
  }
}

static DataLayoutNodeType getNodeTypeFromStringAttr(StringAttr attr) {
  if (attr.getValue().equals(StringRef("UNINITIALIZED")))
    return DataLayoutNodeType::UNINITIALIZED;
  if (attr.getValue().equals(StringRef("INTERMEDIATE")))
    return DataLayoutNodeType::INTERMEDIATE;
  return DataLayoutNodeType::BARRIER;
}

void setNodeTypeAttribute(Operation *op, DataLayoutNodeType nodeType) {
  op->setAttr(kDataLayoutNodeTypeAttr,
              getNodeTypeStringAttr(op->getContext(), nodeType));
  return;
}

void setFoldablePackUnPackAttribute(Operation *op) {
  op->setAttr(kFoldablePackUnPack, UnitAttr::get(op->getContext()));
  return;
}

bool hasFoldablePackUnPackAttribute(Operation *op) {
  return static_cast<bool>(op->getAttrOfType<UnitAttr>(kFoldablePackUnPack));
}

std::optional<DataLayoutNodeType> getNodeTypeFromAttr(Operation *op) {
  if (auto attr = op->getAttrOfType<StringAttr>(kDataLayoutNodeTypeAttr)) {
    return getNodeTypeFromStringAttr(attr);
  }
  return std::nullopt;
}

void setDataLayoutTransformationAttributes(Operation *op,
                                           DataLayoutTransformation *transform,
                                           StringRef transformID) {
  op->setAttr(transformID, transform->makeTransformArrayAttr(op->getContext()));
  return;
}

} // namespace mlir::iree_compiler::GlobalOptimization
