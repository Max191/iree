// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Parser/Parser.h"

// clang-format off: must be included after all LLVM/MLIR headers.
#include "iree/compiler/Dialect/Util/IR/UtilEnums.cpp.inc" // IWYU pragma: keep
// clang-format on

namespace mlir::iree_compiler::IREE::Util {

//===----------------------------------------------------------------------===//
// AffineSeedDependency
//===----------------------------------------------------------------------===//

namespace {

class AffineDimCoefficientVisitor
    : public AffineExprVisitor<AffineDimCoefficientVisitor,
                               std::optional<int64_t>> {
public:
  explicit AffineDimCoefficientVisitor(unsigned dimPosition)
      : dimPosition(dimPosition) {}

  std::optional<int64_t> visitConstantExpr(AffineConstantExpr expr) {
    return 0;
  }

  std::optional<int64_t> visitDimExpr(AffineDimExpr expr) {
    return expr.getPosition() == dimPosition ? 1 : 0;
  }

  std::optional<int64_t> visitSymbolExpr(AffineSymbolExpr expr) { return 0; }

  std::optional<int64_t> visitAddExpr(AffineBinaryOpExpr expr) {
    std::optional<int64_t> lhs = visit(expr.getLHS());
    std::optional<int64_t> rhs = visit(expr.getRHS());
    if (!lhs || !rhs) {
      return std::nullopt;
    }
    return llvm::checkedAdd(*lhs, *rhs);
  }

  std::optional<int64_t> visitMulExpr(AffineBinaryOpExpr expr) {
    std::optional<int64_t> lhs = visit(expr.getLHS());
    std::optional<int64_t> rhs = visit(expr.getRHS());
    if (!lhs || !rhs) {
      return std::nullopt;
    }
    if (auto rhsConstant = dyn_cast<AffineConstantExpr>(expr.getRHS())) {
      return llvm::checkedMul(*lhs, rhsConstant.getValue());
    }
    if (auto lhsConstant = dyn_cast<AffineConstantExpr>(expr.getLHS())) {
      return llvm::checkedMul(lhsConstant.getValue(), *rhs);
    }
    return *lhs == 0 && *rhs == 0 ? std::optional<int64_t>(0) : std::nullopt;
  }

private:
  std::optional<int64_t> visitInvalidExpr(AffineBinaryOpExpr expr) {
    return std::nullopt;
  }

  unsigned dimPosition;
};

static std::optional<int64_t> getAffineDimCoefficient(AffineExpr expr,
                                                      unsigned dimPosition) {
  return AffineDimCoefficientVisitor(dimPosition).visit(expr);
}

} // namespace

AffineSeedDependency AffineSeedDependency::getIndependent() {
  return AffineSeedDependency(/*expression=*/AffineExpr(),
                              /*seedPositions=*/SeedPositionMap{},
                              /*invalidatedSeeds=*/InvalidatedSeedSet{},
                              /*hasIndependentOffset=*/true);
}

AffineSeedDependency AffineSeedDependency::getSeed(Value seed,
                                                   unsigned position) {
  SeedPositionMap seedPositions;
  seedPositions[seed] = position;
  return AffineSeedDependency(getAffineDimExpr(position, seed.getContext()),
                              std::move(seedPositions),
                              /*invalidatedSeeds=*/InvalidatedSeedSet{},
                              /*hasIndependentOffset=*/false);
}

AffineSeedDependency
AffineSeedDependency::getKnown(AffineExpr expression,
                               SeedPositionMap seedPositions,
                               InvalidatedSeedSet invalidatedSeeds,
                               bool hasIndependentOffset) {
  return AffineSeedDependency(expression, std::move(seedPositions),
                              std::move(invalidatedSeeds),
                              hasIndependentOffset);
}

AffineSeedDependency AffineSeedDependency::getUnknown() {
  return AffineSeedDependency(Kind::Unknown);
}

static bool mergeSeedPositionMaps(
    const AffineSeedDependency::SeedPositionMap &lhs,
    const AffineSeedDependency::SeedPositionMap &rhs,
    AffineSeedDependency::SeedPositionMap &merged) {
  merged = lhs;
  for (auto [seed, position] : rhs) {
    auto [it, inserted] = merged.try_emplace(seed, position);
    if (!inserted && it->second != position) {
      return false;
    }
  }
  return true;
}

static void addInvalidatedSeeds(
    const AffineSeedDependency::InvalidatedSeedSet &from,
    AffineSeedDependency::InvalidatedSeedSet &to) {
  for (Value seed : from) {
    to.insert(seed);
  }
}

static void addPotentiallyDependentSeeds(
    const AffineSeedDependency &dependency,
    AffineSeedDependency::InvalidatedSeedSet &invalidatedSeeds) {
  addInvalidatedSeeds(dependency.getInvalidatedSeeds(), invalidatedSeeds);
  for (auto [seed, position] : dependency.getSeedPositions()) {
    invalidatedSeeds.insert(seed);
  }
}

static AffineExpr getZeroExpression(MLIRContext *context) {
  return getAffineConstantExpr(0, context);
}

static AffineExpr getExpressionOrZero(const AffineSeedDependency &dependency,
                                      MLIRContext *context) {
  AffineExpr expression = dependency.getExpression(context);
  return expression ? expression : getZeroExpression(context);
}

AffineSeedDependency
AffineSeedDependency::join(const AffineSeedDependency &lhs,
                           const AffineSeedDependency &rhs) {
  if (lhs.isUninitialized()) {
    return rhs;
  }
  if (rhs.isUninitialized()) {
    return lhs;
  }
  if (lhs.isUnknown() || rhs.isUnknown()) {
    return getUnknown();
  }

  SeedPositionMap joinedSeedPositions;
  if (!mergeSeedPositionMaps(lhs.seedPositions, rhs.seedPositions,
                             joinedSeedPositions)) {
    return getUnknown();
  }
  InvalidatedSeedSet joinedInvalidatedSeeds;
  addInvalidatedSeeds(lhs.invalidatedSeeds, joinedInvalidatedSeeds);
  addInvalidatedSeeds(rhs.invalidatedSeeds, joinedInvalidatedSeeds);

  MLIRContext *context = nullptr;
  if (lhs.expression) {
    context = lhs.expression.getContext();
  } else if (rhs.expression) {
    context = rhs.expression.getContext();
  }
  AffineExpr lhsExpression =
      context ? getExpressionOrZero(lhs, context) : AffineExpr();
  AffineExpr rhsExpression =
      context ? getExpressionOrZero(rhs, context) : AffineExpr();
  if (lhsExpression != rhsExpression) {
    addPotentiallyDependentSeeds(lhs, joinedInvalidatedSeeds);
    addPotentiallyDependentSeeds(rhs, joinedInvalidatedSeeds);
    return getKnown(/*expression=*/AffineExpr(), std::move(joinedSeedPositions),
                    std::move(joinedInvalidatedSeeds),
                    lhs.independentOffset || rhs.independentOffset);
  }
  return getKnown(lhs.expression ? lhs.expression : rhs.expression,
                  std::move(joinedSeedPositions),
                  std::move(joinedInvalidatedSeeds),
                  lhs.independentOffset || rhs.independentOffset);
}

AffineSeedDependency
AffineSeedDependency::scale(const AffineSeedDependency &dependency,
                            int64_t scale) {
  if (dependency.isUninitialized() || dependency.isUnknown()) {
    return dependency;
  }
  if (scale == 0) {
    return getIndependent();
  }
  if (dependency.isIndependent()) {
    return dependency;
  }

  AffineExpr scaledExpression;
  if (dependency.expression) {
    for (auto [seed, position] : dependency.seedPositions) {
      std::optional<int64_t> coefficient =
          getAffineDimCoefficient(dependency.expression, position);
      if (coefficient &&
          !llvm::checkedMul(*coefficient, scale).has_value()) {
        return getUnknownForDependentSeeds({dependency});
      }
    }
    scaledExpression = dependency.expression * scale;
  }
  return getKnown(scaledExpression, dependency.seedPositions,
                  dependency.invalidatedSeeds, dependency.independentOffset);
}

AffineSeedDependency AffineSeedDependency::add(const AffineSeedDependency &lhs,
                                               const AffineSeedDependency &rhs,
                                               int64_t lhsScale,
                                               int64_t rhsScale) {
  if (lhs.isUninitialized() || rhs.isUninitialized()) {
    return AffineSeedDependency();
  }
  if (lhs.isUnknown() || rhs.isUnknown()) {
    return getUnknown();
  }

  AffineSeedDependency scaledLhs = scale(lhs, lhsScale);
  AffineSeedDependency scaledRhs = scale(rhs, rhsScale);
  if (scaledLhs.isUnknown() || scaledRhs.isUnknown()) {
    return getUnknown();
  }

  SeedPositionMap resultSeedPositions;
  if (!mergeSeedPositionMaps(scaledLhs.seedPositions, scaledRhs.seedPositions,
                             resultSeedPositions)) {
    return getUnknown();
  }
  InvalidatedSeedSet resultInvalidatedSeeds = scaledLhs.invalidatedSeeds;
  addInvalidatedSeeds(scaledRhs.invalidatedSeeds, resultInvalidatedSeeds);

  MLIRContext *context = nullptr;
  if (scaledLhs.expression) {
    context = scaledLhs.expression.getContext();
  } else if (scaledRhs.expression) {
    context = scaledRhs.expression.getContext();
  }
  AffineExpr resultExpression;
  if (context) {
    resultExpression = getExpressionOrZero(scaledLhs, context) +
                       getExpressionOrZero(scaledRhs, context);
  }
  return getKnown(resultExpression, std::move(resultSeedPositions),
                  std::move(resultInvalidatedSeeds),
                  scaledLhs.independentOffset || scaledRhs.independentOffset);
}

static AffineSeedDependency mapNonLinearAffineExpr(
    const AffineSeedDependency &dependency,
    llvm::function_ref<AffineExpr(AffineExpr)> mapExpr) {
  if (dependency.isUninitialized() || dependency.isUnknown()) {
    return dependency;
  }
  if (dependency.isIndependent()) {
    return AffineSeedDependency::getIndependent();
  }
  if (dependency.hasIndependentOffset() || dependency.hasInvalidatedSeeds()) {
    return AffineSeedDependency::getUnknownForDependentSeeds({dependency});
  }
  MLIRContext *context = nullptr;
  for (auto [seed, position] : dependency.getSeedPositions()) {
    context = seed.getContext();
    break;
  }
  assert(context && "expected dependent affine expression context");
  AffineExpr expression = mapExpr(dependency.getExpression(context));
  return AffineSeedDependency::getKnown(expression, dependency.getSeedPositions(),
                                        /*invalidatedSeeds=*/{},
                                        /*hasIndependentOffset=*/false);
}

AffineSeedDependency
AffineSeedDependency::floorDiv(const AffineSeedDependency &dependency,
                               int64_t divisor) {
  if (divisor == 0) {
    return getUnknownForDependentSeeds({dependency});
  }
  return mapNonLinearAffineExpr(
      dependency, [&](AffineExpr expr) { return expr.floorDiv(divisor); });
}

AffineSeedDependency
AffineSeedDependency::ceilDiv(const AffineSeedDependency &dependency,
                              int64_t divisor) {
  if (divisor == 0) {
    return getUnknownForDependentSeeds({dependency});
  }
  return mapNonLinearAffineExpr(
      dependency, [&](AffineExpr expr) { return expr.ceilDiv(divisor); });
}

AffineSeedDependency AffineSeedDependency::mod(
    const AffineSeedDependency &dependency, int64_t modulus) {
  if (modulus == 0) {
    return getUnknownForDependentSeeds({dependency});
  }
  return mapNonLinearAffineExpr(dependency,
                                [&](AffineExpr expr) { return expr % modulus; });
}

AffineSeedDependency AffineSeedDependency::getUnknownForDependentSeeds(
    ArrayRef<AffineSeedDependency> dependencies) {
  SeedPositionMap seedPositions;
  InvalidatedSeedSet invalidatedSeeds;
  for (const AffineSeedDependency &dependency : dependencies) {
    if (dependency.isUninitialized()) {
      return AffineSeedDependency();
    }
    if (dependency.isUnknown()) {
      return getUnknown();
    }
    if (!mergeSeedPositionMaps(seedPositions, dependency.seedPositions,
                               seedPositions)) {
      return getUnknown();
    }
    addPotentiallyDependentSeeds(dependency, invalidatedSeeds);
  }
  return getKnown(/*expression=*/AffineExpr(), std::move(seedPositions),
                  std::move(invalidatedSeeds),
                  /*hasIndependentOffset=*/false);
}

bool AffineSeedDependency::isIndependent() const {
  if (!isKnown() || hasInvalidatedSeeds()) {
    return false;
  }
  if (seedPositions.empty()) {
    return true;
  }
  if (!expression) {
    return true;
  }
  auto constantExpr = dyn_cast<AffineConstantExpr>(expression);
  return static_cast<bool>(constantExpr);
}

AffineExpr AffineSeedDependency::getExpression(MLIRContext *context) const {
  assert(isKnown() && "expected known affine seed dependency");
  if (expression) {
    return expression;
  }
  return getZeroExpression(context);
}

bool AffineSeedDependency::hasInvalidatedSeeds() const {
  assert(isKnown() && "expected known affine seed dependency");
  return !invalidatedSeeds.empty();
}

std::optional<int64_t> AffineSeedDependency::getCoefficient(Value seed) const {
  assert(isKnown() && "expected known affine seed dependency");
  if (invalidatedSeeds.contains(seed)) {
    return std::nullopt;
  }
  auto it = seedPositions.find(seed);
  if (it == seedPositions.end()) {
    return 0;
  }
  return getAffineDimCoefficient(getExpression(seed.getContext()), it->second);
}

bool AffineSeedDependency::operator==(
    const AffineSeedDependency &rhs) const {
  if (kind != rhs.kind) {
    return false;
  }
  if (!isKnown()) {
    return true;
  }
  if (expression != rhs.expression ||
      independentOffset != rhs.independentOffset ||
      seedPositions.size() != rhs.seedPositions.size() ||
      invalidatedSeeds.size() != rhs.invalidatedSeeds.size()) {
    return false;
  }
  for (auto [seed, position] : seedPositions) {
    auto rhsIt = rhs.seedPositions.find(seed);
    if (rhsIt == rhs.seedPositions.end() || rhsIt->second != position) {
      return false;
    }
  }
  for (Value seed : invalidatedSeeds) {
    if (!rhs.invalidatedSeeds.contains(seed)) {
      return false;
    }
  }
  return true;
}

void AffineSeedDependency::print(raw_ostream &os) const {
  if (isUninitialized()) {
    os << "uninitialized";
    return;
  }
  if (isUnknown()) {
    os << "unknown";
    return;
  }
  os << "known expr=";
  if (expression) {
    os << expression;
  } else {
    os << 0;
  }
  if (independentOffset) {
    os << " + <independent>";
  }
  for (auto [seed, position] : seedPositions) {
    os << " " << seed << "=d" << position;
  }
  for (Value seed : invalidatedSeeds) {
    os << " invalidated(" << seed << ")";
  }
}

//===----------------------------------------------------------------------===//
// !util.buffer
//===----------------------------------------------------------------------===//

bool BufferType::isAccessStorageCompatible(Type accessType) const {
  return isa<IREE::Util::BufferType>(accessType);
}

Value BufferType::inferSizeFromValue(Location loc, Value value,
                                     OpBuilder &builder) const {
  return builder.createOrFold<IREE::Util::BufferSizeOp>(
      loc, builder.getIndexType(), value);
}

Value BufferType::createSubrangeOp(Location loc, Value resource,
                                   Value resourceSize, Value subrangeOffset,
                                   Value subrangeLength,
                                   OpBuilder &builder) const {
  return IREE::Util::BufferSubspanOp::create(
      builder, loc, resource, resourceSize, subrangeOffset, subrangeLength);
}

//===----------------------------------------------------------------------===//
// !util.list<T>
//===----------------------------------------------------------------------===//

static LogicalResult parseListElementType(AsmParser &parser,
                                          Type &elementType) {
  if (succeeded(parser.parseOptionalQuestion())) {
    elementType = IREE::Util::VariantType::get(parser.getContext());
    return success();
  }
  Type type;
  if (succeeded(parser.parseType(type))) {
    elementType = type;
    return success();
  }
  return failure();
}

static void printListElementType(AsmPrinter &printer, Type elementType) {
  if (isa<IREE::Util::VariantType>(elementType)) {
    printer << "?";
  } else {
    printer << elementType;
  }
}

// static
bool ListType::isCompatible(Type type) { return true; }

// static
bool ListType::canImplicitlyCast(Type from, Type to) {
  if (isa<VariantType>(from) || isa<VariantType>(to)) {
    return true;
  } else if (isa<ObjectType>(from) &&
             IREE::Util::ObjectType::isCompatible(to)) {
    return true;
  } else if (IREE::Util::ObjectType::isCompatible(from) &&
             isa<ObjectType>(to)) {
    return true;
  } else if (isa<TensorType>(from) && isa<TensorType>(to)) {
    return true;
  }
  return from == to;
}

// static
LogicalResult ListType::verify(function_ref<InFlightDiagnostic()> emitError,
                               Type elementType) {
  if (!isCompatible(elementType)) {
    return emitError() << "invalid element type for a list: " << elementType;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// !util.ptr<T>
//===----------------------------------------------------------------------===//

// static
bool PtrType::isCompatible(Type type) { return true; }

// static
LogicalResult PtrType::verify(function_ref<InFlightDiagnostic()> emitError,
                              Type targetType) {
  if (!isCompatible(targetType)) {
    return emitError() << "invalid target type for a pointer: " << targetType;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// !util.object
//===----------------------------------------------------------------------===//

// static
bool ObjectType::isCompatible(Type type) {
  if (isa<ObjectType>(type)) {
    // Already an object.
    return true;
  } else if (type.isIntOrIndexOrFloat() || isa<ComplexType>(type)) {
    // Ignore known primitive types.
    return false;
  }
  // Assume all other types (user types, buffers, etc) can be objects.
  return true;
}

//===----------------------------------------------------------------------===//
// Op utilities common in util patterns and folders
//===----------------------------------------------------------------------===//

// Walks up the ancestors of |sourceBlock| until |targetBlock| is reached.
// Returns an insertion point in the |targetBlock|.
static std::pair<Block *, Block::iterator>
findCommonBlockInsertionPoint(Block *targetBlock, Block *sourceBlock,
                              Block::iterator sourceInsertionPoint) {
  auto *ancestorOp = targetBlock->findAncestorOpInBlock(*sourceInsertionPoint);
  if (ancestorOp) {
    return std::make_pair(ancestorOp->getBlock(), Block::iterator(ancestorOp));
  }
  return std::make_pair(sourceBlock, sourceInsertionPoint);
}

bool isValueUsableForOp(Value value, Block *block,
                        Block::iterator insertionPoint) {
  // If the insertion point is nested within an op in the defining block we can
  // use the parent op as the insertion point to check.
  auto *definingBlock = value.getParentBlock();
  std::tie(block, insertionPoint) =
      findCommonBlockInsertionPoint(definingBlock, block, insertionPoint);
  if (block == nullptr) {
    // Op is not in a block; can't analyze (maybe?).
    return false;
  }
  if (definingBlock == block) {
    // Defined in the same block; ensure block order.
    if (isa<BlockArgument>(value)) {
      return true;
    }
    if (insertionPoint == block->end()) {
      return true;
    }
    if (value.getDefiningOp()->isBeforeInBlock(&*insertionPoint)) {
      return true;
    }
  } else if (definingBlock->isEntryBlock() &&
             isa<mlir::FunctionOpInterface>(definingBlock->getParentOp())) {
    // Function entry block always dominates - fast path for constants.
    return true;
  } else {
    // See if block the value is defined in dominates the forOp block.
    // NOTE: this can be very expensive - we hope the fast paths have taken
    // care of things before reaching this point.
    DominanceInfo dominanceInfo(block->getParentOp());
    return dominanceInfo.dominates(definingBlock, block);
  }
  return false;
}

bool isValueUsableForOp(Value value, Operation *op) {
  return isValueUsableForOp(value, op->getBlock(), Block::iterator(op));
}

bool tryMoveProducerBefore(Value value, Operation *consumerOp) {
  auto *producerOp = value.getDefiningOp();
  if (!producerOp) {
    return true; // block arg, ok to use
  }

  // Producers and consumers in the same block are easy to check.
  if (producerOp->getBlock() == consumerOp->getBlock()) {
    if (producerOp->isBeforeInBlock(consumerOp)) {
      // Producer comes before the consumer in the same block and already
      // satisfies the request based on SSA dominance.
      return true;
    }

    // Recursively try to move each operand.
    // TODO(benvanik): change to a worklist to avoid potential stack explosion.
    for (auto operand : producerOp->getOperands()) {
      // Can't move `producerOp` if it is defined by `consumerOp`.
      if (operand.getDefiningOp() == consumerOp) {
        return false;
      }
      if (!tryMoveProducerBefore(operand, consumerOp)) {
        return false;
      }
    }

    producerOp->moveBefore(consumerOp);
    return true;
  }

  // If the value is directly usable from another block (dominates, etc) then
  // the condition is already satisfied. We don't move ops that don't satisfy
  // this yet.
  if (isValueUsableForOp(value, consumerOp)) {
    return true;
  }

  // Could support more cases of satisfaction checks or movement. Ops that exist
  // in ancestors (like those implicitly captured by nested scf.if/scf.for ops)
  // are good candidates to check for.
  return false;
}

Operation *materializeConstant(OpBuilder &builder, Location loc,
                               TypedAttr attr) {
  // Try arith::ConstantOp for compatible types (mostly builtins like index,
  // i32, etc) and fall back to asking the dialects.
  Type type = attr.getType();
  if (arith::ConstantOp::isBuildableWith(attr, type)) {
    return arith::ConstantOp::create(builder, loc, type, attr);
  } else if (auto *op = attr.getDialect().materializeConstant(builder, attr,
                                                              type, loc)) {
    return op;
  } else if (auto *op = type.getDialect().materializeConstant(builder, attr,
                                                              type, loc)) {
    return op;
  }
  return nullptr;
}

bool isPublicOrExternal(CallableOpInterface callableOp) {
  if (auto symbolOp = dyn_cast<SymbolOpInterface>(callableOp.getOperation())) {
    if (symbolOp.isPublic()) {
      return true;
    }
  }
  auto *region = callableOp.getCallableRegion();
  if (!region || region->empty()) {
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Global and structural interface utilities
//===----------------------------------------------------------------------===//

// Returns true if the given |accessType| is compatible with the |globalType|.
// For example, this will return true if the global type is a tensor<?xf32>
// and the access is tensor<4xf32>.
static bool isGlobalTypeCompatible(Type globalType, Type accessType) {
  // If one is a shaped type, then they both must be and have compatible
  // shapes.
  if (isa<ShapedType>(globalType) && isa<ShapedType>(accessType)) {
    return succeeded(mlir::verifyCompatibleShape(globalType, accessType));
  }

  if (auto knownType = dyn_cast<IREE::Util::GlobalTypeInterface>(globalType)) {
    return knownType.isAccessStorageCompatible(accessType);
  }

  // Otherwise, the types must be the same.
  return globalType == accessType;
}

LogicalResult detail::verifyGlobalOp(IREE::Util::GlobalOpInterface globalOp) {
  auto initialValue = globalOp.getGlobalInitialValue();
  if (auto typedInitialValue = dyn_cast_if_present<TypedAttr>(initialValue)) {
    // Ensure the value is something we can convert to a const.
    if (!isGlobalTypeCompatible(globalOp.getGlobalType(),
                                typedInitialValue.getType())) {
      return globalOp->emitOpError()
             << "initial value type mismatch; global "
             << globalOp.getGlobalName() << " is " << globalOp.getGlobalType()
             << " but initial value provided is "
             << typedInitialValue.getType();
    }
  }
  return success();
}

LogicalResult
detail::verifyGlobalAddressOp(GlobalAddressOpInterface addressOp,
                              SymbolTableCollection &symbolTable) {
  if (!isa_and_nonnull<IREE::Util::GlobalOpInterface>(
          symbolTable.lookupNearestSymbolFrom(addressOp.getOperation(),
                                              addressOp.getGlobalAttr()))) {
    return addressOp->emitOpError(
        "attribute 'global' failed to satisfy constraint: flat symbol "
        "reference attribute referencing to a 'IREE::Util::GlobalOpInterface' "
        "symbol");
  }
  auto globalOp =
      lookupGlobalOp(addressOp, addressOp.getGlobalAttr(), symbolTable);
  if (!globalOp) {
    return addressOp->emitOpError()
           << "undefined global: " << addressOp.getGlobalAttr();
  }
  // TODO(benvanik): allow type conversion here? probably better on the indirect
  // access ops instead as it's then easier to fold the conversion.
  if (addressOp.isGlobalImmutable() && globalOp.isGlobalMutable()) {
    return addressOp->emitOpError()
           << "is marked as immutable but the global is mutable";
  }
  return success();
}

LogicalResult detail::verifyGlobalLoadOp(GlobalLoadOpInterface loadOp,
                                         SymbolTableCollection &symbolTable) {
  auto globalOp = lookupGlobalOp(loadOp, loadOp.getGlobalAttr(), symbolTable);
  if (!globalOp) {
    return loadOp->emitOpError()
           << "undefined global: " << loadOp.getGlobalAttr();
  }
  auto loadType = loadOp->getResult(0).getType();
  if (!isGlobalTypeCompatible(globalOp.getGlobalType(), loadType)) {
    return loadOp->emitOpError()
           << "global type mismatch; global " << globalOp.getGlobalName()
           << " is " << globalOp.getGlobalType() << " but load is " << loadType;
  }
  if (loadOp.isGlobalImmutable() && globalOp.isGlobalMutable()) {
    return loadOp->emitOpError()
           << "is marked as immutable but the global is mutable";
  }
  return success();
}

LogicalResult detail::verifyGlobalStoreOp(GlobalStoreOpInterface storeOp,
                                          SymbolTableCollection &symbolTable) {
  auto globalOp = lookupGlobalOp(storeOp, storeOp.getGlobalAttr(), symbolTable);
  if (!globalOp) {
    return storeOp->emitOpError()
           << "undefined global: " << storeOp.getGlobalAttr();
  }
  auto storeType = storeOp.getStoredGlobalValue().getType();
  if (globalOp.getGlobalType() != storeType) {
    return storeOp->emitOpError()
           << "global type mismatch; global " << globalOp.getGlobalName()
           << " is " << globalOp.getGlobalType() << " but store is "
           << storeType;
  }
  return success();
}

IREE::Util::GlobalOpInterface
lookupGlobalOp(Operation *accessorOp, SymbolRefAttr globalRefAttr,
               SymbolTableCollection &symbolTable) {
  return symbolTable.lookupNearestSymbolFrom<IREE::Util::GlobalOpInterface>(
      accessorOp->getParentOp(), globalRefAttr);
}

//===----------------------------------------------------------------------===//
// IREE::Util::TiedOpInterface
//===----------------------------------------------------------------------===//

void detail::getAllTiedOperands(Operation *op,
                                SmallVectorImpl<int64_t> &indices) {
  if (auto tiedOperandsAttr = op->getAttrOfType<ArrayAttr>(
          IREE::Util::TiedOpInterface::getStorageAttrName())) {
    for (auto indexAttr : tiedOperandsAttr.getAsRange<IntegerAttr>()) {
      indices.push_back(indexAttr.getInt());
    }
  } else if (auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(op)) {
    indices.assign(op->getNumResults(),
                   IREE::Util::TiedOpInterface::kUntiedIndex);
  } else if (auto callableOp = dyn_cast<CallableOpInterface>(op)) {
    indices.assign(callableOp.getResultTypes().size(),
                   IREE::Util::TiedOpInterface::kUntiedIndex);
  }
}

std::optional<unsigned>
detail::getTiedResultOperandIndex(Operation *op, unsigned resultIndex) {
  auto storageAttr = op->getAttrOfType<ArrayAttr>(
      IREE::Util::TiedOpInterface::getStorageAttrName());
  if (!storageAttr) {
    return std::nullopt;
  }
  auto valueAttrs = storageAttr.getValue();
  if (valueAttrs.empty() || resultIndex >= valueAttrs.size()) {
    return std::nullopt;
  }
  if (auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(op)) {
    auto indexAndLength = tiedOp.getTiedResultsIndexAndLength();
    if (resultIndex < indexAndLength.first) {
      return std::nullopt;
    }
    resultIndex -= indexAndLength.first;
    if (resultIndex >= indexAndLength.second) {
      return std::nullopt;
    }
  }
  int64_t value = cast<IntegerAttr>(valueAttrs[resultIndex]).getInt();
  if (value == IREE::Util::TiedOpInterface::kUntiedIndex) {
    return std::nullopt;
  }
  if (auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(op)) {
    unsigned tiedOperandsOffset = tiedOp.getTiedOperandsIndexAndLength().first;
    return tiedOperandsOffset + static_cast<unsigned>(value);
  } else {
    return static_cast<unsigned>(value);
  }
}

void detail::setTiedResultOperandIndex(Operation *op, unsigned resultIndex,
                                       std::optional<unsigned> operandIndex) {
  auto tiedOp = cast<IREE::Util::TiedOpInterface>(op);
  auto resultRange = tiedOp.getTiedResultsIndexAndLength();
  resultIndex -= resultRange.first;

  auto indices = getTiedResultOperandIndices(op);
  if (indices.empty()) {
    indices.resize(resultRange.second,
                   IREE::Util::TiedOpInterface::kUntiedIndex);
  } else {
    // Well, getTiedResultOperandIndices() returns indices into the full range
    // of the op, but in the attribute, we expect to store ranges into the range
    // returned by `getTiedOperandsIndexAndLength`.
    unsigned tiedOperandsOffset = tiedOp.getTiedOperandsIndexAndLength().first;
    for (auto &index : indices) {
      if (index != TiedOpInterface::kUntiedIndex) {
        index -= tiedOperandsOffset;
      }
    }
  }

  indices[resultIndex] =
      operandIndex.value_or(IREE::Util::TiedOpInterface::kUntiedIndex);
  op->setAttr(IREE::Util::TiedOpInterface::getStorageAttrName(),
              Builder(op).getIndexArrayAttr(indices));
}

SmallVector<int64_t> detail::getTiedResultOperandIndices(Operation *op) {
  SmallVector<int64_t> indices;
  auto storageAttr = op->getAttrOfType<ArrayAttr>(
      IREE::Util::TiedOpInterface::getStorageAttrName());
  if (!storageAttr) {
    return indices;
  }
  auto valueAttrs = storageAttr.getValue();
  if (valueAttrs.empty()) {
    return indices;
  }
  auto tiedOp = cast<IREE::Util::TiedOpInterface>(op);
  auto resultRange = tiedOp.getTiedResultsIndexAndLength();
  unsigned tiedOperandsOffset = tiedOp.getTiedOperandsIndexAndLength().first;
  indices.resize(resultRange.second);
  for (unsigned i = 0; i < valueAttrs.size(); ++i) {
    int64_t index = cast<IntegerAttr>(valueAttrs[i]).getInt();
    indices[i] = index != IREE::Util::TiedOpInterface::kUntiedIndex
                     ? tiedOperandsOffset + index
                     : IREE::Util::TiedOpInterface::kUntiedIndex;
  }
  return indices;
}

// static
Value TiedOpInterface::findTiedBaseValue(Value derivedValue) {
  Value baseValue = derivedValue;
  while (auto definingOp = dyn_cast_if_present<IREE::Util::TiedOpInterface>(
             baseValue.getDefiningOp())) {
    auto tiedValue = definingOp.getTiedResultOperand(baseValue);
    if (!tiedValue) {
      break;
    }
    baseValue = tiedValue;
  }
  return baseValue;
}

// static
bool TiedOpInterface::hasAnyTiedUses(Value value) {
  return llvm::any_of(value.getUses(), [](auto &use) {
    if (auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(use.getOwner())) {
      return tiedOp.isOperandTied(use.getOperandNumber());
    }
    return false;
  });
}

bool detail::isOperandTied(Operation *op, unsigned operandIndex) {
  if (auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(op)) {
    auto tiedIndices = tiedOp.getTiedResultOperandIndices();
    return llvm::find(tiedIndices, operandIndex) != tiedIndices.end();
  }
  return false;
}

SmallVector<Value> detail::getOperandTiedResults(Operation *op,
                                                 unsigned operandIndex) {
  auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(op);
  if (!tiedOp) {
    return {};
  }
  auto resultRange = tiedOp.getTiedResultsIndexAndLength();
  SmallVector<Value> results;
  auto tiedIndices = tiedOp.getTiedResultOperandIndices();
  for (unsigned i = 0; i < tiedIndices.size(); ++i) {
    if (tiedIndices[i] == operandIndex) {
      results.push_back(op->getResult(resultRange.first + i));
    }
  }
  return results;
}

LogicalResult detail::verifyTiedOp(IREE::Util::TiedOpInterface tiedOp) {
  auto tiedOperandIndices = tiedOp.getTiedResultOperandIndices();
  if (tiedOperandIndices.empty()) {
    return success();
  }
  auto resultRange = tiedOp.getTiedResultsIndexAndLength();
  if (tiedOperandIndices.size() != resultRange.second) {
    return tiedOp.emitError("op results/tied operand indices mismatch");
  }
  return success();
}

void excludeTiedOperandAndResultIndices(
    ArrayRef<unsigned> excludedOperandIndices,
    ArrayRef<unsigned> excludedResultIndices,
    SmallVector<int64_t> &tiedOperandIndices) {
  SmallVector<int64_t> oldTiedOperandIndices = tiedOperandIndices;
  tiedOperandIndices.clear();

  // To adjust operand indices we need to know the how many operands to offset
  // the indices by - if 2 operands before operand N were removed then we know
  // it needs to be -2. This is nasty but that's why we have this helper
  // function.
  unsigned numBits = 1;
  if (!excludedOperandIndices.empty()) {
    numBits += *std::max_element(excludedOperandIndices.begin(),
                                 excludedOperandIndices.end());
  }
  llvm::BitVector excludedOperands(numBits, false);
  for (unsigned i = 0; i < excludedOperandIndices.size(); ++i) {
    excludedOperands[excludedOperandIndices[i]] = true;
  }

  for (auto it : llvm::enumerate(oldTiedOperandIndices)) {
    unsigned resultIndex = it.index();
    if (llvm::is_contained(excludedResultIndices, resultIndex)) {
      continue; // result removed
    }

    int64_t tiedOperandIndex = it.value();
    if (tiedOperandIndex != IREE::Util::TiedOpInterface::kUntiedIndex) {
      // Check whether this operand is removed. If so, untie. We need to do this
      // before calculating the new operand index given `excludedOperandIndices`
      // contains the old indices.
      if (llvm::is_contained(excludedOperandIndices, tiedOperandIndex)) {
        tiedOperandIndex = IREE::Util::TiedOpInterface::kUntiedIndex;
      }

      // Count up the number of removed operands prior to this one.
      unsigned offset = 0;
      for (unsigned i = 0; i < tiedOperandIndex; ++i) {
        if (i < excludedOperands.size() && excludedOperands[i]) {
          ++offset;
        }
      }

      tiedOperandIndex -= offset;
    }
    tiedOperandIndices.push_back(tiedOperandIndex);
  }
}

//===----------------------------------------------------------------------===//
// IREE::Util::SizeAwareTypeInterface
//===----------------------------------------------------------------------===//

// static
Value SizeAwareTypeInterface::findSizeValue(Value resourceValue, Block *block,
                                            Block::iterator insertionPoint) {
  // See if the value is produced by a size-aware op; we can just ask for the
  // size it has tied. Walking upward is always good as we know any size we find
  // dominates {|block|, |insertionPoint|}.
  SmallVector<Value> worklist;
  worklist.push_back(resourceValue);
  while (!worklist.empty()) {
    auto value = worklist.pop_back_val();
    auto *definingOp = value.getDefiningOp();
    if (!definingOp) {
      continue;
    }
    if (auto sizeAwareOp =
            dyn_cast<IREE::Util::SizeAwareOpInterface>(definingOp)) {
      return sizeAwareOp.getResultSizeFromValue(value);
    }
    if (auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(definingOp)) {
      auto tiedOperand = tiedOp.getTiedResultOperand(value);
      if (tiedOperand) {
        worklist.push_back(tiedOperand);
      }
    }
  }

  // Walk the users to see if any can report the size.
  worklist.push_back(resourceValue);
  while (!worklist.empty()) {
    auto value = worklist.pop_back_val();
    for (auto &use : value.getUses()) {
      if (auto sizeAwareOp =
              dyn_cast<IREE::Util::SizeAwareOpInterface>(use.getOwner())) {
        auto sizeValue = sizeAwareOp.getOperandSize(use.getOperandNumber());
        if (sizeValue) {
          if (isValueUsableForOp(sizeValue, block, insertionPoint)) {
            return sizeValue;
          }
        }
      }
      if (auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(use.getOwner())) {
        worklist.append(tiedOp.getOperandTiedResults(use.getOperandNumber()));
      }
    }
  }

  return {};
}

// static
Value SizeAwareTypeInterface::queryValueSize(Location loc, Value resourceValue,
                                             OpBuilder &builder) {
  auto sizeAwareType =
      dyn_cast<IREE::Util::SizeAwareTypeInterface>(resourceValue.getType());
  if (!sizeAwareType) {
    return {}; // Not a sized type.
  }
  if (!builder.getInsertionPoint().getNodePtr()->isKnownSentinel()) {
    auto sizeValue = sizeAwareType.findSizeValue(
        resourceValue, builder.getBlock(), builder.getInsertionPoint());
    if (sizeValue) {
      return sizeValue; // Found in IR.
    }
  }
  // TODO(benvanik): make this cleaner.
  auto *definingOp = resourceValue.getDefiningOp();
  if (auto sizeAwareOp =
          dyn_cast_if_present<IREE::Util::SizeAwareOpInterface>(definingOp)) {
    return sizeAwareOp.getResultSizeFromValue(resourceValue);
  } else if (auto inferSizeType = dyn_cast<IREE::Util::InferTypeSizeInterface>(
                 resourceValue.getType())) {
    return inferSizeType.inferSizeFromValue(loc, resourceValue, builder);
  }
  return {};
}

//===----------------------------------------------------------------------===//
// IREE::Util::ShapeAware*
//===----------------------------------------------------------------------===//

std::optional<ValueRange> findDynamicDims(Value workValue) {
  // Look up the use-def chain: always safe, as any value we reach dominates
  // {|block|, |insertionPoint|} implicitly.
  while (workValue) {
    auto workOp = workValue.getDefiningOp();
    if (!workOp) {
      break;
    }
    if (auto shapeAwareOp =
            dyn_cast<IREE::Util::ShapeAwareOpInterface>(workOp)) {
      return shapeAwareOp.getResultDynamicDimsFromValue(workValue);
    } else if (auto sizeAwareOp =
                   dyn_cast<IREE::Util::SizeAwareOpInterface>(workOp)) {
      return sizeAwareOp.getResultSizeFromValue(workValue);
    } else if (auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(workOp)) {
      workValue = tiedOp.getTiedResultOperand(workValue);
    } else {
      break;
    }
  }
  return std::nullopt;
}

OpFoldResult findDim(Value workValue, int64_t dim) {
  auto shapedType = cast<ShapedType>(workValue.getType());
  int64_t rank = shapedType.getRank();
  assert(rank > dim && "querying out of range dim");
  int64_t staticSize = shapedType.getDimSize(dim);
  if (ShapedType::isStatic(staticSize)) {
    Builder b(workValue.getContext());
    return b.getIndexAttr(dim);
  }

  // Look up the use-def chain for the dynamic dims of the shaped value.
  auto upwardRange = findDynamicDims(workValue);
  if (!upwardRange.has_value()) {
    return OpFoldResult();
  }

  // Count the number of dynamic dims before the queried dim. This is the index
  // of the queried dim out of the range of dynamic dims.
  int64_t dynamicIndex = llvm::count_if(
      shapedType.getShape().drop_back(rank - dim), ShapedType::isDynamic);
  return upwardRange.value()[dynamicIndex];
}

std::optional<ValueRange> findDynamicDims(Value shapedValue, Block *block,
                                          Block::iterator insertionPoint) {
  // Look up the use-def chain: always safe, as any value we reach dominates
  // {|block|, |insertionPoint|} implicitly.
  auto upwardRange = findDynamicDims(shapedValue);
  if (upwardRange.has_value()) {
    return upwardRange.value();
  }

  // Look down the use-def chain: not safe at some point because we'll move past
  // where {|block|, |insertionPoint|} is dominated. This is often fine for a
  // bit, though, as {|block|, |insertionPoint|} may be a user of |shapedValue|
  // and be able to provide the shape itself.
  for (auto &use : shapedValue.getUses()) {
    if (auto shapeAwareOp = dyn_cast<ShapeAwareOpInterface>(use.getOwner())) {
      auto dynamicDims =
          shapeAwareOp.getOperandDynamicDims(use.getOperandNumber());
      if (llvm::all_of(dynamicDims, [&](Value dim) {
            return isValueUsableForOp(dim, block, insertionPoint);
          })) {
        return dynamicDims;
      }
    } else if (auto sizeAwareOp =
                   dyn_cast<SizeAwareOpInterface>(use.getOwner())) {
      auto size = sizeAwareOp.getOperandSize(use.getOperandNumber());
      if (isValueUsableForOp(size, block, insertionPoint)) {
        return size;
      }
    }
  }

  return std::nullopt;
}

ValueRange findDynamicDimsInList(unsigned idx, ValueRange values,
                                 ValueRange dynamicDims) {
  // Get the total number of dynamic dimensions for the value. Note that this
  // may be 0 if all dimensions are static.
  auto value = values[idx];
  unsigned dynamicDimCount = 0;
  if (auto shapedType = dyn_cast<ShapedType>(value.getType())) {
    dynamicDimCount = shapedType.getNumDynamicDims();
  } else if (isa<IREE::Util::SizeAwareTypeInterface>(value.getType())) {
    dynamicDimCount = 1;
  }
  if (!dynamicDimCount) {
    return ValueRange{};
  }

  // Find where the dynamic dims start in the flattened list.
  unsigned offset = 0;
  for (unsigned i = 0; i < idx; ++i) {
    auto prefixValueType = values[i].getType();
    if (auto shapedType = dyn_cast<ShapedType>(prefixValueType)) {
      offset += shapedType.getNumDynamicDims();
    } else if (isa<IREE::Util::SizeAwareTypeInterface>(prefixValueType)) {
      offset += 1; // fixed 1 dynamic dim
    }
  }

  // Return the subrange of dynamic dims for the value being queried.
  return dynamicDims.slice(offset, dynamicDimCount);
}

Value findValueSizeInList(unsigned idx, ValueRange values,
                          ValueRange dynamicDims) {
  auto dims = findDynamicDimsInList(idx, values, dynamicDims);
  assert(dims.size() == 1 && "expected exactly one dynamic dimension");
  return dims.front();
}

SmallVector<Value> buildDynamicDimsForValue(Location loc, Value value,
                                            OpBuilder &builder) {
  auto valueType = dyn_cast<ShapedType>(value.getType());
  if (!valueType) {
    mlir::emitError(loc) << "cannot construct shape for non shaped value: "
                         << value.getType();
    return {};
  }

  // Early-exit if all dimensions are static.
  if (valueType.hasStaticShape()) {
    return {};
  }

  // Try the fast-path of scanning for the dynamic dims that exist in the IR
  // already. For shape-aware ops this is free as the dynamic dim SSA values are
  // always available.
  auto foundDynamicDims = IREE::Util::findDynamicDims(
      value, builder.getBlock(), builder.getInsertionPoint());
  if (foundDynamicDims.has_value()) {
    return llvm::to_vector(foundDynamicDims.value());
  }

  // Slower path that materializes the entire shape for a result. Some
  // implementations may only support this (vs the fast find above).
  if (auto shapeAwareOp =
          dyn_cast_if_present<IREE::Util::ShapeAwareOpInterface>(
              value.getDefiningOp())) {
    return shapeAwareOp.buildResultValueShape(value, builder);
  }

  // TODO(benvanik): add support for ReifyRankedShapedTypeOpInterface;
  // unfortunately it is for all results and all dimensions so a lot of unneeded
  // IR will be inserted.

  // Fallback to inserting dim ops that can be resolved via normal upstream
  // mechanisms. Depending on where this is called from within the parent
  // pipeline these ops may not be desirable, but that's what the
  // ShapeAwareOpInterface is for.
  SmallVector<Value> dynamicDims;
  for (unsigned i = 0; i < valueType.getRank(); ++i) {
    if (valueType.isDynamicDim(i)) {
      dynamicDims.push_back(builder.createOrFold<tensor::DimOp>(loc, value, i));
    }
  }
  return dynamicDims;
}

SmallVector<Value> buildDynamicDimsForValues(Location loc, ValueRange values,
                                             OpBuilder &builder) {
  SmallVector<Value> dynamicDims;
  for (auto value : values) {
    dynamicDims.append(buildDynamicDimsForValue(loc, value, builder));
  }
  return dynamicDims;
}

static SmallVector<Value> buildShape(Location loc, ShapedType type,
                                     ValueRange dynamicDims,
                                     OpBuilder &builder) {
  SmallVector<Value> dims;
  dims.reserve(type.getRank());
  unsigned dynamicIdx = 0;
  for (unsigned i = 0; i < type.getRank(); ++i) {
    int64_t dim = type.getDimSize(i);
    if (ShapedType::isDynamic(dim)) {
      dims.push_back(dynamicDims[dynamicIdx++]);
    } else {
      dims.push_back(arith::ConstantIndexOp::create(builder, loc, dim));
    }
  }
  return dims;
}

SmallVector<Value> buildOperandShape(IREE::Util::ShapeAwareOpInterface op,
                                     unsigned operandIdx, OpBuilder &builder) {
  auto operand = op->getOperand(operandIdx);
  auto type = cast<ShapedType>(operand.getType());
  auto dynamicDims = op.getOperandDynamicDims(operandIdx);
  return buildShape(op.getLoc(), type, dynamicDims, builder);
}

SmallVector<Value> buildResultShape(IREE::Util::ShapeAwareOpInterface op,
                                    unsigned resultIdx, OpBuilder &builder) {
  auto result = op->getResult(resultIdx);
  auto type = cast<ShapedType>(result.getType());
  auto dynamicDims = op.getResultDynamicDims(resultIdx);
  return buildShape(op.getLoc(), type, dynamicDims, builder);
}

} // namespace mlir::iree_compiler::IREE::Util

//===----------------------------------------------------------------------===//
// IREE::Util::UtilDialect
//===----------------------------------------------------------------------===//

// clang-format off: must be included after all LLVM/MLIR headers.
#define GET_TYPEDEF_CLASSES
#include "iree/compiler/Dialect/Util/IR/UtilTypes.cpp.inc" // IWYU pragma: keep
// clang-format on

namespace mlir::iree_compiler::IREE::Util {

// At the end so it can use functions above:
#include "iree/compiler/Dialect/Util/IR/UtilOpInterfaces.cpp.inc"
#include "iree/compiler/Dialect/Util/IR/UtilTypeInterfaces.cpp.inc"

void UtilDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "iree/compiler/Dialect/Util/IR/UtilTypes.cpp.inc" // IWYU pragma: keep
      >();
}

} // namespace mlir::iree_compiler::IREE::Util
