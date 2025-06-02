// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-fuse-and-hoist-parallel-loops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUFUSEANDHOISTPARALLELLOOPSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

using namespace IREE::GPU;

namespace {
struct GPUFuseAndHoistParallelLoopsPass final
    : impl::GPUFuseAndHoistParallelLoopsPassBase<
          GPUFuseAndHoistParallelLoopsPass> {
  void runOnOperation() override;
};
} // namespace

static std::optional<int64_t> getStaticForallTripCount(scf::ForallOp forall) {
  // TODO: Handle non-normalized loops.
  if (!forall.isNormalized()) {
    return std::nullopt;
  }
  int64_t tripCount = 1;
  for (OpFoldResult ub : forall.getMixedUpperBound()) {
    std::optional<int64_t> maybeConstantUb = getConstantIntValue(ub);
    if (!maybeConstantUb) {
      return std::nullopt;
    }
    tripCount *= *maybeConstantUb;
  }
  return tripCount;
}

static bool forallTripCountMatchesWorkgroupSize(scf::ForallOp forallOp,
                                                int64_t flatWorkgroupSize) {
  std::optional<int64_t> maybeTripCount = getStaticForallTripCount(forallOp);
  if (!maybeTripCount) {
    return false;
  }

  // For lane mapped foralls we need to verify that it is contained within
  // a parent warp mapped op that combines to match the workggroup size.
  if (forallOpHasMappingType<IREE::GPU::LaneIdAttr>(forallOp)) {
    auto parentForall = forallOp->getParentOfType<scf::ForallOp>();
    if (!parentForall ||
        !forallOpHasMappingType<gpu::GPUWarpMappingAttr>(parentForall)) {
      return false;
    }

    std::optional<int64_t> maybeParentTripCount =
        getStaticForallTripCount(parentForall);
    if (!maybeParentTripCount) {
      return false;
    }

    return *maybeParentTripCount * *maybeTripCount == flatWorkgroupSize;
  }

  // All other loops must be mapped to threads to compare.
  if (!forallOpHasMappingType<gpu::GPUThreadMappingAttr>(forallOp)) {
    return false;
  }

  return *maybeTripCount == flatWorkgroupSize;
}
struct FuseForalls final : OpRewritePattern<scf::ForallOp> {
  using OpRewritePattern::OpRewritePattern;
  FuseForalls(MLIRContext *ctx, int64_t flatWorkgroupSize, PatternBenefit b = 1)
      : OpRewritePattern<scf::ForallOp>(ctx, b),
        flatWorkgroupSize(flatWorkgroupSize) {}
  LogicalResult matchAndRewrite(scf::ForallOp producerForall,
                                PatternRewriter &rewriter) const override {
    if (!producerForall->hasOneUse()) {
      return rewriter.notifyMatchFailure(producerForall,
                                         "multi-use producer forall");
    }

    SmallVector<Operation *> consumerChain;
    Operation *currProducer = *producerForall->user_begin();
    while (currProducer && currProducer->hasOneUse()) {
      consumerChain.push_back(currProducer);
      if (!isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(currProducer)) {
        break;
      }
      currProducer = *currProducer->user_begin();
    }

    auto consumerForall = currProducer->getParentOfType<scf::ForallOp>();
    if (!consumerForall || !forallTripCountMatchesWorkgroupSize(
                               consumerForall, flatWorkgroupSize)) {
      return rewriter.notifyMatchFailure(
          producerForall,
          "no consumer forall with trip count matching workgroup size");
    }

    // TODO: Allow extracting multiple uses within the same consumer loop. Still
    // single producer single consumer loop, but multiple uses within the
    // consumer.
    if (!producerForall->hasOneUse()) {
      return failure();
    }

    return fuseForallIntoConsumer(rewriter, producerForall, consumerForall,
                                  consumerChain);
  }

private:
  int64_t flatWorkgroupSize;
};

struct FuseTilableDestinationProducers final : OpRewritePattern<scf::ForallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForallOp forallOp,
                                PatternRewriter &rewriter) const override {
    TilingInterface tileableProducer;
    tensor::ExtractSliceOp sliceOp;
    for (auto iterArg : forallOp.getRegionIterArgs()) {
      for (auto user : iterArg.getUsers()) {
        sliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
        if (sliceOp) {
          break;
        }
      }
      if (!sliceOp) {
        continue;
      }
      tileableProducer = forallOp.getTiedLoopInit(iterArg)
                             ->get()
                             .getDefiningOp<TilingInterface>();
      if (tileableProducer) {
        break;
      }
    }
    if (!tileableProducer) {
      return failure();
    }

    SmallVector<LoopLikeOpInterface> loops = {forallOp};
    rewriter.startOpModification(forallOp);
    std::optional<scf::SCFFuseProducerOfSliceResult> fusionResult =
        mlir::scf::tileAndFuseProducerOfSlice(rewriter, sliceOp, loops);
    if (!fusionResult) {
      return failure();
    }
    rewriter.finalizeOpModification(forallOp);
    return success();
  }
};

struct FuseUnitLoopDestination final : OpRewritePattern<scf::ForallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForallOp forallOp,
                                PatternRewriter &rewriter) const override {
    std::optional<int64_t> maybeTripCount = getStaticForallTripCount(forallOp);
    if (!maybeTripCount || *maybeTripCount != 1) {
      return rewriter.notifyMatchFailure(forallOp,
                                         "not a unit trip count loop");
    }
    DestinationStyleOpInterface dpsProducer;
    BlockArgument bodyArg;
    Value dpsResult;
    for (auto iterArg : forallOp.getRegionIterArgs()) {
      dpsResult = forallOp.getTiedLoopInit(iterArg)->get();
      bodyArg = iterArg;
      dpsProducer = dpsResult.getDefiningOp<DestinationStyleOpInterface>();
      if (dpsProducer) {
        break;
      }
    }
    if (!dpsProducer || !dpsProducer->hasOneUse()) {
      return rewriter.notifyMatchFailure(forallOp,
                                         "no single use DPS producer");
    }

    Operation *parallelInsert = nullptr;
    for (auto user : bodyArg.getUsers()) {
      if (isa<tensor::ParallelInsertSliceOp>(user)) {
        // This should be illegal but check anyway.
        if (parallelInsert) {
          return rewriter.notifyMatchFailure(forallOp, "multiple insert users");
        }
        parallelInsert = user;
      }
    }
    if (!parallelInsert) {
      return rewriter.notifyMatchFailure(
          forallOp, "destination not used by a parallel insert");
    }

    rewriter.startOpModification(forallOp);
    // Move the producer into the body of the forall loop.
    rewriter.moveOpBefore(dpsProducer, forallOp.getBody(),
                          forallOp.getBody()->begin());

    // Replace all uses of the region iter arg with the moved dps op.
    rewriter.replaceAllUsesExcept(bodyArg, dpsResult, parallelInsert);

    // Set the init operand of the forall op to the init operand of the
    // producer.
    int64_t dpsInitIndex = cast<OpResult>(dpsResult).getResultNumber();
    forallOp->setOperand(forallOp.getTiedOpOperand(bodyArg)->getOperandNumber(),
                         dpsProducer.getDpsInitOperand(dpsInitIndex)->get());

    // Finally replace the init operand of the moved producer with the region
    // iter arg.
    dpsProducer.setDpsInitOperand(dpsInitIndex, bodyArg);
    rewriter.finalizeOpModification(forallOp);
    return success();
  }
};

struct FuseTilableSliceProducers final
    : OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    if (sliceOp->use_empty()) {
      return failure();
    }
    auto tilableProducer = sliceOp.getSource().getDefiningOp<TilingInterface>();
    if (!tilableProducer) {
      return failure();
    }

    auto parentForall = sliceOp->getParentOfType<scf::ForallOp>();
    if (!parentForall) {
      return failure();
    }

    auto producerParent = tilableProducer->getParentOfType<scf::ForallOp>();
    if (producerParent && producerParent == parentForall) {
      return failure();
    }

    SmallVector<LoopLikeOpInterface> loops = {parentForall};
    std::optional<scf::SCFFuseProducerOfSliceResult> fusionResult =
        mlir::scf::tileAndFuseProducerOfSlice(rewriter, sliceOp, loops);
    if (!fusionResult) {
      return failure();
    }
    return success();
  }
};

struct FuseTilableForallConsumers final
    : OpInterfaceRewritePattern<TilingInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(TilingInterface tilableOp,
                                PatternRewriter &rewriter) const override {
    // Currently consumer fusion requires DPS, and we don't want to fuse through
    // inits anyway.
    auto dpsOp = dyn_cast<DestinationStyleOpInterface>(*tilableOp);
    if (!dpsOp) {
      return failure();
    }
    if (isa<IREE::LinalgExt::MapScatterOp>(tilableOp)) {
      return failure();
    }

    tensor::ParallelInsertSliceOp producerSlice;
    LoopLikeOpInterface sliceOwner;
    Value fusionOperand;
    for (auto operand : dpsOp.getDpsInputs()) {
      auto forallProducer = operand.getDefiningOp<scf::ForallOp>();
      if (!forallProducer) {
        continue;
      }
      Value iterArg = forallProducer.getTiedBlockArgument(
          forallProducer.getTiedOpOperand(cast<OpResult>(operand)));

      for (auto user : iterArg.getUsers()) {
        auto sliceOp = dyn_cast<tensor::ParallelInsertSliceOp>(user);
        if (sliceOp && sliceOp.getDest() == iterArg) {
          producerSlice = sliceOp;
          sliceOwner = forallProducer;
          fusionOperand = operand;
          break;
        }
      }
      if (producerSlice) {
        break;
      }
    }

    if (!producerSlice) {
      return rewriter.notifyMatchFailure(tilableOp,
                                         "no scf.forall producer to fuse into");
    }

    for (auto operand : tilableOp->getOperands()) {
      if (operand != fusionOperand && operand.getDefiningOp() == sliceOwner) {
        return rewriter.notifyMatchFailure(tilableOp,
                                           "unimplemented: Cannot fuse op with "
                                           "multiple uses of producer loop");
      }
    }

    FailureOr<scf::SCFFuseConsumerOfSliceResult> fuseConsumerResults =
        scf::tileAndFuseConsumerOfSlice(rewriter, producerSlice, {sliceOwner});
    if (failed(fuseConsumerResults)) {
      return failure();
    }
    return success();
  }
};

struct FuseCollapseShapeConsumers final
    : OpRewritePattern<tensor::CollapseShapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::CollapseShapeOp collapseOp,
                                PatternRewriter &rewriter) const override {
    auto forallOp = collapseOp.getSrc().getDefiningOp<scf::ForallOp>();
    if (!forallOp) {
      return rewriter.notifyMatchFailure(collapseOp, "No forall op producer");
    }

    if (failed(fuseCollapseShapeIntoProducerForall(rewriter, forallOp,
                                                   collapseOp))) {
      return failure();
    }
    return success();
  }
};

struct FuseExtractSliceConsumers final
    : OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractSliceOp,
                                PatternRewriter &rewriter) const override {
    // Find the scf::ForallOp producer, and get the corresponding
    // tensor::ParallelInsertSliceOp.
    auto forallOp = extractSliceOp.getSource().getDefiningOp<scf::ForallOp>();
    if (!forallOp) {
      return rewriter.notifyMatchFailure(extractSliceOp,
                                         "No forall op producer");
    }

    if (failed(fuseExtractSliceIntoProducerForall(rewriter, forallOp,
                                                  extractSliceOp))) {
      return failure();
    }
    return success();
  }
};

template <typename MapScatterOpTy>
static FailureOr<Value>
fuseMapScatterIntoProducerForall(RewriterBase &rewriter, scf::ForallOp forallOp,
                                 MapScatterOpTy mapScatterOp) {
  auto forallResult = cast<OpResult>(mapScatterOp.getInput());
  auto parallelIterationOp = cast<ParallelIterationOpInterface>(*forallOp);
  SmallVector<Operation *> updatingOps =
      parallelIterationOp.getUpdatingOps(forallResult);
  for (Operation *op : updatingOps) {
    auto parallelInsertSliceOp = dyn_cast<tensor::ParallelInsertSliceOp>(op);
    if (!parallelInsertSliceOp) {
      return rewriter.notifyMatchFailure(
          op, "updating ops are not all parallel_insert_slice ops");
    }
    if (!areAllConstantIntValue(parallelInsertSliceOp.getMixedStrides(), 1)) {
      return rewriter.notifyMatchFailure(
          op, "parallel_insert_slice op has non-unit strides");
    }
  }
  // The tied output argument will be replaced with the map_scatter output, and
  // the only users of the new block argument will be the ParallelMapScatterOp.
  // Replace all uses now, and save the tied output block arg to use as the
  // ParallelMapScatterOp output.
  int64_t resultIdx = forallResult.getResultNumber();
  BlockArgument resultTiedArg = forallOp.getRegionOutArgs()[resultIdx];
  rewriter.replaceAllUsesWith(resultTiedArg, forallOp.getOutputs()[resultIdx]);
  for (Operation *op : updatingOps) {
    // Start by getting the tiled implementation of the map_scatter op. Then
    // use its transformation body to create a ParallelMapScatterOp to replace
    // it.
    auto parallelInsertSliceOp = cast<tensor::ParallelInsertSliceOp>(op);
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(parallelInsertSliceOp);
    Location loc = parallelInsertSliceOp.getLoc();
    SmallVector<OpFoldResult> offsets = parallelInsertSliceOp.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = parallelInsertSliceOp.getMixedSizes();
    IREE::LinalgExt::ParallelMapScatterOp parallelMapScatterOp =
        IREE::LinalgExt::ParallelMapScatterOp::createWithTransformationRegion(
            rewriter, loc, &mapScatterOp.getTransformationRegion(),
            parallelInsertSliceOp.getSource(), resultTiedArg);
    auto indexTransformBuilder =
        [&](ArrayRef<BlockArgument> srcIndices) -> SmallVector<Value> {
      SmallVector<OpFoldResult> offsetIndices;
      auto addMap = AffineMap::get(
          2, 0, {rewriter.getAffineDimExpr(0) + rewriter.getAffineDimExpr(1)});
      for (auto [srcIdx, offset] : llvm::zip_equal(srcIndices, offsets)) {
        offsetIndices.push_back(affine::makeComposedFoldedAffineApply(
            rewriter, loc, addMap, {OpFoldResult(srcIdx), offset}));
      }
      return getValueOrCreateConstantIndexOp(rewriter, loc, offsetIndices);
    };
    parallelMapScatterOp.insertTransformationAtStart(
        rewriter, indexTransformBuilder, offsets.size());
    rewriter.eraseOp(parallelInsertSliceOp);
  }

  // Clone the forall op with the extracted init operand to replace the
  // original forall op.
  Location loc = forallOp.getLoc();
  rewriter.setInsertionPoint(forallOp);
  SmallVector<Value> newForallOutputs(forallOp.getOutputs());
  newForallOutputs[resultIdx] = mapScatterOp.getOutput();

  scf::ForallOp newForallOp = rewriter.create<scf::ForallOp>(
      loc, forallOp.getMixedLowerBound(), forallOp.getMixedUpperBound(),
      forallOp.getMixedStep(), newForallOutputs, forallOp.getMappingAttr());

  SmallVector<Value> argReplacements(newForallOp.getInductionVars());
  argReplacements.append(newForallOp.getRegionIterArgs().begin(),
                         newForallOp.getRegionIterArgs().end());
  newForallOp.getTerminator()->erase();
  rewriter.mergeBlocks(forallOp.getBody(), newForallOp.getBody(),
                       argReplacements);

  rewriter.replaceOp(forallOp, newForallOp);
  if (mapScatterOp->getNumResults() > 0) {
    rewriter.replaceOp(mapScatterOp, newForallOp.getResult(resultIdx));
  }
  return newForallOp.getResult(resultIdx);
}

struct FuseMapScatterConsumer final
    : OpRewritePattern<IREE::LinalgExt::MapScatterOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::LinalgExt::MapScatterOp mapScatterOp,
                                PatternRewriter &rewriter) const override {
    auto forallOp = mapScatterOp.getInput().getDefiningOp<scf::ForallOp>();
    if (!forallOp) {
      return rewriter.notifyMatchFailure(mapScatterOp, "no forall op producer");
    }
    if (!mapScatterOp.getInput().hasOneUse()) {
      return rewriter.notifyMatchFailure(mapScatterOp,
                                         "map_scatter input has multiple uses");
    }

    if (failed(fuseMapScatterIntoProducerForall(rewriter, forallOp,
                                                mapScatterOp))) {
      return failure();
    }
    return success();
  }
};

struct FuseParallelMapScatterConsumer final
    : OpRewritePattern<IREE::LinalgExt::ParallelMapScatterOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult
  matchAndRewrite(IREE::LinalgExt::ParallelMapScatterOp parallelMapScatterOp,
                  PatternRewriter &rewriter) const override {
    auto forallOp =
        parallelMapScatterOp.getInput().getDefiningOp<scf::ForallOp>();
    if (!forallOp) {
      return rewriter.notifyMatchFailure(parallelMapScatterOp,
                                         "no forall op producer");
    }
    if (!parallelMapScatterOp.getInput().hasOneUse()) {
      return rewriter.notifyMatchFailure(parallelMapScatterOp,
                                         "map_scatter input has multiple uses");
    }

    FailureOr<Value> scatterResult = fuseMapScatterIntoProducerForall(
        rewriter, forallOp, parallelMapScatterOp);
    if (failed(scatterResult)) {
      return failure();
    }

    int64_t sliceRank = parallelMapScatterOp.getOutputRank();
    Location loc = parallelMapScatterOp.getLoc();
    SmallVector<OpFoldResult> offsets(sliceRank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(sliceRank, rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes =
        tensor::getMixedSizes(rewriter, loc, parallelMapScatterOp.getOutput());
    rewriter.setInsertionPoint(parallelMapScatterOp);
    rewriter.replaceOpWithNewOp<tensor::ParallelInsertSliceOp>(
        parallelMapScatterOp, *scatterResult, parallelMapScatterOp.getOutput(),
        offsets, sizes, strides);
    return success();
  }
};

void GPUFuseAndHoistParallelLoopsPass::runOnOperation() {
  MLIRContext *context = &getContext();

  FunctionOpInterface funcOp = getOperation();

  // Try to get the flat workgroup size if possible.
  std::optional<int64_t> maybeFlatWorkgroupSize = std::nullopt;
  if (std::optional<SmallVector<int64_t>> workgroupSize =
          getWorkgroupSize(funcOp)) {
    maybeFlatWorkgroupSize =
        std::accumulate(workgroupSize->begin(), workgroupSize->end(), 1,
                        std::multiplies<int64_t>());
  }

  // First run the hoisting and fusion patterns.
  {
    RewritePatternSet patterns(context);
    // These two patterns are run to a fixed point, allowing fusion within
    // potentially nested loops, hoisting from said loops, and continued fusion.
    if (maybeFlatWorkgroupSize) {
      // Forall fusion requires knowing the workgroup size to verify the fusion
      // is valid. Without validation we risk putting barriers inside
      // conditioned regions (e.g. scf.if/for).
      patterns.add<FuseForalls>(context, *maybeFlatWorkgroupSize,
                                /*benefit=*/1);
    }
    patterns.add<FuseTilableForallConsumers>(context);
    // patterns.add<FuseMapScatterConsumer>(context);
    populateForallLoopHoistingPattern(patterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  LDBG("After fusing and hoisting loops\n" << funcOp);

  // After hoisting parallel loops, try to fuse in any newly revealed consumers
  // and destinations.
  // TODO: Move the consumer fusion pattern to an explicit worklist rather than
  // using the GreedyPatternRewriter.
  {
    RewritePatternSet patterns(context);
    patterns.add<FuseTilableDestinationProducers>(context);
    patterns.add<FuseUnitLoopDestination>(context);
    patterns.add<FuseTilableForallConsumers>(context);
    patterns.add<FuseCollapseShapeConsumers>(context);
    patterns.add<FuseExtractSliceConsumers>(context);
    patterns.add<FuseMapScatterConsumer>(context);
    patterns.add<FuseParallelMapScatterConsumer>(context);
    populateSwapExtractWithExpandPattern(patterns);
    tensor::populateFoldTensorEmptyPatterns(patterns);
    scf::ForallOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  LDBG("After fusing new consumers\n" << funcOp);

  // Finally try to do any new producer fusions.
  {
    RewritePatternSet patterns(context);
    patterns.add<FuseTilableDestinationProducers>(context);
    patterns.add<FuseTilableSliceProducers>(context);
    tensor::populateFoldTensorEmptyPatterns(patterns);
    scf::ForallOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  LDBG("After fusing new producers\n" << funcOp);
}

} // namespace mlir::iree_compiler
