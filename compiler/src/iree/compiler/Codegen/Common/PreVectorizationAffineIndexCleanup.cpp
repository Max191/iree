#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"

#define DEBUG_TYPE "iree-codegen-pre-vectorization-affine-index-cleanup"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_PREVECTORIZATIONAFFINEINDEXCLEANUPPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

struct PreVectorizationAffineIndexCleanupPass final
    : impl::PreVectorizationAffineIndexCleanupPassBase<
          PreVectorizationAffineIndexCleanupPass> {
  using Base::Base;

  LogicalResult initialize(MLIRContext *context) override {
    configureCodegenGreedyRewrite(config);

    RewritePatternSet owningPatterns(context);
    populateCodegenCanonicalizationPatterns(context, owningPatterns);
    affine::populateSimplifyAffineWithBoundsPatterns(owningPatterns);

    patterns =
        std::make_shared<FrozenRewritePatternSet>(std::move(owningPatterns));
    return success();
  }

  void runOnOperation() override {
    if (failed(applyPatternsWithConfigTracking(
            getOperation(), *patterns, config, this->testConvergence,
            "Pre-vectorization affine index cleanup failed to converge"))) {
      return signalPassFailure();
    }
  }

  GreedyRewriteConfig config;
  std::shared_ptr<const FrozenRewritePatternSet> patterns;
};

} // namespace
} // namespace mlir::iree_compiler
