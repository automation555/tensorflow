/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/transforms/remapper/pass.h"

#include <memory>
#include <utility>

#include "mlir/IR/BuiltinTypes.h"                        // from @llvm-project
#include "mlir/IR/OperationSupport.h"                    // from @llvm-project
#include "mlir/IR/PatternMatch.h"                        // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/tf_op_wrapper.h"
#include "tensorflow/core/transforms/pass_detail.h"
#include "tensorflow/core/transforms/remapper/remapping_helper.h"
#include "tensorflow/core/transforms/utils/utils.h"
#include "tensorflow/core/util/util.h"

namespace mlir {
namespace tfg {

using namespace remapping;

// Convert Sigmoid+Mul to Swish
// Mul(x, Sigmoid(x)) --> _MklSwish(x)
class MatchMulSigmoid : public RewritePattern {
 public:
  explicit MatchMulSigmoid(MLIRContext *context)
      : RewritePattern("tfg.Mul", PatternBenefit(/*benefit=*/1), context),
        sigmoid_name_("tfg.Sigmoid", context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    TypeAttr dtype_attr = op->getAttrOfType<TypeAttr>("T");
    if (!dtype_attr.getValue().isa<Float32Type>() &&
        !dtype_attr.getValue().isa<BFloat16Type>())
      return failure();

    if (!util::NodeIsOnCpu(op)) return failure();

    TFOp mul_wrapper(op);

    Value sigmoid = op->getOperand(0);
    Value x = op->getOperand(1);

    auto sigmoidOperandEqToX = [&](Value sigmoid, Value x) {
      Operation *op = sigmoid.getDefiningOp();
      return op && op->getName() == sigmoid_name_ && op->getOperand(0) == x;
    };

    if (!sigmoidOperandEqToX(sigmoid, x)) {
      // The operands are commutative and it may have both sigmoid operands.
      // Swap them then check it again.
      std::swap(sigmoid, x);
      if (!sigmoidOperandEqToX(sigmoid, x)) return failure();
    }

    SmallVector<Value> operands;
    // Set up non-control operand.
    operands.push_back(x);
    // Control operands come after regular operands.
    llvm::append_range(operands, mul_wrapper.getControlOperands());

    Operation *new_op =
        rewriter.create(op->getLoc(), rewriter.getStringAttr("tfg._MklSwish"),
                        operands, op->getResultTypes(), op->getAttrs());
    rewriter.replaceOp(op, new_op->getResults());

    return success();
  }

 private:
  // This is used to eliminate the string comparison by caching the handle of an
  // operation name.
  OperationName sigmoid_name_;
};

static FailureOr<TFOp> CreateContractionWithBiasAddOp(PatternRewriter &rewriter,
                                                      OpPropertyHelper &helper,
                                                      Operation *contraction_op,
                                                      Operation *bias_add_op) {
  // Get all operands for fused op
  Value input = contraction_op->getOperand(0);
  Value filter = contraction_op->getOperand(1);
  Value bias = bias_add_op->getOperand(1);

  SmallVector<Value> operands;
  operands.push_back(input);
  operands.push_back(filter);
  operands.push_back(bias);

  // Get the contraction type, MatMul/Conv2D/DepthwiseConv2dNative etc.
  std::string fused_contraction_type;
  if (helper.getDialect()->IsConv2D(contraction_op)) {
    fused_contraction_type = "tfg._FusedConv2D";
  } else if (helper.getDialect()->IsMatMul(contraction_op)) {
    fused_contraction_type = "tfg._FuseMatMul";
  } else if (helper.getDialect()->IsDepthwiseConv2dNative(contraction_op)) {
    fused_contraction_type = "tfg._FusedDepthwiseConv2dNative";
  } else {
    // TODO(intel-tf): Silently return. Add some loggin info that fusion is
    // not supported.
    return failure();
  }

  // matchAndRewrite function will set the appropriate location.
  Operation *new_op =
      rewriter.create(UnknownLoc::get(contraction_op->getContext()),
                      rewriter.getStringAttr(fused_contraction_type), operands,
                      bias_add_op->getResultTypes(), {});

  // Fill in attributes
  mlir::NamedAttrList fused_attrs(contraction_op->getAttrDictionary());
  fused_attrs.set("fused_ops", rewriter.getStrArrayAttr({"BiasAdd"}));
  fused_attrs.set("num_args", rewriter.getI32IntegerAttr(1));
  // Set default values for epsilon and leakyrelu_alpha
  fused_attrs.set("epsilon", rewriter.getF32FloatAttr(0.0001));
  fused_attrs.set("leakyrelu_alpha", rewriter.getF32FloatAttr(0.2));
  new_op->setAttrs(fused_attrs.getDictionary(new_op->getContext()));
  return TFOp(new_op);
}

// Contraction+BiasAdd
class RewriteContractionBiasAdd : public RewritePattern {
 public:
  explicit RewriteContractionBiasAdd(MLIRContext *context,
                                     OpPropertyHelper &helper)
      : RewritePattern("tfg.BiasAdd", PatternBenefit(/*benefit=*/1), context),
        helper_(helper) {}

  bool matchPattern(Operation *op, ContractionWithBiasAdd &pattern) const {
    // Not allowing control flow on BiasAdd
    if (helper_.HasControlFaninOrFanOut(op)) return false;
    // Contraction Op
    Operation *contraction_op = op->getOperand(0).getDefiningOp();
    if (!helper_.IsContraction(contraction_op) ||
        helper_.HasControlFaninOrFanOut(contraction_op) ||
        !helper_.HaveSameDataType(op, contraction_op) ||
        !helper_.HasAtMostOneFanoutAtPort0(contraction_op))
      return false;

    pattern.contraction = contraction_op;
    pattern.bias_add = op;
    return true;
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    ContractionWithBiasAdd pattern;
    if (!matchPattern(op, pattern)) return failure();
    if (!(helper_.IsCpuCompatible(pattern.contraction) ||
          helper_.IsGpuCompatible(pattern)))
      return failure();
    FailureOr<TFOp> fused_op = CreateContractionWithBiasAddOp(
        rewriter, helper_, pattern.contraction, pattern.bias_add);
    if (failed(fused_op)) return failure();
    fused_op->setName(TFOp(op).nameAttr());
    (*fused_op)->setLoc(op->getLoc());
    rewriter.replaceOp(op, (*fused_op)->getResults());
    return success();
  }

 protected:
  OpPropertyHelper &helper_;
  TFGraphDialect *dialect_;
};

// Contraction+BiasAdd+Activation
template <Activation activation>
class RewriteContractionBiasAddActivation : public RewritePattern {
 public:
  explicit RewriteContractionBiasAddActivation(MLIRContext *context,
                                               OpPropertyHelper &helper)
      : RewritePattern(GetTFGActivation(activation),
                       PatternBenefit(/*benefit=*/1), context),
        helper_(helper),
        dialect_(helper.getDialect()) {}

  bool matchPattern(Operation *op, ContractionWithBiasAddAndActivation &pattern,
                    RewriteContractionBiasAdd &rewrite_base,
                    ContractionWithBiasAdd &base_pattern) const {
    // Although template instantiation gurantuees that only valid activation op
    // sanity check is added.
    if (dialect_->IsNoOp(op)) return false;
    if (helper_.HasControlFaninOrFanOut(op)) return false;
    Operation *bias_add_op = op->getOperand(0).getDefiningOp();
    if (!dialect_->IsBiasAdd(bias_add_op) ||
        !helper_.HaveSameDataType(op, bias_add_op) ||
        !helper_.HasAtMostOneFanoutAtPort0(bias_add_op))
      return false;
    if (!rewrite_base.matchPattern(bias_add_op, base_pattern)) return false;
    pattern.contraction = base_pattern.contraction;
    pattern.bias_add = base_pattern.bias_add;
    pattern.activation = op;
    return true;
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    ContractionWithBiasAddAndActivation pattern;
    RewriteContractionBiasAdd rewrite_base(op->getContext(), helper_);
    ContractionWithBiasAdd base_pattern;
    if (!matchPattern(op, pattern, rewrite_base, base_pattern))
      return failure();
    if (!(helper_.IsCpuCompatible(pattern.contraction) ||
          helper_.IsGpuCompatible(pattern)))
      return failure();
    Operation *&contraction_op = pattern.contraction;
    Operation *&bias_add_op = pattern.bias_add;
    Operation *&activation_op = pattern.activation;
    FailureOr<TFOp> fused_op = CreateContractionWithBiasAddOp(
        rewriter, helper_, contraction_op, bias_add_op);
    if (failed(fused_op)) return failure();
    const StringRef activation_type = activation_op->getName().stripDialect();
    (*fused_op)->setAttr(
        "fused_ops", rewriter.getStrArrayAttr({"BiasAdd", activation_type}));
    if (dialect_->IsLeakyRelu(activation_op))
      (*fused_op)->setAttr("leakyrelu_alpha", activation_op->getAttr("alpha"));
    fused_op->setName(TFOp(op).nameAttr());
    (*fused_op)->setLoc(op->getLoc());
    rewriter.replaceOp(op, (*fused_op)->getResults());
    return success();
  }

 protected:
  OpPropertyHelper &helper_;
  TFGraphDialect *dialect_;
};

template <template <Activation> class PatternT, Activation... activations,
          typename... Args>
static void InsertActivationFusionPatterns(RewritePatternSet &patterns,
                                           Args &&...args) {
  patterns.insert<PatternT<activations>...>(std::forward<Args>(args)...);
}

class Remapper : public RemapperBase<Remapper> {
 public:
  Remapper() = default;
  explicit Remapper(bool enable_mkl_patterns, bool xla_auto_clustering) {
    enable_mkl_patterns_ = enable_mkl_patterns;
    xla_auto_clustering_ = xla_auto_clustering;
  }

  LogicalResult initialize(MLIRContext *context) override {
    helper_ = std::make_shared<OpPropertyHelper>(
        context->getOrLoadDialect<TFGraphDialect>(), tensorflow::IsMKLEnabled(),
        xla_auto_clustering_);
    RewritePatternSet patterns(context);
    populateRemapperPatterns(context, patterns);
    final_patterns_ = std::move(patterns);
    return success();
  }

  void runOnOperation() override;

 private:
  void populateRemapperPatterns(MLIRContext *context,
                                RewritePatternSet &patterns) {
    if (enable_mkl_patterns_) patterns.insert<MatchMulSigmoid>(context);
    patterns.insert<RewriteContractionBiasAdd>(context, *helper_);
    InsertActivationFusionPatterns<RewriteContractionBiasAddActivation,
                                   Activation::Relu, Activation::Relu6,
                                   Activation::Elu, Activation::LeakyRelu,
                                   Activation::Sigmoid>(patterns, context,
                                                        *helper_);
  }

  FrozenRewritePatternSet final_patterns_;
  std::shared_ptr<OpPropertyHelper> helper_;
};

void Remapper::runOnOperation() {
  if (failed(applyPatternsAndFoldGreedily(getOperation(), final_patterns_)))
    signalPassFailure();
}

std::unique_ptr<Pass> CreateRemapperPass(bool enable_mkl_patterns,
                                         bool xla_auto_clustering) {
  return std::make_unique<Remapper>(enable_mkl_patterns, xla_auto_clustering);
}

}  // namespace tfg
}  // namespace mlir
