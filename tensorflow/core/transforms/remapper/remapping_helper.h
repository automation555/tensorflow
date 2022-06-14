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
#ifndef TENSORFLOW_CORE_TRANSFORMS_REMAPPER_REMAPPING_HELPER_H_
#define TENSORFLOW_CORE_TRANSFORMS_REMAPPER_REMAPPING_HELPER_H_

#include "tensorflow/core/transforms/utils/op_cat_helper.h"
#include "tensorflow/core/transforms/utils/utils.h"

namespace mlir {
namespace tfg {
namespace remapping {

// Helper structure for collecting fusion info
struct ContractionWithBiasAdd {
  Operation* contraction;
  Operation* bias_add;
};

// Helper structure for collecting fusion info
struct ContractionWithBiasAddAndActivation {
  Operation* contraction;
  Operation* bias_add;
  Operation* activation;
};

enum class Activation { Relu, Relu6, Elu, LeakyRelu, Tanh, Sigmoid };

inline std::string GetTFGActivation(Activation activation) {
  switch (activation) {
    case Activation::Relu:
      return "tfg.Relu";
    case Activation::Relu6:
      return "tfg.Relu6";
    case Activation::Elu:
      return "tfg.Elu";
    case Activation::LeakyRelu:
      return "tfg.LeakyRelu";
    case Activation::Tanh:
      return "tfg.Tanh";
    case Activation::Sigmoid:
      return "tfg.Sigmoid";
    default:
      return "tfg.NoOp";
  }
}

class OpPropertyHelper : public OpCatHelper {
 public:
  OpPropertyHelper(TFGraphDialect* dialect, bool onednn_enabled = false,
                   bool xla_auto_clustering = false)
      : OpCatHelper(dialect),
        dialect_(dialect),
        is_onednn_enabled_(onednn_enabled),
        xla_auto_clustering_(xla_auto_clustering) {}

  // Get the TFG dialect instance.
  TFGraphDialect* getDialect() { return dialect_; }

  bool HasControlFaninOrFanOut(Operation* op) {
    TFOp wrapper_op(op);
    bool has_ctl_fanin = !(wrapper_op.getControlOperands().empty());
    bool has_ctl_fanout = !(wrapper_op.controlRet().getUses().empty());
    if (has_ctl_fanin || has_ctl_fanout)
      return true;
    else
      return false;
  }

  bool HasAtMostOneFanoutAtPort0(Operation* op) {
    // Note tfg operations have at least 2 results: at least 1 non-control
    // and exactly 1 control result.
    return op->getNumResults() > 0 &&
           (op->getResult(0).hasOneUse() || op->getResult(0).use_empty());
  }

  bool IsContraction(Operation* op) {
    return dialect_->IsConv2D(op) || dialect_->IsConv3D(op) ||
           dialect_->IsDepthwiseConv2dNative(op) || dialect_->IsMatMul(op) ||
           dialect_->IsMatMul(op);
  }

  bool HaveSameDataType(Operation* lhs_op, Operation* rhs_op,
                        const StringRef& attr_name = "T") {
    if (!lhs_op->hasAttrOfType<TypeAttr>(attr_name) ||
        !rhs_op->hasAttrOfType<TypeAttr>(attr_name))
      return false;
    Type lhs_dtype = lhs_op->getAttrOfType<TypeAttr>(attr_name).getValue();
    Type rhs_dtype = rhs_op->getAttrOfType<TypeAttr>(attr_name).getValue();
    return !lhs_dtype.isa<NoneType>() && !rhs_dtype.isa<NoneType>() &&
           (lhs_dtype == rhs_dtype);
  }

  // This function is currently used by contraction ops.
  bool IsGpuCompatibleDataType(Operation* contraction_op,
                               const StringRef& attr_name = "T") {
    Type dtype;
    if (contraction_op->hasAttrOfType<TypeAttr>(attr_name))
      dtype = contraction_op->getAttrOfType<TypeAttr>(attr_name).getValue();
    else
      return false;
    if (dialect_->IsConv2D(contraction_op))
      return dtype.isa<Float32Type>();
    else if (dialect_->IsMatMul(contraction_op))
      return dtype.isa<Float32Type>() || dtype.isa<Float64Type>();
    else
      return false;
  }

  // This function is currently used by contraction ops.
  bool IsCpuCompatibleDataType(Operation* contraction_op,
                               const StringRef& attr_name = "T") {
    Type dtype;
    if (contraction_op->hasAttrOfType<TypeAttr>(attr_name))
      dtype = contraction_op->getAttrOfType<TypeAttr>(attr_name).getValue();
    else
      return false;
    if (is_onednn_enabled_) {
      // Only contraction ops are supported.
      bool is_supported = IsContraction(contraction_op) ||
                          dialect_->IsAnyBatchMatMul(contraction_op);
      return is_supported &&
             (dtype.isa<Float32Type>() || dtype.isa<BFloat16Type>());
    }

    if (dialect_->IsConv2D(contraction_op))
      return dtype.isa<Float32Type>() || dtype.isa<Float64Type>();
    else if (dialect_->IsMatMul(contraction_op))
      return dtype.isa<Float32Type>();
    else
      return false;
  }

  // This function is currently used by convolution type op
  bool IsGpuCompatibleDataFormat(Operation* conv_op,
                                 const StringRef& attr_name = "data_format") {
    StringRef data_format;
    if (conv_op->hasAttrOfType<StringAttr>(attr_name))
      data_format = conv_op->getAttrOfType<StringAttr>(attr_name).getValue();
    else
      return false;
    if (dialect_->IsConv2D(conv_op))
      return data_format == "NHWC" || data_format == "NCHW";
    else
      return false;
  }

  // This function is currently used by convolution type op
  bool IsCpuCompatibleDataFormat(Operation* conv_op,
                                 const StringRef& attr_name = "data_format") {
    StringRef data_format;
    if (conv_op->hasAttrOfType<StringAttr>(attr_name))
      data_format = conv_op->getAttrOfType<StringAttr>(attr_name).getValue();
    else
      return false;
    if (dialect_->IsConv2D(conv_op))
      return data_format == "NHWC" ||
             (is_onednn_enabled_ && data_format == "NCHW");
    else if (dialect_->IsConv3D(conv_op))
      return data_format == "NDHWC" ||
             (is_onednn_enabled_ && data_format == "NCDHW");
    else
      return false;
  }

  bool IsGpuCompatible(const ContractionWithBiasAddAndActivation& pattern) {
#if TENSORFLOW_USE_ROCM
    // ROCm does not support _FusedConv2D. Does it suppport _FusedMatMul?
    return false;
#endif
    // The TF->XLA bridge does not support `_FusedMatMul` so we avoid creating
    // this op. Furthermore, XLA already does this fusion internally so there
    // is no true benefit from doing this optimization if XLA is going to
    // compile the unfused operations anyway.
    if (xla_auto_clustering_) return false;
    if (!util::NodeIsOnGpu(pattern.contraction)) return false;
    if (!dialect_->IsRelu(pattern.activation)) return false;
    if (dialect_->IsMatMul(pattern.contraction)) {
      return IsGpuCompatibleDataType(pattern.contraction);
    } else {
      // TODO(intel-tf): Add spatial convolution support on GPU
      return false;
    }
  }

  // Currently GPU does not supprt contraction + bias_add
  bool IsGpuCompatible(const ContractionWithBiasAdd&) { return false; }

  bool IsCpuCompatible(Operation* contraction_op) {
    if (!util::NodeIsOnCpu(contraction_op)) return false;
    if (dialect_->IsConv2D(contraction_op) ||
        dialect_->IsConv3D(contraction_op))
      return IsCpuCompatibleDataType(contraction_op) &&
             IsCpuCompatibleDataFormat(contraction_op);
    else if (dialect_->IsMatMul(contraction_op) ||
             dialect_->IsAnyBatchMatMul(contraction_op) ||
             dialect_->IsDepthwiseConv2dNative(contraction_op))
      return IsCpuCompatibleDataType(contraction_op);
    else
      return false;
  }

  template <typename Pattern>
  bool IsDeviceCompatible(const Pattern& pattern) {
    return IsGpuCompatible(pattern) || IsCpuCompatible(pattern.contraction);
  }

 private:
  TFGraphDialect* dialect_;
  bool is_onednn_enabled_;
  bool xla_auto_clustering_;
};

}  // namespace remapping
}  // namespace tfg
}  // namespace mlir

#endif  // TENSORFLOW_CORE_TRANSFORMS_REMAPPER_REMAPPING_HELPER_H_
