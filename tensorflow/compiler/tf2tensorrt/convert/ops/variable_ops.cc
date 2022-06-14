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
#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"
#include "tensorflow/compiler/tf2tensorrt/convert/op_converter.h"
#include "tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "third_party/tensorrt/NvInfer.h"
#include "third_party/tensorrt/NvInferRuntimeCommon.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

class ConvertVariableV2 : public OpConverterBase<ConvertVariableV2> {
 public:
  ConvertVariableV2(OpConverterParams* params)
      : OpConverterBase<ConvertVariableV2>(params) {}

  struct VariableV2Attributes {
    TensorShapeProto shape_proto;
    TensorShape shape;
    string name;
    DataType dtype;
    string shared_name;
    string container;
  };

  static constexpr std::array<InputArgSpec, 0> InputSpec() { return {}; }

  static constexpr std::array<DataType, 2> AllowedDataTypes() {
    return {DataType::DT_FLOAT, DataType::DT_HALF};
  }

  static constexpr const char* NodeDefDataTypeAttributeName() {
    return "dtype";
  }

  template <typename T>
  Status ValidateImpl() {
    const auto& node_def = params_->node_def;

    // Verify and consume node attributes.
    StatusOr<TensorShapeProto> shape_proto =
        GetAttrValue<TensorShapeProto>("shape");
    StatusOr<string> shared_name = GetAttrValue<string>("shared_name");
    StatusOr<string> container = GetAttrValue<string>("container");
    TRT_ENSURE_OK(shape_proto);
    TRT_ENSURE_OK(shared_name);
    TRT_ENSURE_OK(container);

    attrs_.shape_proto = *shape_proto;
    attrs_.shape = TensorShape(*shape_proto);
    attrs_.name = node_def.name();
    attrs_.shared_name = *shared_name;
    attrs_.container = *container;

    Tensor tensor(attrs_.dtype, attrs_.shape);
    auto tensor_flat = tensor.flat<T>();
    for (int64_t i = 0; i < tensor_flat.size(); i++) {
      tensor_flat(i) = T(0.0f);
    }

    TRT_ShapedWeights weights;
    TF_RETURN_IF_ERROR(
        TfTensorToTrtWeights(tensor, params_->weight_store, &weights));

    // Only push outputs during validation and when outputs are expected.
    if (params_->validation_only && params_->outputs != nullptr) {
      AddOutput(TRT_TensorOrWeights(weights));
    }
    return Status::OK();
  }

  Status Validate() {
    StatusOr<DataType> dtype = GetAttrValue<DataType>("dtype");
    TRT_ENSURE_OK(dtype);
    attrs_.dtype = *dtype;

    switch (attrs_.dtype) {
      case DT_FLOAT:
        return ValidateImpl<float>();
      case DT_HALF:
        return ValidateImpl<Eigen::half>();
    }
  }

  template <typename T>
  Status ConvertImpl() {
    Tensor tensor(attrs_.dtype, attrs_.shape);
    auto tensor_flat = tensor.flat<T>();

    auto ctx = params_->converter->context();
    CHECK(ctx);
    auto lib = ctx->function_library();

    // Clone function library runtime in order to get a mutable library
    // definition to add and run a function with the variable operation.
    std::unique_ptr<FunctionLibraryDefinition> lib_def;
    std::unique_ptr<ProcessFunctionLibraryRuntime> lib_pflr;
    FunctionLibraryRuntime* lib_clone;  // Not owned.
    TF_RETURN_IF_ERROR(lib->Clone(&lib_def, &lib_pflr, &lib_clone));

    // Create function definition.
    string func_name = attrs_.name + "/func";
    FunctionDef fdef = FunctionDefHelper::Define(
        func_name,                                                 // Name
        {},                                                        // Args
        {strings::StrCat("out: ", DataTypeString(attrs_.dtype))},  // Returns
        {},                                                        // Attr def
        // Nodes
        {{{attrs_.name},
          "VariableV2",
          {},
          {{"dtype", attrs_.dtype},
           {"shape", attrs_.shape_proto},
           {"container", attrs_.container},
           {"shared_name", attrs_.shared_name}}},
         {{"out"}, "Identity", {attrs_.name}, {{"T", attrs_.dtype}}}});

    // Add function definition to the library.
    lib_def->AddFunctionDef(fdef);

    // Instantiate function.
    FunctionLibraryRuntime::Handle func_handle;
    FunctionLibraryRuntime::InstantiateOptions inst_ops;
    inst_ops.state_handle = "";
    inst_ops.target = ctx->device()->name();
    AttrValueMap attr_list;
    TF_RETURN_IF_ERROR(lib_clone->Instantiate(func_name, AttrSlice(&attr_list),
                                              inst_ops, &func_handle));

    FunctionLibraryRuntime::Options opts;
    opts.rendezvous = ctx->rendezvous();
    opts.cancellation_manager = ctx->cancellation_manager();
    opts.runner = ctx->runner();

    std::vector<Tensor> args;  // empty
    std::vector<Tensor>* rets = new std::vector<Tensor>();
    std::unique_ptr<std::vector<Tensor>> outputs_wrapper(rets);

    // Run the new function synchronously.
    TF_RETURN_IF_ERROR(lib_clone->RunSync(opts, func_handle, args, rets));

    CHECK(ctx->op_device_context());
    CHECK(ctx->op_device_context()->stream());

    // Copy tensor.
    const cudaStream_t* stream = CHECK_NOTNULL(
        reinterpret_cast<const cudaStream_t*>(ctx->op_device_context()
                                                  ->stream()
                                                  ->implementation()
                                                  ->GpuStreamMemberHack()));

    auto ret = cudaMemcpyAsync(tensor_flat.data(), rets->at(0).flat<T>().data(),
                               rets->at(0).NumElements() * sizeof(T),
                               cudaMemcpyDeviceToHost, *stream);
    if (ret != 0) {
      return errors::Internal("Could not copy the variable ", attrs_.name);
    }
    cudaStreamSynchronize(*stream);

    TRT_ShapedWeights weights;
    TF_RETURN_IF_ERROR(
        TfTensorToTrtWeights(tensor, params_->weight_store, &weights));

    AddOutput(TRT_TensorOrWeights(weights));
    return Status::OK();
  }

  Status Convert() {
    switch (attrs_.dtype) {
      case DT_FLOAT:
        return ConvertImpl<float>();
      case DT_HALF:
        return ConvertImpl<Eigen::half>();
    }
  }

 private:
  VariableV2Attributes attrs_{};
};
REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertVariableV2>(),
                                  {"VariableV2"});

class ConvertReadVariableOp : public OpConverterBase<ConvertReadVariableOp> {
 public:
  ConvertReadVariableOp(OpConverterParams* params)
      : OpConverterBase<ConvertReadVariableOp>(params) {}

  struct ReadVariableOpAttributes {
    TensorShapeProto shape_proto;
    TensorShape shape;
    string name;
    DataType dtype;
  };

  static constexpr std::array<InputArgSpec, 1> InputSpec() {
    return {InputArgSpec::Create("resource", TrtInputArg::kResource)};
  }

  static constexpr std::array<DataType, 2> AllowedDataTypes() {
    return {DataType::DT_FLOAT, DataType::DT_HALF};
  }

  static constexpr const char* NodeDefDataTypeAttributeName() {
    return "dtype";
  }

  template <typename T>
  Status ValidateImpl() {
    const auto& node_def = params_->node_def;

    // Verify and consume node attributes.
    StatusOr<TensorShapeProto> shape_proto =
        GetAttrValue<TensorShapeProto>("_shape");
    TRT_ENSURE_OK(shape_proto);

    attrs_.shape_proto = *shape_proto;
    attrs_.shape = TensorShape(*shape_proto);
    attrs_.name = node_def.name();

    Tensor tensor(attrs_.dtype, attrs_.shape);
    auto tensor_flat = tensor.flat<T>();
    for (int64_t i = 0; i < tensor_flat.size(); i++) {
      tensor_flat(i) = T(0.0f);
    }

    TRT_ShapedWeights weights;
    TF_RETURN_IF_ERROR(
        TfTensorToTrtWeights(tensor, params_->weight_store, &weights));

    // Only push outputs during validation and when outputs are expected.
    if (params_->validation_only && params_->outputs != nullptr) {
      AddOutput(TRT_TensorOrWeights(weights));
    }
    return Status::OK();
  }

  Status Validate() {
    if (params_->use_implicit_batch) {
      return errors::Unimplemented("Implicit batch mode not supported, at ",
                                   params_->node_def.name());
    }

    StatusOr<DataType> dtype = GetAttrValue<DataType>("dtype");
    TRT_ENSURE_OK(dtype);
    attrs_.dtype = *dtype;

    switch (attrs_.dtype) {
      case DT_FLOAT:
        return ValidateImpl<float>();
      case DT_HALF:
        return ValidateImpl<Eigen::half>();
    }
  }

  template <typename T>
  Status ConvertImpl() {
    Tensor tensor(attrs_.dtype, attrs_.shape);
    auto tensor_flat = tensor.flat<T>();

    auto ctx = params_->converter->context();
    CHECK(ctx);
    auto lib = ctx->function_library();

    const auto& inputs = params_->inputs;
    const TRT_TensorOrWeights& handle = inputs.at(0);

    // Clone function library runtime in order to get a mutable library
    // definition to add and run a function with the variable operation.
    std::unique_ptr<FunctionLibraryDefinition> lib_def;
    std::unique_ptr<ProcessFunctionLibraryRuntime> lib_pflr;
    FunctionLibraryRuntime* lib_clone;  // Not owned.
    TF_RETURN_IF_ERROR(lib->Clone(&lib_def, &lib_pflr, &lib_clone));

    // Create function definition.
    string func_name = attrs_.name + "/func";
    FunctionDef fdef = FunctionDefHelper::Define(
        func_name,         // Name
        {"in: resource"},  // Args
        {"out: float"},    // Returns
        {},                // Attr def
        // Nodes
        {{{attrs_.name},
          "ReadVariableOp",
          {"in"},  // Name of the Placeholder or VarHandleOp
          {{"dtype", DT_FLOAT}}},
         {{"out"}, "Identity", {attrs_.name}, {{"T", DT_FLOAT}}}});

    // Add function definition to the library.
    lib_def->AddFunctionDef(fdef);

    // Instantiate function.
    FunctionLibraryRuntime::Handle func_handle;
    FunctionLibraryRuntime::InstantiateOptions inst_ops;
    inst_ops.state_handle = "";
    inst_ops.target = ctx->device()->name();
    AttrValueMap attr_list;
    TF_RETURN_IF_ERROR(lib_clone->Instantiate(func_name, AttrSlice(&attr_list),
                                              inst_ops, &func_handle));

    FunctionLibraryRuntime::Options opts;
    opts.rendezvous = ctx->rendezvous();
    opts.cancellation_manager = ctx->cancellation_manager();
    opts.runner = ctx->runner();

    // Create input tensor with the resource handle.
    std::vector<Tensor> args;
    args.emplace_back(handle.resource());

    std::vector<Tensor>* rets = new std::vector<Tensor>();
    std::unique_ptr<std::vector<Tensor>> outputs_wrapper(rets);

    // Run the new function synchronously.
    TF_RETURN_IF_ERROR(lib_clone->RunSync(opts, func_handle, args, rets));

    CHECK(ctx->op_device_context());
    CHECK(ctx->op_device_context()->stream());

    // Copy tensor.
    const cudaStream_t* stream = CHECK_NOTNULL(
        reinterpret_cast<const cudaStream_t*>(ctx->op_device_context()
                                                  ->stream()
                                                  ->implementation()
                                                  ->GpuStreamMemberHack()));

    auto ret = cudaMemcpyAsync(tensor_flat.data(), rets->at(0).flat<T>().data(),
                               rets->at(0).NumElements() * sizeof(T),
                               cudaMemcpyDeviceToHost, *stream);
    if (ret != 0) {
      return errors::Internal("Could not copy the variable ", attrs_.name);
    }
    cudaStreamSynchronize(*stream);

    TRT_ShapedWeights weights;
    TF_RETURN_IF_ERROR(
        TfTensorToTrtWeights(tensor, params_->weight_store, &weights));

    AddOutput(TRT_TensorOrWeights(weights));
    return Status::OK();
  }

  Status Convert() {
    switch (attrs_.dtype) {
      case DT_FLOAT:
        return ConvertImpl<float>();
      case DT_HALF:
        return ConvertImpl<Eigen::half>();
    }
  }

 private:
  ReadVariableOpAttributes attrs_{};
};
REGISTER_DEFAULT_TRT_OP_CONVERTER(
    MakeConverterFunction<ConvertReadVariableOp>(), {"ReadVariableOp"});

class ConvertResourceGather : public OpConverterBase<ConvertResourceGather> {
 public:
  ConvertResourceGather(OpConverterParams* params)
      : OpConverterBase<ConvertResourceGather>(params) {}

  struct ResourceGatherAttributes {
    TensorShapeProto shape_proto;
    TensorShape shape;
    string name;
    DataType dtype;
  };

  static constexpr std::array<InputArgSpec, 2> InputSpec() {
    return {InputArgSpec::Create("resource", TrtInputArg::kResource),
            InputArgSpec::Create("indices", TrtInputArg::kBoth)};
  }

  static constexpr std::array<DataType, 2> AllowedDataTypes() {
    return {DataType::DT_FLOAT, DataType::DT_HALF};
  }

  static constexpr const char* NodeDefDataTypeAttributeName() {
    return "dtype";
  }

  template <typename T>
  Status ValidateImpl() {
    const auto& node_def = params_->node_def;

    // Verify and consume node attributes.
    StatusOr<TensorShapeProto> shape_proto =
        GetAttrValue<TensorShapeProto>("_shape");
    TRT_ENSURE_OK(shape_proto);

    attrs_.shape_proto = *shape_proto;
    attrs_.shape = TensorShape(*shape_proto);
    attrs_.name = node_def.name();

    StatusOr<DimsAdapter> dims_adapter =
        DimsAdapter::Create(attrs_.shape, false);
    TRT_ENSURE_OK(dims_adapter);
    nvinfer1::Dims params_dims = dims_adapter->AsTrtDims();

    const auto& inputs = params_->inputs;
    const TRT_TensorOrWeights& handle = inputs.at(0);
    const TRT_TensorOrWeights& indices_input = inputs.at(1);

    const int params_tf_rank = params_dims.nbDims;
    const int indices_tf_rank = indices_input.GetTrtDims().nbDims;
    const int tf_gather_output_rank = params_tf_rank + indices_tf_rank - 1;
    if (tf_gather_output_rank > nvinfer1::Dims::MAX_DIMS) {
      return errors::InvalidArgument(
          "Result of gather has dimension greater than ",
          nvinfer1::Dims::MAX_DIMS + 1);
    }

    return Status::OK();
  }

  Status Validate() {
    if (params_->use_implicit_batch) {
      return errors::Unimplemented("Implicit batch mode not supported, at ",
                                   params_->node_def.name());
    }

    StatusOr<DataType> dtype = GetAttrValue<DataType>("dtype");
    TRT_ENSURE_OK(dtype);
    attrs_.dtype = *dtype;

    switch (attrs_.dtype) {
      case DT_FLOAT:
        return ValidateImpl<float>();
      case DT_HALF:
        return ValidateImpl<Eigen::half>();
    }
  }

  template <typename T>
  Status ConvertImpl() {
    auto ctx = params_->converter->context();
    CHECK(ctx);
    auto lib = ctx->function_library();

    const auto& node_def = params_->node_def;
    const auto& inputs = params_->inputs;
    const TRT_TensorOrWeights& handle = inputs.at(0);
    const TRT_TensorOrWeights& indices_input = inputs.at(1);

    // ResourceGather doesn't have an axis attribute like GatherV2.
    int trt_axis = 0;

    // Clone function library runtime in order to get a mutable library
    // definition to add and run a function with the variable operation.
    std::unique_ptr<FunctionLibraryDefinition> lib_def;
    std::unique_ptr<ProcessFunctionLibraryRuntime> lib_pflr;
    FunctionLibraryRuntime* lib_clone;  // Not owned.
    TF_RETURN_IF_ERROR(lib->Clone(&lib_def, &lib_pflr, &lib_clone));

    // Create function definition.
    string func_name = attrs_.name + "/func";
    FunctionDef fdef = FunctionDefHelper::Define(
        func_name,         // Name
        {"in: resource"},  // Args
        {"out: float"},    // Returns
        {},                // Attr def
        // Nodes
        {{{attrs_.name},
          "ReadVariableOp",
          {"in"},  // Name of the Placeholder or VarHandleOp
          {{"dtype", DT_FLOAT}}},
         {{"out"}, "Identity", {attrs_.name}, {{"T", DT_FLOAT}}}});

    // Add function definition to the library.
    lib_def->AddFunctionDef(fdef);

    // Instantiate function.
    FunctionLibraryRuntime::Handle func_handle;
    FunctionLibraryRuntime::InstantiateOptions inst_ops;
    inst_ops.state_handle = "";
    inst_ops.target = ctx->device()->name();
    AttrValueMap attr_list;
    TF_RETURN_IF_ERROR(lib_clone->Instantiate(func_name, AttrSlice(&attr_list),
                                              inst_ops, &func_handle));

    FunctionLibraryRuntime::Options opts;
    opts.rendezvous = ctx->rendezvous();
    opts.cancellation_manager = ctx->cancellation_manager();
    opts.runner = ctx->runner();

    // Create input tensor with the resource handle.
    std::vector<Tensor> args;
    args.emplace_back(handle.resource());

    std::vector<Tensor>* rets = new std::vector<Tensor>();
    std::unique_ptr<std::vector<Tensor>> outputs_wrapper(rets);

    // Run the new function synchronously.
    TF_RETURN_IF_ERROR(lib_clone->RunSync(opts, func_handle, args, rets));

    // Create weights with the same shape as the variable.
    Tensor tensor(DataType::DT_FLOAT, (*rets)[0].shape());
    auto tensor_flat = tensor.flat<float>();

    CHECK_NOTNULL(ctx->op_device_context());
    CHECK_NOTNULL(ctx->op_device_context()->stream());

    // Copy tensor.
    const cudaStream_t* stream = CHECK_NOTNULL(
        reinterpret_cast<const cudaStream_t*>(ctx->op_device_context()
                                                  ->stream()
                                                  ->implementation()
                                                  ->GpuStreamMemberHack()));

    auto ret = cudaMemcpyAsync(tensor_flat.data(), rets->at(0).flat<T>().data(),
                               rets->at(0).NumElements() * sizeof(T),
                               cudaMemcpyDeviceToHost, *stream);
    if (ret != 0) {
      return errors::Internal("Could not copy the variable ", attrs_.name);
    }
    cudaStreamSynchronize(*stream);

    TRT_ShapedWeights weights;
    TF_RETURN_IF_ERROR(
        TfTensorToTrtWeights(tensor, params_->weight_store, &weights));

    // Convert indices to tensor if it is a constant.
    ITensorProxyPtr indices_tensor = nullptr;
    if (indices_input.is_weights()) {
      indices_tensor = params_->converter->CreateConstantLayer(
          indices_input.weights(), indices_input.GetTrtDims());
    } else {
      indices_tensor = indices_input.tensor();
    }

    // Convert variable to tensor.
    ITensorProxyPtr params_tensor = nullptr;
    params_tensor = params_->converter->CreateConstantLayer(
        weights, weights.Shape().AsTrtDims());

    // Note on how IGatherLayer works: if both the data and indices tensors have
    // a batch size dimension of size N, it performs:
    // for batchid in xrange(N):
    //   output[batchid, a0, ..., an, i, ..., j, b0, ..., bn] = (
    //       data[batchid, a0, ..., an, indices[batchid, i, ..., j] b0, ...,
    //       bn])
    nvinfer1::IGatherLayer* layer = params_->converter->network()->addGather(
        *params_tensor->trt_tensor(), *indices_tensor->trt_tensor(), trt_axis);
    TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
    params_->converter->SetLayerName(layer, node_def);

    ITensorProxyPtr output_tensor = layer->getOutput(0);
    nvinfer1::Dims trt_gather_output_dims = output_tensor->getDimensions();

    // When input and indices are both constants, for the supported cases,
    // reshape
    // output so that after removing the implicit batch dim it will match the
    // output shape of TF GatherV2 op.
    if (params_->use_implicit_batch && indices_input.is_weights()) {
      for (int i = trt_axis; i < trt_gather_output_dims.nbDims - 1; ++i) {
        trt_gather_output_dims.d[i] = trt_gather_output_dims.d[i + 1];
      }

      // Squeeze the implicit batch dimension out. Note: this works only
      // when batch size for both inputs and indices are 1.
      --trt_gather_output_dims.nbDims;

      TF_RETURN_IF_ERROR(PrepareTensorForShape(
          params_->converter, TRT_TensorOrWeights(output_tensor),
          trt_gather_output_dims,
          /*validation_only=*/false, &output_tensor, node_def));
    }

    AddOutput(TRT_TensorOrWeights(output_tensor));
    return Status::OK();
  }

  Status Convert() {
    switch (attrs_.dtype) {
      case DT_FLOAT:
        return ConvertImpl<float>();
      case DT_HALF:
        return ConvertImpl<Eigen::half>();
    }
  }

 private:
  ResourceGatherAttributes attrs_{};
};
REGISTER_DEFAULT_TRT_OP_CONVERTER(
    MakeConverterFunction<ConvertResourceGather>(), {"ResourceGather"});

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT