// RUN: tfg-transforms-opt -remapper %s | FileCheck %s

module {
  tfg.graph #tf_type.version<producer = 1085, min_consumer = 0> {
    // CHECK: %[[PLACEHOLDER:.*]], {{.*}} name("input_tensor")
    %Placeholder, %ctl = Placeholder device("/device:CPU:0") name("input_tensor") {dtype = f32, shape = #tf_type.shape<1x3x3x1>} : () -> (tensor<*xf32>)
    // CHECK: %[[FILTER:.*]], {{.*}} name("Const")
    %Const, %ctl_0 = Const device("/device:CPU:0") name("Const") {dtype = f32, value = dense<[[[[1.11986792, -3.0272491]]]]> : tensor<1x1x1x2xf32>} : () -> (tensor<*xf32>)
    // CHECK: %[[BIAS:.*]], {{.*}} name("Const_1")
    %Const_1, %ctl_2 = Const device("/device:CPU:0") name("Const_1") {dtype = f32, value = dense<[0.531091094, -0.719168067]> : tensor<2xf32>} : () -> (tensor<*xf32>)
    %Conv2D, %ctl_3 = Conv2D(%Placeholder, %Const) device("/device:CPU:0") name("Conv2D") {T = f32, data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: _FusedConv2D(%[[PLACEHOLDER]], %[[FILTER]], %[[BIAS]]) {{.*}} name("BiasAdd")
    %BiasAdd, %ctl_4 = BiasAdd(%Conv2D, %Const_1) device("/device:CPU:0") name("BiasAdd") {T = f32, data_format = "NHWC"} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: _FusedConv2D(%[[PLACEHOLDER]], %[[FILTER]], %[[BIAS]]) {{.*}} name("LeakyRelu") {{.*}} fused_ops = ["BiasAdd", "LeakyRelu"]
    %LeakyRelu, %ctl_5 = LeakyRelu(%BiasAdd) device("/device:CPU:0") name("LeakyRelu") {T = f32, alpha = 3.000000e-01 : f32} : (tensor<*xf32>) -> (tensor<*xf32>)
    %Conv2D_6, %ctl_7 = Conv2D(%Placeholder, %Const) device("/device:CPU:0") name("Conv2D_1") {T = f32, data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: %[[BIAS_ADD:.*]], {{.*}} _FusedConv2D(%[[PLACEHOLDER]], %[[FILTER]], %[[BIAS]]) {{.*}} name("BiasAdd_1")
    %BiasAdd_8, %ctl_9 = BiasAdd(%Conv2D_6, %Const_1) device("/device:CPU:0") name("BiasAdd_1") {T = f32, data_format = "NHWC"} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: Relu(%[[BIAS_ADD]]) {{.*}} name("Relu")
    %Relu, %ctl_10 = Relu(%BiasAdd_8) device("/device:CPU:0") name("Relu") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: Elu(%[[BIAS_ADD]]) {{.*}} name("Elu")
    %Elu, %ctl_11 = Elu(%BiasAdd_8) device("/device:CPU:0") name("Elu") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: Sigmoid(%[[BIAS_ADD]]) {{.*}} name("Sigmoid")
    %Sigmoid, %ctl_12 = Sigmoid(%BiasAdd_8) device("/device:CPU:0") name("Sigmoid") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
  }
}
