// Copyright 2021 The WebNN-native Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ie_model.h"

#include <gna/gna_config.hpp>
#include <string>
#include <utility>

#include "ngraph/ngraph.hpp"
#include "ngraph/node.hpp"
#include "utils.h"

namespace InferenceEngine {

using namespace ngraph;

namespace {

SizeVector ToVector(int32_t const* value, uint32_t count) {
  SizeVector data;
  data.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    data.push_back(value[i]);
  }
  return data;
}

std::shared_ptr<ngraph::Node> CreateConstantNode(const int32_t* dimensions,
                                                 uint32_t dimensionsCount,
                                                 const float* value) {
  std::shared_ptr<ngraph::Node> constant;
  SizeVector constant_dimensions(dimensions, dimensions + dimensionsCount);
  SizeVector constant_value;
  uint32_t size = 1;
  for (uint32_t i = 0; i < dimensionsCount; ++i) {
    size *= dimensions[i];
  }
  constant_value.reserve(size);
  for (uint32_t i = 0; i < size; ++i) {
    constant_value.push_back(value[i]);
  }
  constant =
      op::Constant::create(element::f32, constant_dimensions, constant_value);
  return constant;
}

ie_operand_t* CreateOperand(std::string& name) {
  ie_operand_t* operand = new ie_operand_t();
  std::unique_ptr<char[]> node_name(new char[name.length() + 1]);
  operand->name = node_name.release();
  memcpy(operand->name, name.c_str(), name.length() + 1);
  return operand;
}

ngraph::Output<ngraph::Node> Reshape(
    const ngraph::Output<ngraph::Node>& input_node,
    const std::vector<size_t>& new_shape) {
  auto target_shape_node = std::make_shared<op::Constant>(
      element::i64, Shape{new_shape.size()}, new_shape);
  auto reshape_node = std::make_shared<op::v1::Reshape>(
      input_node, target_shape_node->output(0), true);
  return reshape_node->output(0);
}

// Transpose NHWC <=> NCHW.
std::shared_ptr<ngraph::Node> TransposeInputLayout(
    ngraph::Output<ngraph::Node> node,
    bool nhwc_to_nchw) {
  AxisVector order =
      nhwc_to_nchw ? AxisVector{0, 3, 1, 2} : AxisVector{0, 2, 3, 1};
  const auto order_node =
      op::Constant::create(element::i64, Shape{order.size()}, order);
  auto transpose_node = std::make_shared<op::v1::Transpose>(node, order_node);
  return transpose_node;
}

// hwio => oihw or ohwi => oihw
ngraph::Output<ngraph::Node> TransposeFilterLayout(
    ngraph::Output<ngraph::Node> node,
    ie_filter_operand_layout layout) {
  if (layout == ie_filter_operand_layout::Oihw) {
    return node;
  }

  AxisVector order;
  switch (layout) {
    case ie_filter_operand_layout::Hwio:
      order = AxisVector{3, 2, 0, 1};
      break;
    case ie_filter_operand_layout::Ohwi:
      order = AxisVector{0, 3, 1, 2};
      break;
    case ie_filter_operand_layout::Ihwo:
      order = AxisVector{3, 0, 1, 2};
      break;
    default:
      assert(0);
      break;
  }
  const auto order_node =
      op::Constant::create(element::i64, Shape{order.size()}, order);
  auto transpose_node = std::make_shared<op::v1::Transpose>(node, order_node);
  return transpose_node->output(0);
}

}  // namespace

ie_operand_t* Model::AddConstant(ie_operand_descriptor_t const* desc,
                                 void const* value,
                                 size_t length) {
  SizeVector dims = ToVector(desc->dimensions, desc->dimensionsCount);
  // Generally, FP16 is preferable as it is most ubiquitous and performant
  // documented in
  // https://docs.openvinotoolkit.org/2021.1/openvino_docs_IE_DG_supported_plugins_Supported_Devices.html.
  bool fp32_precision = true;
  Blob::Ptr blob;
  if (fp32_precision) {
    // GNA only accepts FP32 precision, cpu/gpu use FP32 currently.
    blob = make_shared_blob<float>({Precision::FP32, dims, Layout::ANY});
  } else {
    // MYRIAD only accepts FP16 precision.
    blob = make_shared_blob<int16_t>({Precision::FP16, dims, Layout::ANY});
  }
  blob->allocate();
  const float* src = reinterpret_cast<const float*>(value);
  std::shared_ptr<op::Constant> node;
  if (fp32_precision) {
    float* dst = blob->buffer().as<float*>();
    CopyDataToBuffer<float>(dst, src, length);
    node = std::make_shared<op::Constant>(element::f32, dims, dst);
  } else {
    int16_t* dst = blob->buffer().as<int16_t*>();
    CopyDataToBuffer<int16_t>(dst, src, length);
    node = std::make_shared<op::Constant>(element::f16, dims, dst);
  }

  std::string node_name = node->get_name();
  name_node_map_[node_name] = node->output(0);
  return CreateOperand(node_name);
}

ie_operand_t* Model::AddInput(ie_operand_descriptor_t const* desc) {
  SizeVector dims = ToVector(desc->dimensions, desc->dimensionsCount);
  auto input_node =
      std::make_shared<op::v0::Parameter>(element::f32, Shape(dims));
  ngraph_inputs_.push_back(input_node);

  std::string node_name = input_node->get_name();
  name_node_map_[node_name] = input_node->output(0);
  return CreateOperand(node_name);
}

void Model::AddOutput(ie_operand_t* operand) {
  auto node_name = std::string(operand->name);
  auto output_node = std::make_shared<op::Result>(name_node_map_[node_name]);
  ngraph_outputs_.push_back(output_node);

  return;
}

ie_operand_t* Model::AddMatMul(ie_operand_t* a, ie_operand_t* b) {
  auto primary_node = name_node_map_[a->name];
  auto primary_shape = primary_node.get_shape();
  // Unsqueeze to 2D by adding axes with size 1 to the left of the shape if
  // first input is equal to 1.
  if (primary_shape.size() == 1) {
    primary_node = Reshape(primary_node, {1, primary_shape[0]});
  }
  auto secondary_node = name_node_map_[b->name];
  auto secondary_shape = secondary_node.get_shape();
  // Unsqueeze to 2D by adding axes with size 1 to the right of the shape if
  // second input is equal to 1.
  if (secondary_shape.size() == 1) {
    secondary_node = Reshape(secondary_node, {secondary_shape[0], 1});
  }
  auto matmul_node = std::make_shared<op::v0::MatMul>(
      primary_node, secondary_node, false, false);
  std::string node_name = matmul_node->get_name();
  name_node_map_[node_name] = matmul_node->output(0);
  // "max_size - 2" is out of range for unsigned int type when max_size is 1D
  // input shape in openvino implementation[1] that is traked with issue [2],
  // although nGraph has unsqueeze to 2D [3].
  // Here is a workaround to convert 2D output to scalar node.
  // [1]
  // https://github.com/openvinotoolkit/openvino/blob/releases/2021/1/inference-engine/src/transformations/src/transformations/convert_opset1_to_legacy/convert_matmul_to_fc_or_gemm.cpp#L61,
  // [2] https://github.com/openvinotoolkit/openvino/issues/4373
  // [3]
  // https://github.com/openvinotoolkit/openvino/blob/master/ngraph/core/src/op/matmul.cpp#L85
  if (primary_shape.size() == 1 && secondary_shape.size() == 1) {
    auto scalar_node = Reshape(matmul_node->output(0), {1});
    node_name = scalar_node.get_node()->get_name();
    name_node_map_[node_name] = scalar_node;
  }
  return CreateOperand(node_name);
}

ie_operand_t* Model::AddBatchNorm(ie_operand_t* input,
                                  ie_operand_t* mean,
                                  ie_operand_t* variance,
                                  ie_batch_norm_options_t* options) {
  // When input is a 4-D tensor of the "nchw" or "nhwc" layout,
  // options.axis should be set to 1 or 3 respectively.
  auto input_node = name_node_map_[input->name];
  bool nhwc = options->axis == 3;
  if (nhwc) {
    options->axis = 1;
    input_node =
        TransposeInputLayout(name_node_map_[input->name], true)->output(0);
  }
  auto mean_node = name_node_map_[mean->name];
  auto variance_node = name_node_map_[variance->name];
  auto channel = input_node.get_shape()[options->axis];
  auto scale_node =
      options->scale.name != nullptr
          ? name_node_map_[options->scale.name]
          : op::Constant::create(element::f32, Shape{channel}, {1})->output(0);
  auto bias_node =
      options->bias.name != nullptr
          ? name_node_map_[options->bias.name]
          : op::Constant::create(element::f32, Shape{channel}, {0})->output(0);
  auto batch_norm_node = std::make_shared<op::v0::BatchNormInference>(
      input_node, scale_node, bias_node, mean_node, variance_node,
      options->epsilon);
  auto node = nhwc ? TransposeInputLayout(batch_norm_node->output(0), false)
                   : batch_norm_node;
  std::string node_name = node->get_name();
  name_node_map_[node_name] = node->output(0);
  return CreateOperand(node_name);
}

ie_operand_t* Model::AddBinary(ie_binary_type type,
                               ie_operand_t* a,
                               ie_operand_t* b) {
  auto primary_node = name_node_map_[a->name];
  auto secondary_node = name_node_map_[b->name];
  std::shared_ptr<ngraph::Node> binary_node;
  switch (type) {
    case ie_binary_type::ADD:
      binary_node = std::make_shared<op::v1::Add>(primary_node, secondary_node);
      break;
    case ie_binary_type::MUL:
      binary_node =
          std::make_shared<op::v1::Multiply>(primary_node, secondary_node);
      break;
    default:
      THROW_IE_EXCEPTION << "The operation isn't supported";
  }

  std::string node_name = binary_node->get_name();
  name_node_map_[node_name] = binary_node->output(0);
  return CreateOperand(node_name);
}

ie_operand_t* Model::AddClamp(ie_operand_t* input,
                              ie_clamp_options_t* options) {
  auto input_node = name_node_map_[input->name];
  std::shared_ptr<ngraph::Node> clamp_node;
  if (options->minDimensionsCount == 0 && options->maxDimensionsCount == 0) {
    float min = options->minValue == nullptr
                    ? std::numeric_limits<float>::lowest()
                    : options->minValue[0];
    float max = options->maxValue == nullptr ? std::numeric_limits<float>::max()
                                             : options->maxValue[0];
    clamp_node = std::make_shared<op::v0::Clamp>(input_node, min, max);
  } else {
    std::shared_ptr<ngraph::Node> min_constant, max_constant;
    if (options->minValue != nullptr) {
      min_constant =
          CreateConstantNode(options->minDimensions,
                             options->minDimensionsCount, options->minValue);
    }

    if (options->maxValue != nullptr) {
      max_constant =
          CreateConstantNode(options->maxDimensions,
                             options->maxDimensionsCount, options->maxValue);
    }

    // Compare input with min value.
    auto max_node =
        options->minValue != nullptr
            ? std::make_shared<op::v1::Maximum>(input_node, min_constant)
                  ->output(0)
            : input_node;
    // Compare input with max value.
    clamp_node = options->maxValue != nullptr
                     ? std::make_shared<op::v1::Minimum>(max_node, max_constant)
                     : max_node.get_node_shared_ptr();
  }

  std::string node_name = clamp_node->get_name();
  name_node_map_[node_name] = clamp_node->output(0);
  return CreateOperand(node_name);
}

ie_operand_t* Model::AddConv2d(ie_operand_t* input,
                               ie_operand_t* filter,
                               ie_conv2d_options_t* options) {
  CoordinateDiff pad_begin = {options->padding[0], options->padding[2]};
  CoordinateDiff pad_end = {options->padding[1], options->padding[3]};
  Strides strides = {static_cast<size_t>(options->strides[0]),
                     static_cast<size_t>(options->strides[1])};
  Strides dilations = {static_cast<size_t>(options->dilations[0]),
                       static_cast<size_t>(options->dilations[1])};

  auto input_node =
      options->inputLayout == ie_input_operand_layout::Nchw
          ? name_node_map_[input->name]
          : TransposeInputLayout(name_node_map_[input->name], true)->output(0);
  auto filter_node = TransposeFilterLayout(name_node_map_[filter->name],
                                           options->filterLayout);
  op::PadType auto_pad;
  switch (options->autoPad) {
    case SameUpper:
      auto_pad = op::PadType::SAME_UPPER;
      break;
    case SameLower:
      auto_pad = op::PadType::SAME_LOWER;
      break;
    default:
      auto_pad = op::PadType::EXPLICIT;
  }
  std::shared_ptr<ngraph::Node> conv2d_node;
  if (options->groups > 1) {
    // Insert the groups to the shape of filter as first item.
    auto filters_shape = filter_node.get_shape();
    filters_shape.at(0) = filters_shape.at(0) / options->groups;
    filters_shape.insert(filters_shape.begin(), options->groups);
    // Reshape the filter to support groups conv.
    auto reshaped_filters = Reshape(filter_node, filters_shape);
    conv2d_node = std::make_shared<op::v1::GroupConvolution>(
        input_node, reshaped_filters, strides, pad_begin, pad_end, dilations,
        auto_pad);
  } else {
    conv2d_node = std::make_shared<op::v1::Convolution>(
        input_node, filter_node, strides, pad_begin, pad_end, dilations,
        auto_pad);
  }

  auto node = options->inputLayout == ie_input_operand_layout::Nhwc
                  ? TransposeInputLayout(conv2d_node->output(0), false)
                  : conv2d_node;
  std::string node_name = node->get_name();
  name_node_map_[node_name] = node->output(0);
  return CreateOperand(node_name);
}

ie_operand_t* Model::AddPool2d(ie_pool_type type,
                               ie_operand_t* input,
                               ie_pool2d_options_t* options) {
  // Use the height and width dimensions of the input shape as windowDimensions
  // if it is not present in options.
  auto input_node =
      options->layout == ie_input_operand_layout::Nchw
          ? name_node_map_[input->name]
          : TransposeInputLayout(name_node_map_[input->name], true)->output(0);
  Shape window_dimensions;
  window_dimensions.reserve(2);
  if (options->windowDimensions == nullptr ||
      options->windowDimensionsCount == 0) {
    auto shape = input_node.get_shape();
    if (shape.size() <= 1 || shape.size() > 4)
      return nullptr;
    size_t height_index = shape.size() == 2 ? 0 : shape.size() == 3 ? 1 : 2;
    window_dimensions.push_back(shape[height_index]);
    window_dimensions.push_back(shape[height_index + 1]);
  } else {
    window_dimensions.push_back(
        static_cast<size_t>(options->windowDimensions[0]));
    window_dimensions.push_back(
        static_cast<size_t>(options->windowDimensions[1]));
  }
  Shape pad_begin = {static_cast<size_t>(options->padding[0]),
                     static_cast<size_t>(options->padding[2])};
  Shape pad_end = {static_cast<size_t>(options->padding[1]),
                   static_cast<size_t>(options->padding[3])};
  Strides strides = {static_cast<size_t>(options->strides[0]),
                     static_cast<size_t>(options->strides[1])};
  Shape dilations = {static_cast<size_t>(options->dilations[0]),
                     static_cast<size_t>(options->dilations[1])};
  op::PadType auto_pad;
  switch (options->autoPad) {
    case SameUpper:
      auto_pad = op::PadType::SAME_UPPER;
      break;
    case SameLower:
      auto_pad = op::PadType::SAME_LOWER;
      break;
    default:
      auto_pad = op::PadType::EXPLICIT;
  }

  std::shared_ptr<ngraph::Node> pool2d_node;
  switch (type) {
    case ie_pool_type::AVERAGE_POOL:
      pool2d_node = std::make_shared<op::v1::AvgPool>(
          input_node, strides, pad_begin, pad_end, window_dimensions, true,
          op::RoundingType::FLOOR, auto_pad);
      break;
    case ie_pool_type::MAX_POOL:
      pool2d_node = std::make_shared<op::v1::MaxPool>(
          input_node, strides, pad_begin, pad_end, window_dimensions,
          op::RoundingType::FLOOR, auto_pad);
      break;
    default:
      assert(0);
  }

  auto node = options->layout == ie_input_operand_layout::Nhwc
                  ? TransposeInputLayout(pool2d_node->output(0), false)
                  : pool2d_node;
  std::string node_name = node->get_name();
  name_node_map_[node_name] = node->output(0);
  return CreateOperand(node_name);
}

ie_operand_t* Model::AddRelu(ie_operand_t* input) {
  auto input_node = name_node_map_[input->name];
  auto relu_node = std::make_shared<op::v0::Relu>(input_node);

  std::string node_name = relu_node->get_name();
  name_node_map_[node_name] = relu_node->output(0);
  return CreateOperand(node_name);
}

ie_operand_t* Model::AddReshape(ie_operand_t* input,
                                int32_t const* new_shape,
                                uint32_t new_shape_count) {
  auto input_node = name_node_map_[input->name];
  SizeVector shape = ToVector(new_shape, new_shape_count);
  auto reshape_node = Reshape(input_node, shape);

  std::string node_name = reshape_node.get_node()->get_name();
  name_node_map_[node_name] = reshape_node;
  return CreateOperand(node_name);
}

ie_operand_t* Model::AddSoftmax(ie_operand_t* input) {
  auto input_node = name_node_map_[input->name];
  // new Spec only define 2-D input tensor along axis 1.
  auto softmax_node = std::make_shared<op::v1::Softmax>(input_node, 1);

  std::string node_name = softmax_node->get_name();
  name_node_map_[node_name] = softmax_node->output(0);
  return CreateOperand(node_name);
}

ie_operand_t* Model::AddTranspose(ie_operand_t* input,
                                  ie_transpose_options_t* options) {
  auto input_node = name_node_map_[input->name];
  auto input_shape = input_node.get_shape();
  SizeVector permutation;
  permutation.reserve(input_shape.size());
  if (options->permutationCount == 0) {
    // When it’s not specified, it’s set to [N-1...0].
    for (int i = 0; i < input_shape.size(); ++i) {
      permutation.insert(permutation.begin(), i);
    }
  } else {
    permutation = ToVector(options->permutation, options->permutationCount);
  }
  const auto order_node = op::Constant::create(
      element::i64, Shape{permutation.size()}, permutation);
  auto transpose_node =
      std::make_shared<op::v1::Transpose>(input_node, order_node);

  std::string node_name = transpose_node->get_name();
  name_node_map_[node_name] = transpose_node->output(0);
  return CreateOperand(node_name);
}

ie_operand_t* Model::AddLeakyRelu(ie_operand_t* input,
                                  ie_leaky_relu_options_t* options) {
  auto input_node = name_node_map_[input->name];
  const auto alpha_node =
      op::Constant::create(element::f32, Shape{1}, {options->alpha});
  auto leaky_relu_node =
      std::make_shared<op::v0::PRelu>(input_node, alpha_node);

  std::string node_name = leaky_relu_node->get_name();
  name_node_map_[node_name] = leaky_relu_node->output(0);
  return CreateOperand(node_name);
}

ie_operand_t* Model::AddConcat(const ie_operand_t* inputs,
                               uint32_t inputs_count,
                               uint32_t axis) {
  ngraph::OutputVector inputs_vector;
  inputs_vector.reserve(inputs_count);
  for (uint32_t i = 0; i < inputs_count; ++i) {
    inputs_vector.push_back(name_node_map_[inputs[i].name]);
  }
  auto concat_node = std::make_shared<op::v0::Concat>(inputs_vector, axis);

  std::string node_name = concat_node->get_name();
  name_node_map_[node_name] = concat_node->output(0);
  return CreateOperand(node_name);
}

ie_operand_t* Model::AddGemm(const ie_operand_t* inputs,
                             uint32_t inputs_count,
                             const ie_gemm_options_t* options) {
  // The behavior of Gemm can be generically emulated from the usage of other
  // operations as "alpha * A * B + beta * C". Transpose if it's necessary.
  auto a_node = name_node_map_[inputs[0].name];
  auto b_node = name_node_map_[inputs[1].name];
  const auto order_node =
      op::Constant::create(element::i64, Shape{0}, SizeVector());
  if (options->aTranspose) {
    a_node = std::make_shared<op::v1::Transpose>(a_node, order_node)->output(0);
  }
  if (options->bTranspose) {
    b_node = std::make_shared<op::v1::Transpose>(b_node, order_node)->output(0);
  }
  std::shared_ptr<ngraph::Node> matmul_node =
      std::make_shared<op::v0::MatMul>(a_node, b_node);

  const auto alpha_node =
      op::Constant::create(element::f32, Shape{}, {options->alpha});
  if (options->alpha != 1) {
    matmul_node = std::make_shared<op::v1::Multiply>(matmul_node->output(0),
                                                     alpha_node->output(0));
  }

  const auto beta_node =
      op::Constant::create(element::f32, Shape{}, {options->beta});
  Output<ngraph::Node> c_node =
      inputs_count == 3 ? name_node_map_[inputs[2].name]
                        : op::Constant::create(element::f32, Shape{}, {0});
  auto beta_mul_c =
      std::make_shared<op::v1::Multiply>(beta_node->output(0), c_node);
  auto gemm_node = std::make_shared<op::v1::Add>(matmul_node->output(0),
                                                 beta_mul_c->output(0));
  std::string node_name = gemm_node->get_name();
  name_node_map_[node_name] = gemm_node->output(0);
  return CreateOperand(node_name);
}

void Model::Finish() {
  if (ngraph_inputs_.empty()) {
    THROW_IE_EXCEPTION << "The input must be set.";
  }

  auto ngraph_function =
      std::make_shared<Function>(ngraph_outputs_, ngraph_inputs_);
  network_ = std::make_unique<CNNNetwork>(ngraph_function);
  InputsDataMap input_info(network_->getInputsInfo());
  for (auto itr : input_info) {
    itr.second->setPrecision(Precision::FP32);
  }
  OutputsDataMap output_info(network_->getOutputsInfo());
  for (auto itr : output_info) {
    itr.second->setPrecision(Precision::FP32);
  }
}

size_t Model::GetOutputsNumber() {
  OutputsDataMap outputs = network_->getOutputsInfo();
  return outputs.size();
}

IEStatusCode Model::GetOutputName(const size_t number, char** name) {
  OutputsDataMap outputs = network_->getOutputsInfo();
  // check if the number is out of bounds.
  if (number < 0 || number >= outputs.size()) {
    return IEStatusCode::OUT_OF_BOUNDS;
  }
  OutputsDataMap::iterator iter = outputs.begin();
  for (size_t i = 0; i < number; ++i) {
    ++iter;
  }
  *name = new char[iter->first.length() + 1];
  memcpy(*name, iter->first.c_str(), iter->first.length() + 1);
  return IEStatusCode::OK;
}

}  // namespace InferenceEngine
