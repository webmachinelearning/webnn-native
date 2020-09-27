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
    node = std::make_shared<op::Constant>(element::f32, Shape(dims), dst);
  } else {
    int16_t* dst = blob->buffer().as<int16_t*>();
    CopyDataToBuffer<int16_t>(dst, src, length);
    node = std::make_shared<op::Constant>(element::f16, Shape(dims), dst);
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
      assert(0);
  }

  std::string node_name = binary_node->get_name();
  name_node_map_[node_name] = binary_node->output(0);
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

  auto input_node = name_node_map_[input->name];
  auto filter_node = name_node_map_[filter->name];
  auto conv2d_node = std::make_shared<op::v1::Convolution>(
      input_node, filter_node, strides, pad_begin, pad_end, dilations);

  std::string node_name = conv2d_node->get_name();
  name_node_map_[node_name] = conv2d_node->output(0);
  return CreateOperand(node_name);
}

ie_operand_t* Model::AddPool2d(ie_pool_type type,
                               ie_operand_t* input,
                               ie_pool2d_options_t* options) {
  // Use the height and width dimensions of the input shape as windowDimensions
  // if it is not present in options.
  // TODO: Transpose to NCHW if the layout is NHWC.
  auto input_node = name_node_map_[input->name];
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

  std::shared_ptr<ngraph::Node> pool2d_node;
  switch (type) {
    case ie_pool_type::AVERAGE_POOL:
      pool2d_node = std::make_shared<op::v1::AvgPool>(
          input_node, strides, pad_begin, pad_end, window_dimensions, true,
          op::RoundingType::FLOOR, op::PadType::EXPLICIT);
      break;
    case ie_pool_type::MAX_POOL:
      pool2d_node = std::make_shared<op::v1::MaxPool>(
          input_node, strides, pad_begin, pad_end, window_dimensions,
          op::RoundingType::FLOOR, op::PadType::EXPLICIT);
      break;
    default:
      assert(0);
  }

  std::string node_name = pool2d_node->get_name();
  name_node_map_[node_name] = pool2d_node->output(0);
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

void Model::Finish() {
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
