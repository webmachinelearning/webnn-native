// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ngraph_c_api.h"

#include <c_api/ie_c_api.h>
#include <inference_engine.hpp>
#include <iostream>
#include <map>
#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/node.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/pass.hpp>
#include <vector>

#include "transpose_sinking.h"

#define TRY_IE_EXCEPTIONS try {
#define CATCH_IE_EXCEPTIONS                                               \
  }                                                                       \
  catch (const IE::Exception& e) {                                        \
    std::cout << "The Inference Engine error messages are : " << e.what() \
              << std::endl;                                               \
    return IEStatusCode::UNEXPECTED;                                      \
  }                                                                       \
  catch (std::exception & e) {                                            \
    std::cout << "The unexpected error message are : " << e.what()        \
              << std::endl;                                               \
    return IEStatusCode::UNEXPECTED;                                      \
  }

#define CREATE_NGRAPH_NODE(impl, node, index) \
  *node = new ngraph_node_t();                \
  (*node)->object = impl->output(index);      \
  return IEStatusCode::OK;

#define CREATE_NODE_AND_CATCH_EXCEPTIONS(impl, node) \
  CREATE_NGRAPH_NODE(impl, node, 0) CATCH_IE_EXCEPTIONS

#define CREATE_NODE_WITH_INDEX_AND_CATCH_EXCEPTIONS(impl, index, node) \
  CREATE_NGRAPH_NODE(impl, node, index) CATCH_IE_EXCEPTIONS

#define BUILD_BINARY(operation, a, b, node)                                  \
  TRY_IE_EXCEPTIONS                                                          \
  auto impl = std::make_shared<ngraph::op::operation>(a->object, b->object); \
  CREATE_NODE_AND_CATCH_EXCEPTIONS(impl, node)

#define BUILD_UNARY(operation, a, node)                           \
  TRY_IE_EXCEPTIONS                                               \
  auto impl = std::make_shared<ngraph::op::operation>(a->object); \
  CREATE_NODE_AND_CATCH_EXCEPTIONS(impl, node)

struct ngraph_node {
  ngraph::Output<ngraph::Node> object;
};

struct ngraph_function {
  std::shared_ptr<ngraph::Function> object;
};

std::map<ngraph::element::Type, precision_e> type_map = {
    {ngraph::element::f32, precision_e::FP32},
    {ngraph::element::f64, precision_e::FP16},
    {ngraph::element::i16, precision_e::I16},
    {ngraph::element::u8, precision_e::U8},
    {ngraph::element::i8, precision_e::I8},
    {ngraph::element::u16, precision_e::U16},
    {ngraph::element::i32, precision_e::I32},
    {ngraph::element::u32, precision_e::U32},
    {ngraph::element::i64, precision_e::I64},
    {ngraph::element::u64, precision_e::U64}};

namespace IE = InferenceEngine;

/**
 * @struct ie_network
 * @brief This is the main interface to describe the NN topology
 */
struct ie_network {
  IE::CNNNetwork object;
};

inline ngraph::element::Type get_tensor_type(const tensor_desc_t* tensorDesc) {
  ngraph::element::Type type;
  for (auto it : type_map) {
    if (it.second == tensorDesc->precision) {
      type = it.first;
      break;
    }
  }
  return type;
}

inline ngraph::Shape get_tensor_shape(const tensor_desc_t* tensorDesc) {
  IE::SizeVector dimensions;
  dimensions.reserve(tensorDesc->dims.ranks);
  for (size_t i = 0; i < tensorDesc->dims.ranks; ++i) {
    dimensions.push_back(tensorDesc->dims.dims[i]);
  }
  return ngraph::Shape(dimensions);
}

inline ngraph::op::PadType GetAutoPad(ngraph_auto_pad autoPad) {
  ngraph::op::PadType auto_pad = ngraph::op::PadType::EXPLICIT;
  switch (autoPad) {
    case SameUpper:
      auto_pad = ngraph::op::PadType::SAME_UPPER;
      break;
    case SameLower:
      auto_pad = ngraph::op::PadType::SAME_LOWER;
      break;
    case Explicit:
      auto_pad = ngraph::op::PadType::EXPLICIT;
      break;
    default:
      assert(0);
  }
  return auto_pad;
}

inline ngraph::op::RecurrentSequenceDirection get_recurrent_sequence_direction(
    ngraph_recurrent_sequence_direction direction) {
  ngraph::op::RecurrentSequenceDirection ngraph_direction;
  switch (direction) {
    case Forward:
      ngraph_direction = ngraph::op::RecurrentSequenceDirection::FORWARD;
      break;
    case Reverse:
      ngraph_direction = ngraph::op::RecurrentSequenceDirection::REVERSE;
      break;
    case Bidirectional:
      ngraph_direction = ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL;
      break;
    default:
      assert(0);
  }
  return ngraph_direction;
}

IEStatusCode ngraph_get_output_number(const ngraph_node_t* node,
                                      uint32_t* number) {
  TRY_IE_EXCEPTIONS
  auto node_shared_ptr = node->object.get_node_shared_ptr();
  *number = node_shared_ptr->get_output_size();
  CATCH_IE_EXCEPTIONS
  return IEStatusCode::OK;
}

IEStatusCode ngraph_get_output(const ngraph_node_t* input,
                               uint32_t index,
                               ngraph_node_t** node) {
  TRY_IE_EXCEPTIONS
  CREATE_NODE_WITH_INDEX_AND_CATCH_EXCEPTIONS(
      input->object.get_node_shared_ptr(), index, node);
}

IEStatusCode ngraph_get_index(const ngraph_node_t* node, size_t* index) {
  TRY_IE_EXCEPTIONS
  *index = node->object.get_index();
  CATCH_IE_EXCEPTIONS
  return IEStatusCode::OK;
}

IEStatusCode ngraph_get_shape(const ngraph_node_t* node,
                              dimensions_t* dimensions) {
  if (node == nullptr) {
    return IEStatusCode::GENERAL_ERROR;
  }
  IE::SizeVector shape = node->object.get_shape();
  dimensions->ranks = shape.size();
  for (size_t i = 0; i < dimensions->ranks; ++i) {
    dimensions->dims[i] = shape[i];
  }
  return IEStatusCode::OK;
}

IEStatusCode ngraph_get_name(const ngraph_node_t* node, char** name) {
  std::string node_name = node->object.get_node_shared_ptr()->get_name();
  *name = new char[node_name.length() + 1];
  memcpy(*name, node_name.c_str(), node_name.length() + 1);
  return IEStatusCode::OK;
}

IEStatusCode ngraph_input(const tensor_desc_t* tensorDesc,
                          ngraph_node_t** node) {
  if (tensorDesc == nullptr) {
    return IEStatusCode::GENERAL_ERROR;
  }

  TRY_IE_EXCEPTIONS
  auto input = std::make_shared<ngraph::op::v0::Parameter>(
      get_tensor_type(tensorDesc), get_tensor_shape(tensorDesc));
  CREATE_NODE_AND_CATCH_EXCEPTIONS(input, node);
  return IEStatusCode::OK;
}

void ngraph_node_free(ngraph_node_t** node) {
  if (node) {
    delete *node;
    *node = NULL;
  }
}

IEStatusCode ngraph_constant(const tensor_desc_t* tensorDesc,
                             const ie_blob_t* blob,
                             ngraph_node_t** node) {
  if (tensorDesc == nullptr || blob == nullptr) {
    return IEStatusCode::GENERAL_ERROR;
  }

  TRY_IE_EXCEPTIONS
  ie_blob_buffer_t buffer;
  auto status = ie_blob_get_buffer(blob, &buffer);
  if (status != IEStatusCode::OK) {
    return status;
  }
  auto constant = std::make_shared<ngraph::op::Constant>(
      get_tensor_type(tensorDesc), get_tensor_shape(tensorDesc), buffer.buffer);
  CREATE_NODE_AND_CATCH_EXCEPTIONS(constant, node);
}

IEStatusCode ngraph_add(const ngraph_node_t* a,
                        const ngraph_node_t* b,
                        ngraph_node_t** node) {
  BUILD_BINARY(v1::Add, a, b, node);
}

IEStatusCode ngraph_output(const ngraph_node_t* result, ngraph_node_t** node) {
  BUILD_UNARY(v0::Result, result, node);
}

IEStatusCode create_ngraph_function(ngraph_node_t** output,
                                    uint32_t output_count,
                                    ngraph_node_t** input,
                                    uint32_t input_count,
                                    ngraph_function_t** function) {
  std::vector<std::shared_ptr<ngraph::op::v0::Parameter>> ngraph_inputs;
  ngraph_inputs.reserve(input_count);
  for (size_t i = 0; i < input_count; ++i) {
    ngraph_inputs.push_back(std::shared_ptr<ngraph::op::v0::Parameter>(
        reinterpret_cast<ngraph::op::v0::Parameter*>(
            input[i]->object.get_node_shared_ptr().get())));
  }
  ngraph::OutputVector ngraph_nodes;
  ngraph_nodes.reserve(output_count);
  for (size_t i = 0; i < output_count; ++i) {
    ngraph_nodes.push_back(output[i]->object);
  }

  TRY_IE_EXCEPTIONS
  *function = new ngraph_function_t();
  (*function)->object =
      std::make_shared<ngraph::Function>(ngraph_nodes, ngraph_inputs);
  CATCH_IE_EXCEPTIONS
  return IEStatusCode::OK;
}

IEStatusCode transpose_sinking(ngraph_function_t* ngraph_function) {
  TRY_IE_EXCEPTIONS
  ngraph::pass::Manager passes;
  passes
      .register_pass<tensorflow::openvino_tensorflow::pass::TransposeSinking>();
  passes.run_passes(ngraph_function->object);
  CATCH_IE_EXCEPTIONS
  return IEStatusCode::OK;
}

IEStatusCode create_network(ngraph_function_t* ngraph_function,
                            ie_network_t** network) {
  *network = new ie_network_t();
  (*network)->object = IE::CNNNetwork(ngraph_function->object);
  return IEStatusCode::OK;
}

void ngraph_function_free(ngraph_function_t* ngraph_function) {
  if (ngraph_function) {
    delete ngraph_function;
    ngraph_function = nullptr;
  }
}

IEStatusCode ngraph_mul(const ngraph_node_t* a,
                        const ngraph_node_t* b,
                        ngraph_node_t** node) {
  BUILD_BINARY(v1::Multiply, a, b, node);
}

IEStatusCode ngraph_sub(const ngraph_node_t* a,
                        const ngraph_node_t* b,
                        ngraph_node_t** node) {
  BUILD_BINARY(v1::Subtract, a, b, node);
}

IEStatusCode ngraph_leaky_relu(const ngraph_node_t* a,
                               const ngraph_node_t* b,
                               ngraph_node_t** node) {
  BUILD_BINARY(v0::PRelu, a, b, node);
}

IEStatusCode ngraph_mat_mul(const ngraph_node_t* a,
                            const ngraph_node_t* b,
                            ngraph_node_t** node) {
  TRY_IE_EXCEPTIONS
  auto matmul = std::make_shared<ngraph::op::v0::MatMul>(a->object, b->object,
                                                         false, false);
  CREATE_NODE_AND_CATCH_EXCEPTIONS(matmul, node);
}

IEStatusCode ngraph_squeeze(const ngraph_node_t* a,
                            const ngraph_node_t* b,
                            ngraph_node_t** node) {
  if (b == nullptr) {
    BUILD_UNARY(v0::Squeeze, a, node);
  } else {
    BUILD_BINARY(v0::Squeeze, a, b, node);
  }
}

IEStatusCode ngraph_transpose(const ngraph_node_t* a,
                              const ngraph_node_t* b,
                              ngraph_node_t** node) {
  BUILD_BINARY(v1::Transpose, a, b, node);
}

IEStatusCode ngraph_reshape(const ngraph_node_t* a,
                            const ngraph_node_t* b,
                            ngraph_node_t** node) {
  TRY_IE_EXCEPTIONS
  auto reshape =
      std::make_shared<ngraph::op::v1::Reshape>(a->object, b->object, true);
  CREATE_NODE_AND_CATCH_EXCEPTIONS(reshape, node);
}

IEStatusCode ngraph_max(const ngraph_node_t* a,
                        const ngraph_node_t* b,
                        ngraph_node_t** node) {
  BUILD_BINARY(v1::Maximum, a, b, node);
}

IEStatusCode ngraph_min(const ngraph_node_t* a,
                        const ngraph_node_t* b,
                        ngraph_node_t** node) {
  BUILD_BINARY(v1::Minimum, a, b, node);
}

IEStatusCode ngraph_power(const ngraph_node_t* a,
                          const ngraph_node_t* b,
                          ngraph_node_t** node) {
  BUILD_BINARY(v1::Power, a, b, node);
}

IEStatusCode ngraph_divide(const ngraph_node_t* a,
                           const ngraph_node_t* b,
                           ngraph_node_t** node) {
  BUILD_BINARY(v1::Divide, a, b, node);
}

IEStatusCode ngraph_abs(const ngraph_node_t* a, ngraph_node_t** node) {
  BUILD_UNARY(v0::Abs, a, node);
}

IEStatusCode ngraph_ceil(const ngraph_node_t* a, ngraph_node_t** node) {
  BUILD_UNARY(v0::Ceiling, a, node);
}

IEStatusCode ngraph_cos(const ngraph_node_t* a, ngraph_node_t** node) {
  BUILD_UNARY(v0::Cos, a, node);
}

IEStatusCode ngraph_exp(const ngraph_node_t* a, ngraph_node_t** node) {
  BUILD_UNARY(v0::Exp, a, node);
}

IEStatusCode ngraph_floor(const ngraph_node_t* a, ngraph_node_t** node) {
  BUILD_UNARY(v0::Floor, a, node);
}

IEStatusCode ngraph_hard_swish(const ngraph_node_t* a, ngraph_node_t** node) {
  BUILD_UNARY(v4::HSwish, a, node);
}

IEStatusCode ngraph_log(const ngraph_node_t* a, ngraph_node_t** node) {
  BUILD_UNARY(v0::Log, a, node);
}

IEStatusCode ngraph_neg(const ngraph_node_t* a, ngraph_node_t** node) {
  BUILD_UNARY(v0::Negative, a, node);
}

IEStatusCode ngraph_relu(const ngraph_node_t* a, ngraph_node_t** node) {
  BUILD_UNARY(v0::Relu, a, node);
}

IEStatusCode ngraph_sigmoid(const ngraph_node_t* a, ngraph_node_t** node) {
  BUILD_UNARY(v0::Sigmoid, a, node);
}

IEStatusCode ngraph_sin(const ngraph_node_t* a, ngraph_node_t** node) {
  BUILD_UNARY(v0::Sin, a, node);
}

IEStatusCode ngraph_softmax(const ngraph_node_t* a, ngraph_node_t** node) {
  TRY_IE_EXCEPTIONS
  auto softmax = std::make_shared<ngraph::op::v1::Softmax>(a->object, 1);
  CREATE_NODE_AND_CATCH_EXCEPTIONS(softmax, node);
}

IEStatusCode ngraph_tan(const ngraph_node_t* a, ngraph_node_t** node) {
  BUILD_UNARY(v0::Tan, a, node);
}

IEStatusCode ngraph_tanh(const ngraph_node_t* a, ngraph_node_t** node) {
  BUILD_UNARY(v0::Tanh, a, node);
}

IEStatusCode ngraph_concat(ngraph_node_t** inputs,
                           uint32_t input_count,
                           uint32_t axis,
                           ngraph_node_t** node) {
  ngraph::OutputVector inputs_vector;
  inputs_vector.reserve(input_count);
  for (uint32_t i = 0; i < input_count; ++i) {
    inputs_vector.push_back(inputs[i]->object);
  }
  TRY_IE_EXCEPTIONS
  auto concat = std::make_shared<ngraph::op::v0::Concat>(inputs_vector, axis);
  CREATE_NODE_AND_CATCH_EXCEPTIONS(concat, node);
}

IEStatusCode ngraph_interpolate(const ngraph_node_t* input,
                                const ngraph_node_t* sizes,
                                const ngraph_node_t* scales,
                                const ngraph_node_t* axes,
                                const interpolate_attrs_t* attrs,
                                ngraph_node_t** node) {
  ngraph::op::v4::Interpolate::InterpolateAttrs ngraph_attrs;
  switch (attrs->mode) {
    case NearestNeighbor:
      ngraph_attrs.mode = ngraph::op::v4::Interpolate::InterpolateMode::nearest;
      break;
    case Linear:
      ngraph_attrs.mode = ngraph::op::v4::Interpolate::InterpolateMode::linear;
      break;
    default:
      assert(0);
      break;
  }
  switch (attrs->shape_calculation_mode) {
    case Sizes:
      ngraph_attrs.shape_calculation_mode =
          ngraph::op::v4::Interpolate::ShapeCalcMode::sizes;
      break;
    case Scales:
      ngraph_attrs.shape_calculation_mode =
          ngraph::op::v4::Interpolate::ShapeCalcMode::scales;
      break;
    default:
      assert(0);
      break;
  }
  TRY_IE_EXCEPTIONS
  auto resample = std::make_shared<ngraph::op::v4::Interpolate>(
      input->object, sizes->object, scales->object, axes->object, ngraph_attrs);
  CREATE_NODE_AND_CATCH_EXCEPTIONS(resample, node);
}

IEStatusCode ngraph_pad(const ngraph_node_t* input,
                        const ngraph_node_t* begin,
                        const ngraph_node_t* end,
                        const ngraph_node_t* value,
                        ngraph_padding_mode mode,
                        ngraph_node_t** node) {
  ngraph::op::PadMode pad_mode;
  switch (mode) {
    case ngraph_padding_mode::Edge:
      pad_mode = ngraph::op::PadMode::EDGE;
      break;
    case ngraph_padding_mode::Reflection:
      pad_mode = ngraph::op::PadMode::REFLECT;
      break;
    case ngraph_padding_mode::Symmetric:
      pad_mode = ngraph::op::PadMode::SYMMETRIC;
      break;
    case ngraph_padding_mode::Constant:
      pad_mode = ngraph::op::PadMode::CONSTANT;
      break;
    default:
      assert(0);
  }
  TRY_IE_EXCEPTIONS
  auto pad = std::make_shared<ngraph::op::v1::Pad>(
      input->object, begin->object, end->object, value->object, pad_mode);
  CREATE_NODE_AND_CATCH_EXCEPTIONS(pad, node);
}

IEStatusCode ngraph_reduce_l1(const ngraph_node_t* input,
                              const ngraph_node_t* axes,
                              bool keep_dimensions,
                              ngraph_node_t** node) {
  TRY_IE_EXCEPTIONS
  auto reduce_l1 = std::make_shared<ngraph::op::v4::ReduceL1>(
      input->object, axes->object, keep_dimensions);
  CREATE_NODE_AND_CATCH_EXCEPTIONS(reduce_l1, node);
}

IEStatusCode ngraph_reduce_l2(const ngraph_node_t* input,
                              const ngraph_node_t* axes,
                              bool keep_dimensions,
                              ngraph_node_t** node) {
  TRY_IE_EXCEPTIONS
  auto reduce_l2 = std::make_shared<ngraph::op::v4::ReduceL2>(
      input->object, axes->object, keep_dimensions);
  CREATE_NODE_AND_CATCH_EXCEPTIONS(reduce_l2, node);
}

IEStatusCode ngraph_reduce_max(const ngraph_node_t* input,
                               const ngraph_node_t* axes,
                               bool keep_dimensions,
                               ngraph_node_t** node) {
  TRY_IE_EXCEPTIONS
  auto reduce_max = std::make_shared<ngraph::op::v1::ReduceMax>(
      input->object, axes->object, keep_dimensions);
  CREATE_NODE_AND_CATCH_EXCEPTIONS(reduce_max, node);
}

IEStatusCode ngraph_reduce_mean(const ngraph_node_t* input,
                                const ngraph_node_t* axes,
                                bool keep_dimensions,
                                ngraph_node_t** node) {
  TRY_IE_EXCEPTIONS
  auto reduce_mean = std::make_shared<ngraph::op::v1::ReduceMean>(
      input->object, axes->object, keep_dimensions);
  CREATE_NODE_AND_CATCH_EXCEPTIONS(reduce_mean, node);
}

IEStatusCode ngraph_reduce_min(const ngraph_node_t* input,
                               const ngraph_node_t* axes,
                               bool keep_dimensions,
                               ngraph_node_t** node) {
  TRY_IE_EXCEPTIONS
  auto reduce_min = std::make_shared<ngraph::op::v1::ReduceMin>(
      input->object, axes->object, keep_dimensions);
  CREATE_NODE_AND_CATCH_EXCEPTIONS(reduce_min, node);
}

IEStatusCode ngraph_reduce_product(const ngraph_node_t* input,
                                   const ngraph_node_t* axes,
                                   bool keep_dimensions,
                                   ngraph_node_t** node) {
  TRY_IE_EXCEPTIONS
  auto reduce_product = std::make_shared<ngraph::op::v1::ReduceProd>(
      input->object, axes->object, keep_dimensions);
  CREATE_NODE_AND_CATCH_EXCEPTIONS(reduce_product, node);
}

IEStatusCode ngraph_reduce_sum(const ngraph_node_t* input,
                               const ngraph_node_t* axes,
                               bool keep_dimensions,
                               ngraph_node_t** node) {
  TRY_IE_EXCEPTIONS
  auto reduce_sum = std::make_shared<ngraph::op::v1::ReduceSum>(
      input->object, axes->object, keep_dimensions);
  CREATE_NODE_AND_CATCH_EXCEPTIONS(reduce_sum, node);
}

IEStatusCode ngraph_clamp(const ngraph_node_t* input,
                          float min,
                          float max,
                          ngraph_node_t** node) {
  TRY_IE_EXCEPTIONS
  auto clamp = std::make_shared<ngraph::op::v0::Clamp>(input->object, min, max);
  CREATE_NODE_AND_CATCH_EXCEPTIONS(clamp, node);
}

IEStatusCode ngraph_batch_norm_inference(const ngraph_node_t* input,
                                         const ngraph_node_t* scale,
                                         const ngraph_node_t* bias,
                                         const ngraph_node_t* mean,
                                         const ngraph_node_t* variance,
                                         double epsilon,
                                         ngraph_node_t** node) {
  TRY_IE_EXCEPTIONS
  auto batch_norm = std::make_shared<ngraph::op::v0::BatchNormInference>(
      input->object, scale->object, bias->object, mean->object,
      variance->object, epsilon);
  CREATE_NODE_AND_CATCH_EXCEPTIONS(batch_norm, node);
}

IEStatusCode ngraph_slice_inference(const ngraph_node_t* input,
                                    const ngraph_node_t* begin,
                                    const ngraph_node_t* end,
                                    ngraph_node_t** node) {
  TRY_IE_EXCEPTIONS
  std::vector<int64_t> begin_mask_ = {};
  std::vector<int64_t> end_mask_ = {};
  auto slice = std::make_shared<ngraph::op::v1::StridedSlice>(
      input->object, begin->object, end->object, begin_mask_, end_mask_);
  CREATE_NODE_AND_CATCH_EXCEPTIONS(slice, node);
}

IEStatusCode ngraph_average_pool(const ngraph_node_t* input,
                                 size_t const* strides,
                                 uint32_t strides_count,
                                 size_t const* padding,
                                 uint32_t padding_count,
                                 size_t const* dimensions,
                                 uint32_t dimensions_count,
                                 ngraph_auto_pad mode,
                                 ngraph_node_t** node) {
  ngraph::Strides strides_vector(strides, strides + strides_count);
  ngraph::Shape pad_begin = {padding[0], padding[2]};
  ngraph::Shape pad_end = {padding[1], padding[3]};
  ngraph::Shape window_dimensions(dimensions, dimensions + dimensions_count);
  TRY_IE_EXCEPTIONS
  auto pool2d = std::make_shared<ngraph::op::v1::AvgPool>(
      input->object, strides_vector, pad_begin, pad_end, window_dimensions,
      true, ngraph::op::RoundingType::FLOOR, GetAutoPad(mode));
  CREATE_NODE_AND_CATCH_EXCEPTIONS(pool2d, node);
}

IEStatusCode ngraph_l2_pool(const ngraph_node_t* input,
                            size_t const* strides,
                            uint32_t strides_count,
                            size_t const* padding,
                            uint32_t padding_count,
                            size_t const* dimensions,
                            uint32_t dimensions_count,
                            ngraph_auto_pad mode,
                            ngraph_node_t** node) {
  std::vector<float> constantBuffer = {2};
  auto constant = std::make_shared<ngraph::op::v0::Constant>(
      ngraph::element::f32, ngraph::Shape{}, constantBuffer);
  auto pow = std::make_shared<ngraph::op::v1::Power>(input->object,
                                                     constant->output(0));
  ngraph::Strides strides_vector(strides, strides + strides_count);
  ngraph::Shape pad_begin = {padding[0], padding[2]};
  ngraph::Shape pad_end = {padding[1], padding[3]};
  ngraph::Shape window_dimensions(dimensions, dimensions + dimensions_count);
  TRY_IE_EXCEPTIONS
  auto avgPool2d = std::make_shared<ngraph::op::v1::AvgPool>(
      pow->output(0), strides_vector, pad_begin, pad_end, window_dimensions,
      true, ngraph::op::RoundingType::FLOOR, GetAutoPad(mode));
  auto l2Pool2d = std::make_shared<ngraph::op::v0::Sqrt>(avgPool2d->output(0));
  CREATE_NODE_AND_CATCH_EXCEPTIONS(l2Pool2d, node);
}

IEStatusCode ngraph_max_pool(const ngraph_node_t* input,
                             size_t const* strides,
                             uint32_t strides_count,
                             size_t const* padding,
                             uint32_t padding_count,
                             size_t const* dimensions,
                             uint32_t dimensions_count,
                             ngraph_auto_pad mode,
                             ngraph_node_t** node) {
  ngraph::Strides strides_vector(strides, strides + strides_count);
  ngraph::Shape pad_begin = {padding[0], padding[2]};
  ngraph::Shape pad_end = {padding[1], padding[3]};
  ngraph::Shape window_dimensions(dimensions, dimensions + dimensions_count);
  TRY_IE_EXCEPTIONS
  auto pool2d = std::make_shared<ngraph::op::v1::MaxPool>(
      input->object, strides_vector, pad_begin, pad_end, window_dimensions,
      ngraph::op::RoundingType::FLOOR, GetAutoPad(mode));
  CREATE_NODE_AND_CATCH_EXCEPTIONS(pool2d, node);
}

IEStatusCode ngraph_convolution(const ngraph_node_t* input,
                                const ngraph_node_t* filter,
                                size_t const* strides,
                                uint32_t strides_count,
                                int32_t const* padding,
                                uint32_t padding_count,
                                size_t const* dilations,
                                uint32_t dilations_count,
                                ngraph_auto_pad mode,
                                ngraph_node_t** node) {
  ngraph::Strides strides_vector(strides, strides + strides_count);
  ngraph::CoordinateDiff pad_begin = {padding[0], padding[2]};
  ngraph::CoordinateDiff pad_end = {padding[1], padding[3]};
  ngraph::Strides dilations_vector(dilations, dilations + dilations_count);
  TRY_IE_EXCEPTIONS
  auto conv2d = std::make_shared<ngraph::op::v1::Convolution>(
      input->object, filter->object, strides_vector, pad_begin, pad_end,
      dilations_vector, GetAutoPad(mode));
  CREATE_NODE_AND_CATCH_EXCEPTIONS(conv2d, node);
}

IEStatusCode ngraph_convolution_backprop_data(const ngraph_node_t* input,
                                              const ngraph_node_t* filter,
                                              const ngraph_node_t* output_shape,
                                              size_t const* strides,
                                              uint32_t strides_count,
                                              int32_t const* padding,
                                              uint32_t padding_count,
                                              size_t const* dilations,
                                              uint32_t dilations_count,
                                              ngraph_auto_pad mode,
                                              int32_t const* output_padding,
                                              uint32_t output_padding_count,
                                              ngraph_node_t** node) {
  ngraph::Strides strides_vector(strides, strides + strides_count);
  ngraph::CoordinateDiff pad_begin = {padding[0], padding[2]};
  ngraph::CoordinateDiff pad_end = {padding[1], padding[3]};
  ngraph::op::PadType autoPad = GetAutoPad(mode);
  ngraph::Strides dilations_vector(dilations, dilations + dilations_count);
  ngraph::CoordinateDiff output_padding_vector(
      output_padding, output_padding + output_padding_count);
  std::shared_ptr<ngraph::Node> conv2d;

  TRY_IE_EXCEPTIONS
  if (output_shape == nullptr) {
    conv2d = std::make_shared<ngraph::op::v1::ConvolutionBackpropData>(
        input->object, filter->object, strides_vector, pad_begin, pad_end,
        dilations_vector, autoPad, output_padding_vector);
  } else {
    conv2d = std::make_shared<ngraph::op::v1::ConvolutionBackpropData>(
        input->object, filter->object, output_shape->object, strides_vector,
        pad_begin, pad_end, dilations_vector, autoPad, output_padding_vector);
  }
  CREATE_NODE_AND_CATCH_EXCEPTIONS(conv2d, node);
}

IEStatusCode ngraph_group_convolution(const ngraph_node_t* input,
                                      const ngraph_node_t* filter,
                                      size_t const* strides,
                                      uint32_t strides_count,
                                      int32_t const* padding,
                                      uint32_t padding_count,
                                      size_t const* dilations,
                                      uint32_t dilations_count,
                                      ngraph_auto_pad mode,
                                      ngraph_node_t** node) {
  ngraph::Strides strides_vector(strides, strides + strides_count);
  ngraph::CoordinateDiff pad_begin = {padding[0], padding[2]};
  ngraph::CoordinateDiff pad_end = {padding[1], padding[3]};
  ngraph::Strides dilations_vector(dilations, dilations + dilations_count);
  TRY_IE_EXCEPTIONS
  auto conv2d = std::make_shared<ngraph::op::v1::GroupConvolution>(
      input->object, filter->object, strides_vector, pad_begin, pad_end,
      dilations_vector, GetAutoPad(mode));
  CREATE_NODE_AND_CATCH_EXCEPTIONS(conv2d, node);
}

IEStatusCode ngraph_group_convolution_backprop_data(
    const ngraph_node_t* input,
    const ngraph_node_t* filter,
    const ngraph_node_t* output_shape,
    size_t const* strides,
    uint32_t strides_count,
    int32_t const* padding,
    uint32_t padding_count,
    size_t const* dilations,
    uint32_t dilations_count,
    ngraph_auto_pad mode,
    int32_t const* output_padding,
    uint32_t output_padding_count,
    ngraph_node_t** node) {
  ngraph::Strides strides_vector(strides, strides + strides_count);
  ngraph::CoordinateDiff pad_begin = {padding[0], padding[2]};
  ngraph::CoordinateDiff pad_end = {padding[1], padding[3]};
  ngraph::Strides dilations_vector(dilations, dilations + dilations_count);
  ngraph::CoordinateDiff output_padding_vector(
      output_padding, output_padding + output_padding_count);
  std::shared_ptr<ngraph::Node> conv2d;
  TRY_IE_EXCEPTIONS
  if (output_shape == nullptr) {
    conv2d = std::make_shared<ngraph::op::v1::GroupConvolutionBackpropData>(
        input->object, filter->object, strides_vector, pad_begin, pad_end,
        dilations_vector, GetAutoPad(mode), output_padding_vector);
  } else {
    conv2d = std::make_shared<ngraph::op::v1::GroupConvolutionBackpropData>(
        input->object, filter->object, output_shape->object, strides_vector,
        pad_begin, pad_end, dilations_vector, GetAutoPad(mode),
        output_padding_vector);
  }
  CREATE_NODE_AND_CATCH_EXCEPTIONS(conv2d, node);
}

IEStatusCode ngraph_split(const ngraph_node_t* input,
                          const ngraph_node_t* axis,
                          size_t num_splits,
                          ngraph_node_t** node) {
  TRY_IE_EXCEPTIONS
  auto split = std::make_shared<ngraph::op::v1::Split>(
      input->object, axis->object, num_splits);
  CREATE_NODE_AND_CATCH_EXCEPTIONS(split, node);
}

IEStatusCode ngraph_variadic_split(const ngraph_node_t* input,
                                   const ngraph_node_t* axis,
                                   const ngraph_node_t* splits,
                                   ngraph_node_t** node) {
  TRY_IE_EXCEPTIONS
  auto split = std::make_shared<ngraph::op::v1::VariadicSplit>(
      input->object, axis->object, splits->object);
  CREATE_NODE_AND_CATCH_EXCEPTIONS(split, node);
}

IEStatusCode ngraph_gru_sequence(const ngraph_node_t* input,
                                 const ngraph_node_t* initial_hidden_state,
                                 const ngraph_node_t* sequence_lengths,
                                 const ngraph_node_t* weight,
                                 const ngraph_node_t* recurrent_weight,
                                 const ngraph_node_t* bias,
                                 const size_t hidden_size,
                                 ngraph_recurrent_sequence_direction direction,
                                 const char* activations[2],
                                 const bool linear_before_reset,
                                 ngraph_node_t** node) {
  std::vector<std::string> activationsVector = {activations[0], activations[1]};
  const std::vector<float> activations_alpha_vector, activations_beta_vector;
  float clip = std::numeric_limits<float>::max();
  TRY_IE_EXCEPTIONS
  auto gru = std::make_shared<ngraph::op::v5::GRUSequence>(
      input->object, initial_hidden_state->object, sequence_lengths->object,
      weight->object, recurrent_weight->object, bias->object, hidden_size,
      get_recurrent_sequence_direction(direction), activationsVector,
      activations_alpha_vector, activations_beta_vector, clip,
      linear_before_reset);
  CREATE_NODE_AND_CATCH_EXCEPTIONS(gru, node);
}

// namespace IE = InferenceEngine;
