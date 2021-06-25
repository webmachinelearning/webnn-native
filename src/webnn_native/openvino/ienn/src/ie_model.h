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

#ifndef IE_MODEL_H
#define IE_MODEL_H

#include <inference_engine.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_nn_c_api.h"
#include "ngraph/node_output.hpp"
#include "ngraph/op/parameter.hpp"
#include "utils.h"

namespace InferenceEngine {

class Model {
 public:
  Model() = default;
  ~Model() = default;

  ie_operand_t* AddConstant(ie_operand_descriptor_t const* desc,
                            void const* value,
                            size_t size);
  ie_operand_t* AddInput(ie_operand_descriptor_t const* desc);
  void AddOutput(ie_operand_t* operand);
  ie_operand_t* AddMatMul(ie_operand_t* a, ie_operand_t* b);
  ie_operand_t* AddBatchNorm(ie_operand_t* input,
                             ie_operand_t* mean,
                             ie_operand_t* variance,
                             ie_batch_norm_options_t* options);
  ie_operand_t* AddBinary(ie_binary_type type,
                          ie_operand_t* a,
                          ie_operand_t* b);
  ie_operand_t* AddClamp(ie_operand_t* input, ie_clamp_options_t* options);
  ie_operand_t* AddConv2d(ie_operand_t* input,
                          ie_operand_t* filter,
                          ie_conv2d_options_t* options);
  ie_operand_t* AddPool2d(ie_pool_type type,
                          ie_operand_t* input,
                          ie_pool2d_options_t* options);
  ie_operand_t* AddRelu(ie_operand_t* input);
  ie_operand_t* AddReshape(ie_operand_t* input,
                           int32_t const* new_shape,
                           uint32_t new_shape_count);
  ie_operand_t* AddSoftmax(ie_operand_t* input);
  ie_operand_t* AddTranspose(ie_operand_t* input,
                             ie_transpose_options_t* options);
  ie_operand_t* AddLeakyRelu(ie_operand_t* input,
                             ie_leaky_relu_options_t* options);
  ie_operand_t* AddConcat(const ie_operand_t* inputs,
                          uint32_t inputs_count,
                          uint32_t axis);
  ie_operand_t* AddGemm(const ie_operand_t* inputs,
                        uint32_t inputs_count,
                        const ie_gemm_options_t* options);
  void Finish();
  size_t GetOutputsNumber();
  IEStatusCode GetOutputName(const size_t number, char** name);

 private:
  friend class Compilation;
  std::map<std::string, ngraph::Output<ngraph::Node>> name_node_map_;
  std::vector<std::shared_ptr<ngraph::op::v0::Parameter>> ngraph_inputs_;
  std::vector<std::shared_ptr<ngraph::op::v0::Result>> ngraph_outputs_;
  std::unique_ptr<CNNNetwork> network_;
  std::map<std::string, std::string> output_node_map_;

  DISALLOW_COPY_AND_ASSIGN(Model);
};

}  // namespace InferenceEngine

#endif  // IE_MODEL_H
