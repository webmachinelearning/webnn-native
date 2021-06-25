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

#ifndef IE_COMPILATION_H
#define IE_COMPILATION_H

#include <inference_engine.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_model.h"
#include "ie_nn_c_api.h"
#include "ngraph/node_output.hpp"
#include "ngraph/op/parameter.hpp"
#include "utils.h"

namespace InferenceEngine {

class Compilation {
 public:
  explicit Compilation(std::shared_ptr<Model> model);
  ~Compilation();

  StatusCode SetInput(ie_operand_t* operand,
                      const void* buffer,
                      uint32_t length);
  StatusCode GetOutput(ie_operand_t* operand, void* buffer, uint32_t length);
  StatusCode Compute();
  StatusCode GetBuffer(const char* name, void** buffer, size_t* byte_length);
  StatusCode GetDimensions(const char* name, ie_dimensions_t* dimensions);

 private:
  prefer_t preference_;

  std::unique_ptr<InferRequest> infer_request_;
  std::unique_ptr<ExecutableNetwork> execution_;
  std::unique_ptr<Core> ie_core_;
  std::map<std::string, std::string> output_node_map_;

  DISALLOW_COPY_AND_ASSIGN(Compilation);
};

}  // namespace InferenceEngine

#endif  // IE_COMPILATION_H
