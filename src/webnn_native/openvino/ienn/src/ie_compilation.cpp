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

#include "ie_compilation.h"

#include <gna/gna_config.hpp>
#include <ngraph/ngraph.hpp>
#include <ngraph/node.hpp>
#include <string>
#include <utility>

#include "utils.h"

namespace InferenceEngine {

Compilation::Compilation(std::shared_ptr<Model> model)
    : preference_(PREFER_FAST_SINGLE_ANSWER) {
  std::string device_name;
  if (preference_ == prefer_t::PREFER_FAST_SINGLE_ANSWER) {
    device_name = "CPU";
  } else if (preference_ == prefer_t::PREFER_SUSTAINED_SPEED) {
    device_name = "GPU";
  } else if (preference_ == prefer_t::PREFER_LOW_POWER) {
    device_name = "MYRIAD";
  } else if (preference_ == prefer_t::PREFER_ULTRA_LOW_POWER) {
    device_name = "GNA";
  }
  ie_core_.reset(new Core());
  std::map<std::string, std::string> plugin_Config = {};
  if (preference_ == prefer_t::PREFER_ULTRA_LOW_POWER) {
    // TODO(Junwei): The SCALE_FACTOR need to be set.
    plugin_Config[GNAConfigParams::KEY_GNA_DEVICE_MODE] = "GNA_AUTO";
    // Note that it is not always possible to use 8-bit weights due to GNA
    // hardware limitations. For example, convolutional layers always use
    // 16-bit weights (GNA harware verison 1 and 2). This limitation will be
    // removed in GNA hardware version 3 and higher.
    // gnaPluginConfig[GNAConfigParams::KEY_GNA_PRECISION] = "I8";
  }
  execution_.reset(new ExecutableNetwork(static_cast<IExecutableNetwork::Ptr&>(
      ie_core_->LoadNetwork(*(model->network_), device_name, plugin_Config))));
  infer_request_.reset(new InferRequest(
      static_cast<IInferRequest::Ptr>(execution_->CreateInferRequest())));
}

Compilation::~Compilation() {
  // Release in squence to avoid crash.
  infer_request_.reset(nullptr);
  execution_.reset(nullptr);
  ie_core_.reset(nullptr);
}

StatusCode Compilation::SetInput(ie_operand_t* operand,
                                 const void* buffer,
                                 uint32_t length) {
  if (infer_request_ == nullptr) {
    return StatusCode::NETWORK_NOT_LOADED;
  }

  Blob::Ptr input_blob = infer_request_->GetBlob(operand->name);
  memcpy(input_blob->buffer(), buffer, length);

  return StatusCode::OK;
}

StatusCode Compilation::GetOutput(ie_operand_t* operand,
                                  void* buffer,
                                  uint32_t length) {
  if (infer_request_ == nullptr) {
    return StatusCode::NETWORK_NOT_LOADED;
  }

  Blob::Ptr output_blob = infer_request_->GetBlob(operand->name);
  if (output_blob->byteSize() != length) {
    THROW_IE_EXCEPTION << "The output buffer length is invalid.";
  }
  memcpy(buffer, output_blob->buffer(), length);

  return StatusCode::OK;
}

StatusCode Compilation::GetBuffer(const char* name,
                                  void** buffer,
                                  size_t* byte_length) {
  if (infer_request_ == nullptr) {
    return StatusCode::NETWORK_NOT_LOADED;
  }
  Blob::Ptr output_blob = infer_request_->GetBlob(name);
  *byte_length = output_blob->byteSize();
  *buffer = malloc(*byte_length);
  memcpy(*buffer, output_blob->buffer(), *byte_length);

  return StatusCode::OK;
}

StatusCode Compilation::GetDimensions(const char* name,
                                      ie_dimensions_t* dimensions) {
  if (infer_request_ == nullptr) {
    return StatusCode::NETWORK_NOT_LOADED;
  }
  Blob::Ptr output_blob = infer_request_->GetBlob(name);
  InferenceEngine::SizeVector dims = output_blob->getTensorDesc().getDims();
  dimensions->ranks = dims.size();
  dimensions->dims = (int32_t*)malloc(dimensions->ranks * sizeof(int32_t));
  for (size_t i = 0; i < dimensions->ranks; ++i) {
    dimensions->dims[i] = dims[i];
  }

  return StatusCode::OK;
}

StatusCode Compilation::Compute() {
  if (infer_request_ == nullptr) {
    return StatusCode::NETWORK_NOT_LOADED;
  }
  infer_request_->Infer();

  return StatusCode::OK;
}

}  // namespace InferenceEngine
