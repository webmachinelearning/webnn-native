// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ie_nn_c_api.h"

#include <inference_engine.hpp>
#include <map>
#include <memory>
#include <vector>

#include "ie_compilation.h"
#include "ie_model.h"
#include "utils.h"

namespace IE = InferenceEngine;

std::map<IE::StatusCode, IEStatusCode> status_map = {
    {IE::StatusCode::GENERAL_ERROR, IEStatusCode::GENERAL_ERROR},
    {IE::StatusCode::INFER_NOT_STARTED, IEStatusCode::INFER_NOT_STARTED},
    {IE::StatusCode::NETWORK_NOT_LOADED, IEStatusCode::NETWORK_NOT_LOADED},
    {IE::StatusCode::NETWORK_NOT_READ, IEStatusCode::NETWORK_NOT_READ},
    {IE::StatusCode::NOT_ALLOCATED, IEStatusCode::NOT_ALLOCATED},
    {IE::StatusCode::NOT_FOUND, IEStatusCode::NOT_FOUND},
    {IE::StatusCode::NOT_IMPLEMENTED, IEStatusCode::NOT_IMPLEMENTED},
    {IE::StatusCode::OK, IEStatusCode::OK},
    {IE::StatusCode::OUT_OF_BOUNDS, IEStatusCode::OUT_OF_BOUNDS},
    {IE::StatusCode::PARAMETER_MISMATCH, IEStatusCode::PARAMETER_MISMATCH},
    {IE::StatusCode::REQUEST_BUSY, IEStatusCode::REQUEST_BUSY},
    {IE::StatusCode::RESULT_NOT_READY, IEStatusCode::RESULT_NOT_READY},
    {IE::StatusCode::UNEXPECTED, IEStatusCode::UNEXPECTED}};

#define BEGINE_TRY try {
#define END_CATCH                                          \
  }                                                        \
  catch (const IE::details::InferenceEngineException& e) { \
    return e.hasStatus() ? status_map[e.getStatus()]       \
                         : IEStatusCode::UNEXPECTED;       \
  }                                                        \
  catch (...) {                                            \
    return IEStatusCode::UNEXPECTED;                       \
  }

/**
 * @struct ie_Compilation
 * @brief Create model from output operand and compile it for hardwave
 * accelerate including cpu/gpu/gna/vpu.
 */
struct ie_model {
  std::shared_ptr<IE::Model> object;
};

struct ie_compilation {
  std::unique_ptr<IE::Compilation> object;
};

IEStatusCode ie_create_model(ie_model** model) {
  if (model == nullptr) {
    return IEStatusCode::GENERAL_ERROR;
  }

  BEGINE_TRY
  *model = new ie_model_t;
  (*model)->object.reset(new IE::Model());
  END_CATCH

  return IEStatusCode::OK;
}

void ie_model_free(ie_model* model) {
  if (model) {
    delete model;
    model = NULL;
  }
}

IEStatusCode ie_model_add_constant(ie_model_t* model,
                                   ie_operand_descriptor_t const* desc,
                                   void const* value,
                                   size_t length,
                                   ie_operand_t** operand) {
  if (model == nullptr || operand == nullptr) {
    return IEStatusCode::GENERAL_ERROR;
  }

  BEGINE_TRY
  *operand = model->object->AddConstant(desc, value, length);
  END_CATCH

  return IEStatusCode::OK;
}

IEStatusCode ie_model_add_input(ie_model_t* model,
                                ie_operand_descriptor_t const* desc,
                                ie_operand_t** operand) {
  if (model == nullptr || desc == nullptr) {
    return IEStatusCode::GENERAL_ERROR;
  }

  BEGINE_TRY
  *operand = model->object->AddInput(desc);
  END_CATCH

  return IEStatusCode::OK;
}

IEStatusCode ie_model_add_output(ie_model_t* model, ie_operand_t* operand) {
  if (model == nullptr || operand == nullptr) {
    return IEStatusCode::GENERAL_ERROR;
  }

  BEGINE_TRY
  model->object->AddOutput(operand);
  END_CATCH

  return IEStatusCode::OK;
}

IEStatusCode ie_model_add_mat_mul(ie_model_t* model,
                                  ie_operand_t* a,
                                  ie_operand_t* b,
                                  ie_operand_t** operand) {
  if (model == nullptr || a == nullptr || b == nullptr) {
    return IEStatusCode::GENERAL_ERROR;
  }

  BEGINE_TRY
  *operand = model->object->AddMatMul(a, b);
  END_CATCH

  return IEStatusCode::OK;
}

IEStatusCode ie_model_add_batch_norm(ie_model_t* model,
                                     ie_operand_t* input,
                                     ie_operand_t* mean,
                                     ie_operand_t* variance,
                                     ie_batch_norm_options* options,
                                     ie_operand_t** operand) {
  if (model == nullptr || input == nullptr || mean == nullptr ||
      variance == nullptr) {
    return IEStatusCode::GENERAL_ERROR;
  }

  BEGINE_TRY
  *operand = model->object->AddBatchNorm(input, mean, variance, options);
  END_CATCH

  return IEStatusCode::OK;
}

IEStatusCode ie_model_add_binary(ie_model_t* model,
                                 ie_binary_type type,
                                 ie_operand_t* a,
                                 ie_operand_t* b,
                                 ie_operand_t** operand) {
  if (model == nullptr || a == nullptr || b == nullptr) {
    return IEStatusCode::GENERAL_ERROR;
  }

  BEGINE_TRY
  *operand = model->object->AddBinary(type, a, b);
  END_CATCH

  return IEStatusCode::OK;
}

IEStatusCode ie_model_add_clamp(ie_model_t* model,
                                ie_operand_t* input,
                                ie_clamp_options* options,
                                ie_operand_t** operand) {
  if (model == nullptr || input == nullptr) {
    return IEStatusCode::GENERAL_ERROR;
  }

  BEGINE_TRY
  *operand = model->object->AddClamp(input, options);
  END_CATCH

  return IEStatusCode::OK;
}

IEStatusCode ie_model_add_conv2d(ie_model_t* model,
                                 ie_operand_t* input,
                                 ie_operand_t* filter,
                                 ie_conv2d_options_t* options,
                                 ie_operand_t** operand) {
  if (model == nullptr || input == nullptr || filter == nullptr) {
    return IEStatusCode::GENERAL_ERROR;
  }

  BEGINE_TRY
  *operand = model->object->AddConv2d(input, filter, options);
  END_CATCH

  return IEStatusCode::OK;
}

IEStatusCode ie_model_add_pool2d(ie_model_t* model,
                                 ie_pool_type type,
                                 ie_operand_t* input,
                                 ie_pool2d_options_t* options,
                                 ie_operand_t** operand) {
  if (model == nullptr || input == nullptr) {
    return IEStatusCode::GENERAL_ERROR;
  }

  BEGINE_TRY
  *operand = model->object->AddPool2d(type, input, options);
  END_CATCH

  return IEStatusCode::OK;
}

IEStatusCode ie_model_add_relu(ie_model_t* model,
                               ie_operand_t* input,
                               ie_operand_t** operand) {
  if (model == nullptr || input == nullptr) {
    return IEStatusCode::GENERAL_ERROR;
  }

  BEGINE_TRY
  *operand = model->object->AddRelu(input);
  END_CATCH

  return IEStatusCode::OK;
}

IEStatusCode ie_model_add_reshape(ie_model_t* model,
                                  ie_operand_t* input,
                                  int32_t const* new_shape,
                                  uint32_t new_shape_count,
                                  ie_operand_t** operand) {
  if (model == nullptr || input == nullptr) {
    return IEStatusCode::GENERAL_ERROR;
  }

  BEGINE_TRY
  *operand = model->object->AddReshape(input, new_shape, new_shape_count);
  END_CATCH

  return IEStatusCode::OK;
}

IEStatusCode ie_model_add_softmax(ie_model_t* model,
                                  ie_operand_t* input,
                                  ie_operand_t** operand) {
  if (model == nullptr || input == nullptr) {
    return IEStatusCode::GENERAL_ERROR;
  }

  BEGINE_TRY
  *operand = model->object->AddSoftmax(input);
  END_CATCH

  return IEStatusCode::OK;
}

IEStatusCode ie_model_add_transpose(ie_model_t* model,
                                    ie_operand_t* input,
                                    ie_transpose_options* options,
                                    ie_operand_t** operand) {
  if (model == nullptr || input == nullptr) {
    return IEStatusCode::GENERAL_ERROR;
  }

  BEGINE_TRY
  *operand = model->object->AddTranspose(input, options);
  END_CATCH

  return IEStatusCode::OK;
}

IEStatusCode ie_model_add_leaky_relu(ie_model_t* model,
                                     ie_operand_t* input,
                                     ie_leaky_relu_options* options,
                                     ie_operand_t** operand) {
  if (model == nullptr || input == nullptr) {
    return IEStatusCode::GENERAL_ERROR;
  }

  BEGINE_TRY
  *operand = model->object->AddLeakyRelu(input, options);
  END_CATCH

  return IEStatusCode::OK;
}

IEStatusCode ie_model_add_concat(const ie_model_t* model,
                                 const ie_operand_t* inputs,
                                 uint32_t inputs_count,
                                 uint32_t axis,
                                 ie_operand_t** operand) {
  if (model == nullptr || inputs == nullptr || inputs_count == 0) {
    return IEStatusCode::GENERAL_ERROR;
  }

  BEGINE_TRY
  *operand = model->object->AddConcat(inputs, inputs_count, axis);
  END_CATCH

  return IEStatusCode::OK;
}

IEStatusCode ie_model_add_gemm(const ie_model_t* model,
                               const ie_operand_t* inputs,
                               uint32_t inputs_count,
                               const ie_gemm_options* options,
                               ie_operand_t** operand) {
  if (model == nullptr || inputs == nullptr || inputs_count == 0) {
    return IEStatusCode::GENERAL_ERROR;
  }

  BEGINE_TRY
  *operand = model->object->AddGemm(inputs, inputs_count, options);
  END_CATCH

  return IEStatusCode::OK;
}

void ie_operand_free(ie_operand_t* operand) {
  if (operand) {
    delete[] operand->name;
    delete operand;
    operand = NULL;
  }
}

IEStatusCode ie_model_finish(ie_model_t* model) {
  if (model == nullptr) {
    return IEStatusCode::GENERAL_ERROR;
  }

  BEGINE_TRY
  model->object->Finish();
  END_CATCH

  return IEStatusCode::OK;
}

IEStatusCode ie_model_get_outputs_number(const ie_model_t* model,
                                         size_t* size_result) {
  if (model == nullptr || size_result == nullptr) {
    return IEStatusCode::GENERAL_ERROR;
  }

  BEGINE_TRY
  *size_result = model->object->GetOutputsNumber();
  END_CATCH

  return IEStatusCode::OK;
}

IEStatusCode ie_model_get_output_name(const ie_model_t* model,
                                      const size_t number,
                                      char** name) {
  if (model == nullptr || name == nullptr) {
    return IEStatusCode::GENERAL_ERROR;
  }

  BEGINE_TRY
  IEStatusCode status;
  status = model->object->GetOutputName(number, name);
  if (status != IEStatusCode::OK)
    return status;
  END_CATCH

  return IEStatusCode::OK;
}

IEStatusCode ie_model_free_name(char** name) {
  if (*name) {
    delete[] * name;
    *name = NULL;
  }
  return IEStatusCode::OK;
}

IEStatusCode ie_create_compilation(ie_model* model,
                                   ie_compilation_t** compilation) {
  if (model == nullptr) {
    return IEStatusCode::GENERAL_ERROR;
  }

  BEGINE_TRY
  *compilation = new ie_compilation_t;
  (*compilation)->object.reset(new IE::Compilation(model->object));
  END_CATCH

  return IEStatusCode::OK;
}

void ie_compilation_free(ie_compilation_t* compilation) {
  if (compilation) {
    delete compilation;
    compilation = NULL;
  }
}

IEStatusCode ie_compilation_set_input(ie_compilation_t* compilation,
                                      ie_operand_t* operand,
                                      const void* buffer,
                                      uint32_t length) {
  if (compilation == nullptr) {
    return IEStatusCode::GENERAL_ERROR;
  }

  BEGINE_TRY
  compilation->object->SetInput(operand, buffer, length);
  END_CATCH

  return IEStatusCode::OK;
}

IEStatusCode ie_compilation_compute(ie_compilation_t* compilation) {
  if (compilation == nullptr) {
    return IEStatusCode::GENERAL_ERROR;
  }

  BEGINE_TRY
  compilation->object->Compute();
  END_CATCH

  return IEStatusCode::OK;
}

IEStatusCode ie_compilation_get_output(ie_compilation_t* compilation,
                                       ie_operand_t* operand,
                                       void* buffer,
                                       uint32_t length) {
  if (compilation == nullptr) {
    return IEStatusCode::GENERAL_ERROR;
  }

  BEGINE_TRY
  compilation->object->GetOutput(operand, buffer, length);
  END_CATCH

  return IEStatusCode::OK;
}

IEStatusCode ie_compilation_get_buffer(const ie_compilation_t* compilation,
                                       const char* name,
                                       void** buffer,
                                       size_t* byte_length) {
  if (compilation == nullptr || name == nullptr) {
    return IEStatusCode::GENERAL_ERROR;
  }

  BEGINE_TRY
  InferenceEngine::StatusCode status;
  status = compilation->object->GetBuffer(name, buffer, byte_length);
  if (status != InferenceEngine::StatusCode::OK)
    return status_map[status];
  END_CATCH
  return IEStatusCode::OK;
}

IEStatusCode ie_compilation_free_buffer(void** buffer) {
  if (*buffer) {
    free(*buffer);
    *buffer = NULL;
  }
  return IEStatusCode::OK;
}

IEStatusCode ie_compilation_get_dimensions(const ie_compilation_t* compilation,
                                           const char* name,
                                           ie_dimensions_t* dimensions) {
  if (compilation == nullptr) {
    return IEStatusCode::GENERAL_ERROR;
  }

  BEGINE_TRY
  compilation->object->GetDimensions(name, dimensions);
  END_CATCH

  return IEStatusCode::OK;
}

IEStatusCode ie_compilation_free_dimensions(ie_dimensions_t* dimensions) {
  if (dimensions) {
    free(dimensions->dims);
    dimensions->dims = nullptr;
  }
  return IEStatusCode::OK;
}
