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

#ifndef SERVICES_ML_IENN_SYMBOL_TABLE_H_
#define SERVICES_ML_IENN_SYMBOL_TABLE_H_

#include "late_binding_symbol_table.h"

namespace webnn_native {

// The ienn symbols we need, as an X-Macro list.
#define IE_SYMBOLS_LIST            \
  X(ie_create_model)               \
  X(ie_model_free)                 \
  X(ie_model_add_constant)         \
  X(ie_model_add_input)            \
  X(ie_model_add_output)           \
  X(ie_model_add_mat_mul)          \
  X(ie_operand_free)               \
  X(ie_model_finish)               \
  X(ie_create_compilation)         \
  X(ie_compilation_free)           \
  X(ie_compilation_set_input)      \
  X(ie_compilation_compute)        \
  X(ie_compilation_get_output)     \
  X(ie_model_add_batch_norm)       \
  X(ie_model_add_binary)           \
  X(ie_model_add_clamp)            \
  X(ie_model_add_conv2d)           \
  X(ie_model_add_gemm)             \
  X(ie_model_add_pool2d)           \
  X(ie_model_add_relu)             \
  X(ie_model_add_reshape)          \
  X(ie_model_add_softmax)          \
  X(ie_model_add_transpose)        \
  X(ie_model_add_leaky_relu)       \
  X(ie_model_add_concat)           \
  X(ie_model_get_outputs_number)   \
  X(ie_model_get_output_name)      \
  X(ie_model_free_name)            \
  X(ie_compilation_get_buffer)     \
  X(ie_compilation_free_buffer)    \
  X(ie_compilation_get_dimensions) \
  X(ie_compilation_free_dimensions)

LATE_BINDING_SYMBOL_TABLE_DECLARE_BEGIN(IESymbolTable)
#define X(sym) LATE_BINDING_SYMBOL_TABLE_DECLARE_ENTRY(IESymbolTable, sym)
IE_SYMBOLS_LIST
#undef X
LATE_BINDING_SYMBOL_TABLE_DECLARE_END(IESymbolTable)

IESymbolTable* GetIESymbolTable();

#if defined(_WIN32) || defined(_WIN64) || defined(__linux__)
#define IE(sym) LATESYM_GET(IESymbolTable, GetIESymbolTable(), sym)
#else
#define IE(sym) sym
#endif

}  // namespace webnn_native

#endif  // SERVICES_ML_IENN_SYMBOL_TABLE_H_
