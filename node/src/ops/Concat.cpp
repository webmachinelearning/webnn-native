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

#include "ops/Concat.h"

#include "Operand.h"
#include "Utils.h"

namespace node { namespace op {

    Napi::Value Concat::Build(const Napi::CallbackInfo& info, ml::GraphBuilder builder) {
        // Operand concat(sequence<Operand> inputs, long axis);
        WEBNN_NODE_ASSERT(info.Length() == 2, "The number of arguments is invalid.");

        std::vector<napi_value> args;
        std::vector<ml::Operand> inputs;
        WEBNN_NODE_ASSERT(GetOperandArray(info[0], inputs, args),
                          "The input operands are invalid.");
        int32_t axis;
        WEBNN_NODE_ASSERT(GetValue(info[1], axis), "The axis parameter is invalid.");
        Napi::Object object = Operand::constructor.New(args);
        Operand* operand = Napi::ObjectWrap<Operand>::Unwrap(object);
        operand->SetImpl(builder.Concat(inputs.size(), inputs.data(), axis));
        return object;
    }
}}  // namespace node::op
