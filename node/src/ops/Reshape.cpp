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

#include "ops/Reshape.h"

#include "Utils.h"

namespace node { namespace op {

    Napi::Value Reshape::Build(const Napi::CallbackInfo& info, ml::GraphBuilder builder) {
        // Operand reshape(Operand input, sequence<long> newShape);
        WEBNN_NODE_ASSERT(info.Length() == 2, "The number of arguments is invalid.");

        std::vector<napi_value> args;
        ml::Operand input;
        WEBNN_NODE_ASSERT(GetOperand(info[0], input, args), "The input parameter is invalid.");

        std::vector<int32_t> newShape;
        WEBNN_NODE_ASSERT(GetArray(info[1], newShape), "The newShape parameter is invalid.");
        WEBNN_NODE_ASSERT(newShape.empty() == false, "The newShape is empty.");

        ml::Operand reshape = builder.Reshape(input, newShape.data(), newShape.size());
        Napi::Object object = Operand::constructor.New(args);
        Operand* operand = Napi::ObjectWrap<Operand>::Unwrap(object);
        operand->SetImpl(reshape);
        return object;
    }

}}  // namespace node::op
