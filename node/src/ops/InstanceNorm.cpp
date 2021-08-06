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

#include "ops/InstanceNorm.h"

#include "Operand.h"
#include "Utils.h"

namespace node { namespace op {

    Napi::Value InstanceNorm::Build(const Napi::CallbackInfo& info, ml::GraphBuilder builder) {
        // Operand instanceNormalization(Operand input, optional InstanceNormOptions options = {})
        WEBNN_NODE_ASSERT(info.Length() <= 2, "The number of arguments is invalid.");

        std::vector<napi_value> args;
        ml::Operand input;
        WEBNN_NODE_ASSERT(GetOperand(info[0], input, args), "The input parameter is invalid.");

        // dictionary InstanceNormalizationOptions {
        //   Operand scale;
        //   Operand bias;
        //   float epsilon = 1e-5;
        //   InputOperandLayout layout = InputOperandLayout::Nchw;
        // };
        ml::InstanceNormOptions options;
        if (info.Length() == 2 && !info[1].IsUndefined()) {
            WEBNN_NODE_ASSERT(info[1].IsObject(), "The options must be an object.")
            Napi::Object jsOptions = info[1].As<Napi::Object>();
            if (HasOptionMember(jsOptions, "scale")) {
                WEBNN_NODE_ASSERT(GetOperand(jsOptions.Get("scale"), options.scale, args),
                                  "The scale parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "bias")) {
                WEBNN_NODE_ASSERT(GetOperand(jsOptions.Get("bias"), options.bias, args),
                                  "The bias parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "epsilon")) {
                WEBNN_NODE_ASSERT(GetValue(jsOptions.Get("epsilon"), options.epsilon),
                                  "The epsilon parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "layout")) {
                WEBNN_NODE_ASSERT(GetInputOperandLayout(jsOptions.Get("layout"), options.layout),
                                  "The layout parameter is invalid.");
            }
        }

        Napi::Object object = Operand::constructor.New(args);
        Operand* operand = Napi::ObjectWrap<Operand>::Unwrap(object);
        operand->SetImpl(builder.InstanceNorm(input, &options));
        return object;
    }
}}  // namespace node::op
