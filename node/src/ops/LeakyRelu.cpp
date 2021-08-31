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

#include "ops/LeakyRelu.h"

#include "Utils.h"

namespace node { namespace op {

    Napi::Value LeakyRelu::Build(const Napi::CallbackInfo& info, ml::GraphBuilder builder) {
        // Operand leakyRelu(Operand x, optional LeakyReluOptions options = {});
        // Operator leakyReluOperator(optional LeakyReluOptions options = {});
        std::vector<napi_value> args;
        ml::Operand input;
        bool isFusedOperator =
            info.Length() == 0 ||
            (info.Length() == 1 && info[0].IsObject() &&
             !info[0].As<Napi::Object>().InstanceOf(Operand::constructor.Value()));
        if (!isFusedOperator) {
            WEBNN_NODE_ASSERT(info.Length() == 1 || info.Length() == 2,
                              "The number of arguments is invalid.");
            WEBNN_NODE_ASSERT(GetOperand(info[0], input, args), "The input parameter is invalid.");
        } else {
            WEBNN_NODE_ASSERT(info.Length() == 0 || info.Length() == 1,
                              "The number of arguments is invalid.");
        }

        // dictionary LeakyReluOptions {
        //   float alpha = 0.01;
        // };
        ml::LeakyReluOptions options;
        size_t argumentsCount = isFusedOperator ? 1 : 2;
        if (info.Length() == argumentsCount && !info[argumentsCount - 1].IsUndefined()) {
            WEBNN_NODE_ASSERT(info[argumentsCount - 1].IsObject(),
                              "The options must be an object.");
            Napi::Object jsOptions = info[argumentsCount - 1].As<Napi::Object>();
            if (HasOptionMember(jsOptions, "alpha")) {
                WEBNN_NODE_ASSERT(GetValue(jsOptions.Get("alpha"), options.alpha),
                                  "The alpha parameter is invalid.");
            }
        }
        if (!isFusedOperator) {
            Napi::Object object = Operand::constructor.New(args);
            Operand* operand = Napi::ObjectWrap<Operand>::Unwrap(object);
            operand->SetImpl(builder.LeakyRelu(input, &options));
            return object;
        } else {
            Napi::Object object = Operator::constructor.New(args);
            Operator* mlOperator = Napi::ObjectWrap<Operator>::Unwrap(object);
            mlOperator->SetImpl(builder.LeakyReluOperator(&options));
            return object;
        }
    }
}}  // namespace node::op