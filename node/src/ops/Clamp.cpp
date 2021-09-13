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

#include "ops/Clamp.h"

#include "Operand.h"
#include "Operator.h"
#include "Utils.h"

namespace node { namespace op {

    Napi::Value Clamp::Build(const Napi::CallbackInfo& info, ml::GraphBuilder builder) {
        // Operand clamp(Operand x, optional ClampOptions options = {});
        // Operator clamp(optional ClampOptions options = {});
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

        // dictionary ClampOptions {
        //   float minValue = std::numeric_limits<float>::lowest();
        //   float maxValue = std::numeric_limits<float>::max();
        // };
        ml::ClampOptions options;
        size_t argumentsCount = isFusedOperator ? 1 : 2;
        if (info.Length() == argumentsCount && !info[argumentsCount - 1].IsUndefined()) {
            WEBNN_NODE_ASSERT(info[argumentsCount - 1].IsObject(),
                              "The options must be an object.");
            Napi::Object jsOptions = info[argumentsCount - 1].As<Napi::Object>();
            if (HasOptionMember(jsOptions, "minValue")) {
                WEBNN_NODE_ASSERT(GetValue(jsOptions.Get("minValue"), options.minValue),
                                  "The minValue parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "maxValue")) {
                WEBNN_NODE_ASSERT(GetValue(jsOptions.Get("maxValue"), options.maxValue),
                                  "The maxValue parameter is invalid.");
            }
        }
        if (!isFusedOperator) {
            Napi::Object object = Operand::constructor.New(args);
            Operand* operand = Napi::ObjectWrap<Operand>::Unwrap(object);
            operand->SetImpl(builder.Clamp(input, &options));
            return object;
        } else {
            Napi::Object object = Operator::constructor.New(args);
            Operator* mlOperator = Napi::ObjectWrap<Operator>::Unwrap(object);
            mlOperator->SetImpl(builder.ClampOperator(&options));
            return object;
        }
    }

}}  // namespace node::op
