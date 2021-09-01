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

#include "Operator.h"

#include "Utils.h"

Napi::FunctionReference node::Operator::constructor;

namespace node {

    Operator::Operator(const Napi::CallbackInfo& info) : Napi::ObjectWrap<Operator>(info) {
        for (size_t i = 0; i < info.Length(); ++i) {
            Napi::Object operand = info[i].As<Napi::Object>();
            WEBNN_NODE_ASSERT_AND_RETURN(operand.InstanceOf(Operand::constructor.Value()),
                                         "The argument must be an operand object.");
            mOperands.push_back(Napi::Persistent(operand));
        }
    }

    Napi::Object Operator::Initialize(Napi::Env env, Napi::Object exports) {
        Napi::HandleScope scope(env);
        Napi::Function func = DefineClass(env, "MLOperator", {});
        constructor = Napi::Persistent(func);
        constructor.SuppressDestruct();
        exports.Set("MLOperator", func);
        return exports;
    }

}  // namespace node
