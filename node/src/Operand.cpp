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

#include "Operand.h"

#include "Utils.h"

Napi::FunctionReference node::Operand::constructor;

namespace node {

    Operand::Operand(const Napi::CallbackInfo& info) : Napi::ObjectWrap<Operand>(info) {
        for (size_t i = 0; i < info.Length(); ++i) {
            Napi::Object object = info[i].As<Napi::Object>();
            WEBNN_NODE_ASSERT_AND_RETURN(object.InstanceOf(Operand::constructor.Value()) ||
                                             object.InstanceOf(Operator::constructor.Value()),
                                         "The argument must be Operand or Operator object.");
            mObjects.push_back(Napi::Persistent(object));
        }
    }

    Napi::Object Operand::Initialize(Napi::Env env, Napi::Object exports) {
        Napi::HandleScope scope(env);
        Napi::Function func = DefineClass(env, "MLOperand", {});
        constructor = Napi::Persistent(func);
        constructor.SuppressDestruct();
        exports.Set("MLOperand", func);
        return exports;
    }

}  // namespace node
