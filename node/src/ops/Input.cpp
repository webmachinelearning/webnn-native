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

#include "ops/Input.h"

#include "Utils.h"

namespace node { namespace op {

    Napi::Value Input::Build(const Napi::CallbackInfo& info, ml::GraphBuilder builder) {
        // Operand input(DOMString name, OperandDescriptor desc);
        WEBNN_NODE_ASSERT(info.Length() == 2, "The number of arguments is invalid.");
        std::string name;
        WEBNN_NODE_ASSERT(GetValue(info[0], name), "The name must be a string.");

        OperandDescriptor desc;
        WEBNN_NODE_ASSERT(GetOperandDescriptor(info[1], desc), "The desc parameter is invalid.");

        Napi::Object object = Operand::constructor.New({});
        Operand* operand = Napi::ObjectWrap<Operand>::Unwrap(object);
        operand->SetImpl(builder.Input(name.c_str(), desc.AsPtr()));
        return object;
    }

}}  // namespace node::op
