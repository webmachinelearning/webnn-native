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

#include "ops/Split.h"

#include "Operand.h"
#include "Utils.h"

namespace node { namespace op {

    Napi::Value Split::Build(const Napi::CallbackInfo& info, ml::GraphBuilder builder) {
        // Operand Split(Operand input, (unsigned long or sequence<unsigned long>) splits,
        // MLSplitOptions options = {});
        WEBNN_NODE_ASSERT(info.Length() == 2 || info.Length() == 3,
                          "The number of arguments is invalid.");

        std::vector<napi_value> args;
        ml::Operand input;
        WEBNN_NODE_ASSERT(GetOperand(info[0], input, args), "The input parameter is invalid.");

        std::vector<uint32_t> splits;
        if (info[1].IsNumber()) {
            uint32_t temp_splits;
            WEBNN_NODE_ASSERT(GetValue(info[1], temp_splits), "The split parameter is invalid.");
            splits.push_back(temp_splits);
        } else {
            WEBNN_NODE_ASSERT(GetArray(info[1], splits), "The split parameter is invalid.");
            WEBNN_NODE_ASSERT(splits.empty() == false, "The split is empty.");
        }

        ml::SplitOptions options;
        if (info.Length() == 3) {
            WEBNN_NODE_ASSERT(info[2].IsObject(), "The options must be an object.");
            Napi::Object jsOptions = info[2].As<Napi::Object>();
            if (HasOptionMember(jsOptions, "axis")) {
                WEBNN_NODE_ASSERT(GetValue(jsOptions.Get("axis"), options.axis),
                                  "The axis parameter is invalid.");
            }
        }

        ml::OperandArray split = builder.Split(input, splits.data(), splits.size(), &options);
        size_t len = split.Size();
        Napi::Array objectArray = Napi::Array::New(info.Env(), len);
        for (size_t i = 0; i < len; i++) {
            Napi::Object object = Operand::constructor.New(args);
            Operand* operand = Napi::ObjectWrap<Operand>::Unwrap(object);
            operand->SetImpl(split.Get(i));
            objectArray[i] = object;
        }

        return objectArray;
    }

}}  // namespace node::op
