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

#include "ops/Pad.h"

#include "Utils.h"

namespace node { namespace op {

    Napi::Value Pad::Build(const Napi::CallbackInfo& info, ml::GraphBuilder builder) {
        // Operand pad(Operand input, Operand padding, optional PadOptions options = {});
        WEBNN_NODE_ASSERT(info.Length() == 2 || info.Length() == 3,
                          "The number of arguments is invalid.");

        std::vector<napi_value> args;
        ml::Operand input;
        WEBNN_NODE_ASSERT(GetOperand(info[0], input, args), "The input parameter is invalid.");
        ml::Operand padding;
        WEBNN_NODE_ASSERT(GetOperand(info[1], padding, args), "The padding parameter is invalid.");

        // dictionary PadOptions {
        //   PaddingMode mode;
        //   float value = false;
        // };
        ml::PadOptions options;
        if (info.Length() == 3 && !info[2].IsUndefined()) {
            WEBNN_NODE_ASSERT(info[2].IsObject(), "The options must be an object.");
            Napi::Object jsOptions = info[2].As<Napi::Object>();
            if (HasOptionMember(jsOptions, "mode")) {
                WEBNN_NODE_ASSERT(GetPaddingMode(jsOptions.Get("mode"), options.mode),
                                  "The mode parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "value")) {
                WEBNN_NODE_ASSERT(GetValue(jsOptions.Get("value"), options.value),
                                  "The value parameter is invalid.");
            }
        }

        Napi::Object object = Operand::constructor.New(args);
        Operand* operand = Napi::ObjectWrap<Operand>::Unwrap(object);
        operand->SetImpl(builder.Pad(input, padding, &options));
        return object;
    }

}}  // namespace node::op
