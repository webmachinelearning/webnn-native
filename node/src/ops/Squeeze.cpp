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

#include "ops/Squeeze.h"

#include "Operand.h"
#include "Utils.h"

namespace node { namespace op {

    struct SqueezeOptions {
      public:
        std::vector<int32_t> axes;

        const wnn::SqueezeOptions* AsPtr() {
            if (!axes.empty()) {
                mOptions.axesCount = axes.size();
                mOptions.axes = axes.data();
            }
            return &mOptions;
        }

      private:
        wnn::SqueezeOptions mOptions;
    };

    Napi::Value Squeeze::Build(const Napi::CallbackInfo& info, wnn::GraphBuilder builder) {
        // Operand squeeze(Operand input, MLSqueezeOptions options = {})
        WEBNN_NODE_ASSERT(info.Length() == 1 || info.Length() == 2,
                          "The number of arguments is invalid.");

        std::vector<napi_value> args;
        wnn::Operand input;
        WEBNN_NODE_ASSERT(GetOperand(info[0], input, args), "The input parameter is invalid.");

        // dictionary SqueezeOptions {
        //   sequence<long> axes;
        // };
        SqueezeOptions options;
        if (info.Length() == 2) {
            WEBNN_NODE_ASSERT(info[1].IsObject(), "The options must be an object.");
            Napi::Object jsOptions = info[1].As<Napi::Object>();
            if (HasOptionMember(jsOptions, "axes")) {
                WEBNN_NODE_ASSERT(GetArray(jsOptions.Get("axes"), options.axes),
                                  "The axes parameter is invalid.");
            }
        }

        wnn::Operand squeeze = builder.Squeeze(input, options.AsPtr());
        Napi::Object object = Operand::constructor.New(args);
        Operand* operand = Napi::ObjectWrap<Operand>::Unwrap(object);
        operand->SetImpl(squeeze);
        return object;
    }

}}  // namespace node::op