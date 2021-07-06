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

#include "ops/ReduceMean.h"

#include "Operand.h"
#include "Utils.h"

namespace node { namespace op {
    struct ReduceMeanOptions {
      public:
        std::vector<int32_t> axes;
        bool keepDimensions = false;

        const ml::ReduceMeanOptions* AsPtr() {
            if (!axes.empty()) {
                reduceMeanOptions.axesCount = axes.size();
                reduceMeanOptions.axes = axes.data();
            }
            reduceMeanOptions.keepDimensions = keepDimensions;

            return &reduceMeanOptions;
        }

      private:
        ml::ReduceMeanOptions reduceMeanOptions;
    };

    Napi::Value ReduceMean::Build(const Napi::CallbackInfo& info, ml::GraphBuilder builder) {
        // Operand ReduceMean(Operand const& input, ReduceMeanOptions const * options = nullptr)
        WEBNN_NODE_ASSERT(info.Length() == 1 || info.Length() == 2,
                          "The number of arguments is invalid.");

        std::vector<napi_value> args;
        ml::Operand input;
        WEBNN_NODE_ASSERT(GetOperand(info[0], input, args), "The input parameter is invalid.");

        // dictionary ReduceMeanOptions {
        //   sequence<int> axes;
        //   boolean keepDimensions;
        // };
        ReduceMeanOptions options;
        if (info.Length() == 2) {
            WEBNN_NODE_ASSERT(info[1].IsObject(), "The options must be an object.");
            Napi::Object jsOptions = info[1].As<Napi::Object>();
            if (HasOptionMember(jsOptions, "keepDimensions")) {
                WEBNN_NODE_ASSERT(
                    GetBoolean(jsOptions.Get("keepDimensions"), options.keepDimensions),
                    "The keepDimensions parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "axes")) {
                WEBNN_NODE_ASSERT(GetInt32Array(jsOptions.Get("axes"), options.axes, 1),
                                  "The keepDimensions parameter is invalid.");
            }
        }

        Napi::Object object = Operand::constructor.New(args);
        Operand* operand = Napi::ObjectWrap<Operand>::Unwrap(object);
        operand->SetImpl(builder.ReduceMean(input, options.AsPtr()));
        return object;
    }

}}  // namespace node::op
