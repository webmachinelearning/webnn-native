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

#include "ops/Slice.h"

#include "Operand.h"
#include "Utils.h"

namespace node { namespace op {

    struct SliceOptions {
      public:
        std::vector<int32_t> axes;

        const ml::SliceOptions* AsPtr() {
            if (!axes.empty()) {
                mOptions.axesCount = axes.size();
                mOptions.axes = axes.data();
            }
            return &mOptions;
        }

      private:
        ml::SliceOptions mOptions;
    };

    Napi::Value Slice::Build(const Napi::CallbackInfo& info, ml::GraphBuilder builder) {
        // Operand slice(Operand input, sequence<long> starts, sequence<long> sizes,
        // MLSliceOptions options = {})
        WEBNN_NODE_ASSERT(info.Length() == 3 || info.Length() == 4,
                          "The number of arguments is invalid.");

        std::vector<napi_value> args;
        ml::Operand input;
        WEBNN_NODE_ASSERT(GetOperand(info[0], input, args), "The input parameter is invalid.");

        std::vector<int32_t> starts;
        WEBNN_NODE_ASSERT(GetArray(info[1], starts), "The starts parameter is invalid.");
        WEBNN_NODE_ASSERT(starts.empty() == false, "The starts is empty.");

        std::vector<int32_t> sizes;
        WEBNN_NODE_ASSERT(GetArray(info[2], sizes), "The sizes parameter is invalid.");
        WEBNN_NODE_ASSERT(sizes.empty() == false, "The sizes is empty.");

        // dictionary SliceOptions {
        //   sequence<long> axes;
        // };
        SliceOptions options;
        if (info.Length() == 4) {
            WEBNN_NODE_ASSERT(info[3].IsObject(), "The options must be an object.");
            Napi::Object jsOptions = info[3].As<Napi::Object>();
            if (HasOptionMember(jsOptions, "axes")) {
                WEBNN_NODE_ASSERT(GetArray(jsOptions.Get("axes"), options.axes),
                                  "The axes parameter is invalid.");
            }
        }

        ml::Operand slice = builder.Slice(input, starts.data(), starts.size(), sizes.data(),
                                          sizes.size(), options.AsPtr());
        Napi::Object object = Operand::constructor.New(args);
        Operand* operand = Napi::ObjectWrap<Operand>::Unwrap(object);
        operand->SetImpl(slice);
        return object;
    }

}}  // namespace node::op