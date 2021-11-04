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

#include "ops/Resample2d.h"

#include "Utils.h"

namespace node { namespace op {

    Napi::Value Resample2d::Build(const Napi::CallbackInfo& info, ml::GraphBuilder builder) {
        // Operand Resample2d(Operand input, optional Resample2dOptions options = {});
        WEBNN_NODE_ASSERT(info.Length() == 1 || info.Length() == 2,
                          "The number of arguments is invalid.");

        std::vector<napi_value> args;
        ml::Operand input;
        WEBNN_NODE_ASSERT(GetOperand(info[0], input, args), "The input parameter is invalid.");

        // dictionary Resample2dOptions {
        //   InterpolationMode mode = "nearest neighbor"
        //   sequence<float> scales;
        //   sequence<long> sizes;
        //   sequence<long> axes;
        // };
        ml::Resample2dOptions options;
        std::vector<float> scales;
        std::vector<int32_t> sizes;
        std::vector<int32_t> axes;
        if (info.Length() == 2 && !info[1].IsUndefined()) {
            WEBNN_NODE_ASSERT(info[1].IsObject(), "The options must be an object.");
            Napi::Object jsOptions = info[1].As<Napi::Object>();
            if (HasOptionMember(jsOptions, "mode")) {
                WEBNN_NODE_ASSERT(GetInterpolationMode(jsOptions.Get("mode"), options.mode),
                                  "The mode parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "scales")) {
                WEBNN_NODE_ASSERT(GetArray(jsOptions.Get("scales"), scales),
                                  "The scales parameter is invalid.");
                WEBNN_NODE_ASSERT(scales.empty() == false, "The scales is empty.");
                options.scales = scales.data();
                options.scalesCount = scales.size();
            }
            if (HasOptionMember(jsOptions, "sizes")) {
                WEBNN_NODE_ASSERT(GetArray(jsOptions.Get("sizes"), sizes),
                                  "The sizes parameter is invalid.");
                WEBNN_NODE_ASSERT(sizes.empty() == false, "The sizes is empty.");
                options.sizes = sizes.data();
                options.sizesCount = sizes.size();
            }
            if (HasOptionMember(jsOptions, "axes")) {
                WEBNN_NODE_ASSERT(GetArray(jsOptions.Get("axes"), axes),
                                  "The axes parameter is invalid.");
                WEBNN_NODE_ASSERT(axes.empty() == false, "The axes is empty.");
                options.axes = axes.data();
                options.axesCount = axes.size();
            }
        }

        Napi::Object object = Operand::constructor.New(args);
        Operand* operand = Napi::ObjectWrap<Operand>::Unwrap(object);
        operand->SetImpl(builder.Resample2d(input, &options));
        return object;
    }

}}  // namespace node::op
