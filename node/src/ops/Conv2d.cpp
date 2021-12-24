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

#include "ops/Conv2d.h"

#include <iostream>
#include "Operand.h"
#include "Utils.h"

namespace node { namespace op {

    struct Conv2dOptions {
      public:
        std::vector<int32_t> padding;
        std::vector<int32_t> strides;
        std::vector<int32_t> dilations;
        std::vector<int32_t> outputPadding;
        std::vector<int32_t> outputSizes;
        bool transpose = false;
        int32_t groups = 1;
        ml::AutoPad autoPad = ml::AutoPad::Explicit;
        ml::InputOperandLayout inputLayout = ml::InputOperandLayout::Nchw;
        ml::FilterOperandLayout filterLayout = ml::FilterOperandLayout::Oihw;
        ml::Operand bias;
        ml::FusionOperator activation;

        const ml::Conv2dOptions* AsPtr() {
            if (!padding.empty()) {
                mOptions.paddingCount = padding.size();
                mOptions.padding = padding.data();
            }
            if (!strides.empty()) {
                mOptions.stridesCount = strides.size();
                mOptions.strides = strides.data();
            }
            if (!dilations.empty()) {
                mOptions.dilationsCount = dilations.size();
                mOptions.dilations = dilations.data();
            }
            if (!outputPadding.empty()) {
                mOptions.outputPaddingCount = outputPadding.size();
                mOptions.outputPadding = outputPadding.data();
            }
            if (!outputSizes.empty()) {
                mOptions.outputSizesCount = outputSizes.size();
                mOptions.outputSizes = outputSizes.data();
            }
            mOptions.transpose = transpose;
            mOptions.groups = groups;
            mOptions.autoPad = autoPad;
            mOptions.inputLayout = inputLayout;
            mOptions.filterLayout = filterLayout;
            mOptions.bias = bias;
            mOptions.activation = activation;
            return &mOptions;
        }

      private:
        ml::Conv2dOptions mOptions;
    };

    Napi::Value Conv2d::Build(const Napi::CallbackInfo& info, ml::GraphBuilder builder) {
        // Operand conv2d(Operand input, Operand filter, optional Conv2dOptions options = {});
        WEBNN_NODE_ASSERT(info.Length() == 2 || info.Length() == 3,
                          "The number of arguments is invalid.");

        std::vector<napi_value> args;
        ml::Operand input;
        WEBNN_NODE_ASSERT(GetOperand(info[0], input, args), "The input parameter is invalid.");
        ml::Operand filter;
        WEBNN_NODE_ASSERT(GetOperand(info[1], filter, args), "The filter parameter is invalid.");

        // dictionary Conv2dOptions {
        //   sequence<long> padding;
        //   sequence<long> strides;
        //   sequence<long> dilations;
        //   std::vector<int32_t> outputPadding;
        //   std::vector<int32_t> outputSizes;
        //   AutoPad autoPad = "explicit";
        //   bool transpose = false;
        //   long groups = 1;
        //   InputOperandLayout inputLayout = "nchw";
        //   FilterOperandLayout filterLayout = "oihw";
        //   Operand bias;
        //   Operator activation;
        // };
        Conv2dOptions options;
        if (info.Length() == 3 && !info[2].IsUndefined()) {
            WEBNN_NODE_ASSERT(info[2].IsObject(), "The options must be an object.");
            Napi::Object jsOptions = info[2].As<Napi::Object>();
            if (HasOptionMember(jsOptions, "padding")) {
                WEBNN_NODE_ASSERT(GetArray(jsOptions.Get("padding"), options.padding, 4),
                                  "The padding parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "strides")) {
                WEBNN_NODE_ASSERT(GetArray(jsOptions.Get("strides"), options.strides, 2),
                                  "The strides parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "dilations")) {
                WEBNN_NODE_ASSERT(GetArray(jsOptions.Get("dilations"), options.dilations, 2),
                                  "The dilations parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "outputPadding")) {
                WEBNN_NODE_ASSERT(
                    GetArray(jsOptions.Get("outputPadding"), options.outputPadding, 2),
                    "The outputPadding parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "outputSizes")) {
                WEBNN_NODE_ASSERT(GetArray(jsOptions.Get("outputSizes"), options.outputSizes, 2),
                                  "The outputSizes parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "autoPad")) {
                WEBNN_NODE_ASSERT(GetAutopad(jsOptions.Get("autoPad"), options.autoPad),
                                  "The autoPad parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "transpose")) {
                WEBNN_NODE_ASSERT(GetValue(jsOptions.Get("transpose"), options.transpose),
                                  "The transpose parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "groups")) {
                WEBNN_NODE_ASSERT(GetValue(jsOptions.Get("groups"), options.groups),
                                  "The groups parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "inputLayout")) {
                WEBNN_NODE_ASSERT(
                    GetInputOperandLayout(jsOptions.Get("inputLayout"), options.inputLayout),
                    "The inputLayout parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "filterLayout")) {
                WEBNN_NODE_ASSERT(
                    GetFilterOperandLayout(jsOptions.Get("filterLayout"), options.filterLayout),
                    "The filterLayout parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "bias")) {
                WEBNN_NODE_ASSERT(GetOperand(jsOptions.Get("bias"), options.bias, args),
                                  "The bias parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "activation")) {
                WEBNN_NODE_ASSERT(
                    GetOperator(jsOptions.Get("activation"), options.activation, args),
                    "The activation parameter is invalid.");
            }
        }
        Napi::Object object = Operand::constructor.New(args);
        Operand* operand = Napi::ObjectWrap<Operand>::Unwrap(object);
        operand->SetImpl(builder.Conv2d(input, filter, options.AsPtr()));
        return object;
    }

}}  // namespace node::op
