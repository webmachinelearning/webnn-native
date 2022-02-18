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

namespace node { namespace op {

    template <typename T>
    Napi::Value GetConv2dBaseOptions(const Napi::CallbackInfo& info,
                                     std::vector<napi_value>& args,
                                     wnn::Operand& input,
                                     wnn::Operand& filter,
                                     T& options,
                                     Napi::Object& jsOptions) {
        // Operand conv2d(Operand input, Operand filter, optional Conv2dOptions options = {});
        WEBNN_NODE_ASSERT(info.Length() == 2 || info.Length() == 3,
                          "The number of arguments is invalid.");
        WEBNN_NODE_ASSERT(GetOperand(info[0], input, args), "The input parameter is invalid.");
        WEBNN_NODE_ASSERT(GetOperand(info[1], filter, args), "The filter parameter is invalid.");

        if (info.Length() == 3 && !info[2].IsUndefined()) {
            WEBNN_NODE_ASSERT(info[2].IsObject(), "The options must be an object.");
            jsOptions = info[2].As<Napi::Object>();
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
            if (HasOptionMember(jsOptions, "autoPad")) {
                WEBNN_NODE_ASSERT(GetAutopad(jsOptions.Get("autoPad"), options.autoPad),
                                  "The autoPad parameter is invalid.");
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
        return {};
    }

    Napi::Value Conv2d::Build(const Napi::CallbackInfo& info, wnn::GraphBuilder builder) {
        std::vector<napi_value> args;
        wnn::Operand input;
        wnn::Operand filter;
        Conv2dOptions options;
        Napi::Object jsOptions;

        GetConv2dBaseOptions<Conv2dOptions>(info, args, input, filter, options, jsOptions);
        if (info.Length() == 3 && !info[2].IsUndefined()) {
            if (HasOptionMember(jsOptions, "filterLayout")) {
                WEBNN_NODE_ASSERT(GetConv2dFilterOperandLayout(jsOptions.Get("filterLayout"),
                                                               options.filterLayout),
                                  "The filterLayout parameter is invalid.");
            }
        }
        Napi::Object object = Operand::constructor.New(args);
        Operand* operand = Napi::ObjectWrap<Operand>::Unwrap(object);
        operand->SetImpl(builder.Conv2d(input, filter, options.AsPtr()));

        return object;
    }

    Napi::Value ConvTranspose2d::Build(const Napi::CallbackInfo& info, wnn::GraphBuilder builder) {
        std::vector<napi_value> args;
        wnn::Operand input;
        wnn::Operand filter;
        ConvTranspose2dOptions options;
        Napi::Object jsOptions;

        GetConv2dBaseOptions<ConvTranspose2dOptions>(info, args, input, filter, options, jsOptions);
        if (info.Length() == 3 && !info[2].IsUndefined()) {
            if (HasOptionMember(jsOptions, "outputPadding")) {
                WEBNN_NODE_ASSERT(
                    GetArray(jsOptions.Get("outputPadding"), options.outputPadding, 2),
                    "The outputPadding parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "outputSizes")) {
                WEBNN_NODE_ASSERT(GetArray(jsOptions.Get("outputSizes"), options.outputSizes, 2),
                                  "The outputSizes parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "filterLayout")) {
                WEBNN_NODE_ASSERT(GetConvTranspose2dFilterOperandLayout(
                                      jsOptions.Get("filterLayout"), options.filterLayout),
                                  "The filterLayout parameter is invalid.");
            }
        }
        Napi::Object object = Operand::constructor.New(args);
        Operand* operand = Napi::ObjectWrap<Operand>::Unwrap(object);
        operand->SetImpl(builder.ConvTranspose2d(input, filter, options.AsPtr()));

        return object;
    }

}}  // namespace node::op
