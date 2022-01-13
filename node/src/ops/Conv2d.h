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

#ifndef NODE_OPS_CONV2D_H_
#define NODE_OPS_CONV2D_H_

#include <napi.h>
#include <webnn/webnn_cpp.h>

#include "Operand.h"
#include "Utils.h"

namespace node { namespace op {

    template <typename T>
    struct Conv2dBaseOptions {
      public:
        std::vector<int32_t> padding;
        std::vector<int32_t> strides;
        std::vector<int32_t> dilations;
        int32_t groups = 1;
        ml::AutoPad autoPad = ml::AutoPad::Explicit;
        ml::InputOperandLayout inputLayout = ml::InputOperandLayout::Nchw;
        ml::Operand bias;
        ml::FusionOperator activation;

        T& GetBaseOptions() {
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
            mOptions.groups = groups;
            mOptions.autoPad = autoPad;
            mOptions.inputLayout = inputLayout;
            mOptions.bias = bias;
            mOptions.activation = activation;

            return mOptions;
        }

      protected:
        T mOptions;
    };

    struct Conv2dOptions final : public Conv2dBaseOptions<ml::Conv2dOptions> {
      public:
        ml::Conv2dFilterOperandLayout filterLayout = ml::Conv2dFilterOperandLayout::Oihw;

        const ml::Conv2dOptions* AsPtr() {
            mOptions = GetBaseOptions();
            mOptions.filterLayout = filterLayout;
            return &mOptions;
        }
    };

    struct ConvTranspose2dOptions final : public Conv2dBaseOptions<ml::ConvTranspose2dOptions> {
      public:
        std::vector<int32_t> outputPadding;
        std::vector<int32_t> outputSizes;
        ml::ConvTranspose2dFilterOperandLayout filterLayout =
            ml::ConvTranspose2dFilterOperandLayout::Iohw;

        const ml::ConvTranspose2dOptions* AsPtr() {
            mOptions = GetBaseOptions();
            if (!outputPadding.empty()) {
                mOptions.outputPaddingCount = outputPadding.size();
                mOptions.outputPadding = outputPadding.data();
            }
            if (!outputSizes.empty()) {
                mOptions.outputSizesCount = outputSizes.size();
                mOptions.outputSizes = outputSizes.data();
            }
            mOptions.filterLayout = filterLayout;
            return &mOptions;
        }
    };

    struct Conv2d {
        static Napi::Value Build(const Napi::CallbackInfo& info, ml::GraphBuilder builder);
    };

    struct ConvTranspose2d {
        static Napi::Value Build(const Napi::CallbackInfo& info, ml::GraphBuilder builder);
    };

}}  // namespace node::op

#endif  // NODE_OPS_CONV2D_H_