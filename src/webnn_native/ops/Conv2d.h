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

#ifndef WEBNN_NATIVE_OPS_CONV2D_H_
#define WEBNN_NATIVE_OPS_CONV2D_H_

#include "webnn_native/FusionOperator.h"
#include "webnn_native/Graph.h"
#include "webnn_native/Operand.h"
#include "webnn_native/Utils.h"

namespace webnn_native { namespace op {
    template <typename OptionType>
    class Conv2dBase : public OperatorBase {
      public:
        Conv2dBase(GraphBuilderBase* builder,
                   OperandBase* input,
                   OperandBase* filter,
                   OptionType const* options)
            : OperatorBase(builder, {input, filter}) {
            if (options != nullptr && options->bias != nullptr) {
                mInputs.push_back(options->bias);
            }
            if (options == nullptr || options->padding == nullptr) {
                mPadding = std::vector<int32_t>(4, 0);
            } else {
                mPadding.assign(options->padding, options->padding + options->paddingCount);
            }
            mOptions.padding = mPadding.data();
            mOptions.paddingCount = mPadding.size();

            if (options == nullptr || options->strides == nullptr) {
                mStride = std::vector<int32_t>(2, 1);
            } else {
                mStride.assign(options->strides, options->strides + options->stridesCount);
            }
            mOptions.strides = mStride.data();
            mOptions.stridesCount = mStride.size();

            if (options == nullptr || options->dilations == nullptr) {
                mDilations = std::vector<int32_t>(2, 1);
            } else {
                mDilations.assign(options->dilations, options->dilations + options->dilationsCount);
            }
            mOptions.dilations = mDilations.data();
            mOptions.dilationsCount = mDilations.size();

            if (options != nullptr) {
                mOptions.groups = options->groups;
                mOptions.inputLayout = options->inputLayout;
                mOptions.filterLayout = options->filterLayout;
                mOptions.autoPad = options->autoPad;
                mOptions.bias = options->bias;
                mOptions.activation = options->activation;
            }
            mActivation = Ref<FusionOperatorBase>(mOptions.activation);
        }
        ~Conv2dBase() override = default;

      protected:
        MaybeError ValidateBase() {
            MaybeError maybeError = OperatorBase::ValidateAndInferOutputInfo();
            if (maybeError.IsError()) {
                return maybeError;
            }

            auto input = mInputs[0];
            auto filter = mInputs[1];
            if (input->Type() != filter->Type()) {
                return DAWN_VALIDATION_ERROR("Argument types are inconsistent.");
            }
            // The input 4-D tensor
            if (input->Shape().size() != 4) {
                return DAWN_VALIDATION_ERROR("Argument input is not a 4D tensor.");
            }
            // The filter 4-D tensor
            if (filter->Shape().size() != 4) {
                return DAWN_VALIDATION_ERROR("Argument filter is not a 4D tensor.");
            }
            // The bias is 1-D tensor.
            if (mOptions.bias != nullptr) {
                auto bias = mInputs[2];
                if (bias->Shape().size() != 1) {
                    return DAWN_VALIDATION_ERROR("Argument bias is not a 1D tensor.");
                }
            }
            // padding: a sequence of long of length 4
            if (mOptions.paddingCount != 4) {
                return DAWN_VALIDATION_ERROR("PaddingCount is incorrect.");
            }
            // strides: a sequence of long of length 2
            if (mOptions.stridesCount != 2) {
                return DAWN_VALIDATION_ERROR("stridesCount is incorrect.");
            }
            // dilations: a sequence of long of length 2
            if (mOptions.dilationsCount != 2) {
                return DAWN_VALIDATION_ERROR("dilationsCount is incorrect.");
            }

            return {};
        }

        MaybeError ValidateGroup(int32_t filterDepthIn, int32_t inputChannels) {
            if (filterDepthIn != inputChannels / mOptions.groups) {
                return DAWN_VALIDATION_ERROR(
                    "The groups is invalid, it must evenly divides the input channels.");
            }
            return {};
        }

        void calculateOutputSize(int32_t inputHeight,
                                 int32_t inputWidth,
                                 int32_t filterHeight,
                                 int32_t filterWidth,
                                 int32_t& outputHeight,
                                 int32_t& outputWidth) {
            int32_t paddingBeginningHeight = mPadding[0], paddingEndingHeight = mPadding[1],
                    paddingBeginningWidth = mPadding[2], paddingEndingWidth = mPadding[3];
            if (mOptions.autoPad != wnn::AutoPad::Explicit) {
                utils::ComputeImplicitPaddingForAutoPad(
                    mOptions.autoPad, mOptions.dilations[0], inputHeight, filterHeight,
                    mOptions.strides[0], paddingBeginningHeight, paddingEndingHeight);
                utils::ComputeImplicitPaddingForAutoPad(
                    mOptions.autoPad, mOptions.dilations[1], inputWidth, filterWidth,
                    mOptions.strides[1], paddingBeginningWidth, paddingEndingWidth);
            }
            auto dilatedFilterHeight = mDilations[0] * (filterHeight - 1) + 1;
            auto dilatedFilterWidth = mDilations[1] * (filterWidth - 1) + 1;
            outputHeight = 1 + (inputHeight - dilatedFilterHeight + paddingBeginningHeight +
                                paddingEndingHeight) /
                                   mStride[0];
            outputWidth =
                1 + (inputWidth - dilatedFilterWidth + paddingBeginningWidth + paddingEndingWidth) /
                        mStride[1];
        }

        OptionType mOptions;
        std::vector<int32_t> mPadding;
        std::vector<int32_t> mStride;
        std::vector<int32_t> mDilations;
        Ref<FusionOperatorBase> mActivation;
    };

    class Conv2d final : public Conv2dBase<Conv2dOptions> {
      public:
        Conv2d(GraphBuilderBase* builder,
               OperandBase* input,
               OperandBase* filter,
               Conv2dOptions const* options);
        ~Conv2d() override = default;

        MaybeError AddToGraph(GraphBase* graph) const override;
        Conv2dOptions const* GetOptions() const;
        MaybeError ValidateAndInferOutputInfo() override;

      private:
        MaybeError CalculateShape();
    };

    class ConvTranspose2d final : public Conv2dBase<ConvTranspose2dOptions> {
      public:
        ConvTranspose2d(GraphBuilderBase* builder,
                        OperandBase* input,
                        OperandBase* filter,
                        ConvTranspose2dOptions const* options);
        ~ConvTranspose2d() override = default;

        MaybeError AddToGraph(GraphBase* graph) const override;
        ConvTranspose2dOptions const* GetOptions() const;
        MaybeError ValidateAndInferOutputInfo() override;

      private:
        MaybeError CalculateShape();
        std::vector<int32_t> mOutputPadding;
        std::vector<int32_t> mOutputSizes;
    };
}}  // namespace webnn_native::op

#endif  // WEBNN_NATIVE_OPS_CONV2D_H_
