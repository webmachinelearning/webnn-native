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

#include "webnn/native/ops/Conv2d.h"

#include "webnn/native/Error.h"
#include "webnn/native/Operator.h"
#include "webnn/native/Utils.h"

namespace webnn::native::op {

    Conv2d::Conv2d(GraphBuilderBase* builder,
                   OperandBase* input,
                   OperandBase* filter,
                   Conv2dOptions const* options)
        : Conv2dBase(builder, input, filter, options) {
    }

    MaybeError Conv2d::AddToGraph(GraphBase* graph) const {
        return graph->AddConv2d(this);
    }

    Conv2dOptions const* Conv2d::GetOptions() const {
        return &mOptions;
    }

    void Conv2d::calculateOutputSize(int32_t inputHeight,
                                     int32_t inputWidth,
                                     int32_t filterHeight,
                                     int32_t filterWidth,
                                     int32_t& outputHeight,
                                     int32_t& outputWidth) {
        std::vector<int32_t> padding(mPadding);
        if (mOptions.autoPad != wnn::AutoPad::Explicit) {
            std::vector<int32_t> inputSize = {inputHeight, inputWidth};
            std::vector<int32_t> filterSize = {filterHeight, filterWidth};
            padding = utils::ComputeImplicitPaddingForAutoPad(&mOptions, inputSize, filterSize);
        }
        int32_t paddingBeginningHeight = padding[0], paddingEndingHeight = padding[1],
                paddingBeginningWidth = padding[2], paddingEndingWidth = padding[3];
        auto dilatedFilterHeight = mDilations[0] * (filterHeight - 1) + 1;
        auto dilatedFilterWidth = mDilations[1] * (filterWidth - 1) + 1;
        outputHeight =
            1 + (inputHeight - dilatedFilterHeight + paddingBeginningHeight + paddingEndingHeight) /
                    mStride[0];
        outputWidth =
            1 + (inputWidth - dilatedFilterWidth + paddingBeginningWidth + paddingEndingWidth) /
                    mStride[1];
    }

    MaybeError Conv2d::CalculateShape() {
        auto inputShape = mInputs[0]->Shape();
        auto filterShape = mInputs[1]->Shape();
        bool nchw = mOptions.inputLayout == wnn::InputOperandLayout::Nchw;
        int32_t batchSize = inputShape[0];
        int32_t inputHeight = nchw ? inputShape[2] : inputShape[1];
        int32_t inputWidth = nchw ? inputShape[3] : inputShape[2];
        int32_t inputChannels = nchw ? inputShape[1] : inputShape[3];

        int32_t filterHeight = 0, filterWidth = 0, outputChannels = 0, filterDepthIn = 0;
        switch (mOptions.filterLayout) {
            case wnn::Conv2dFilterOperandLayout::Hwio:
                filterHeight = filterShape[0];
                filterWidth = filterShape[1];
                outputChannels = filterShape[3];
                filterDepthIn = filterShape[2];
                break;
            case wnn::Conv2dFilterOperandLayout::Ohwi:
                filterHeight = filterShape[1];
                filterWidth = filterShape[2];
                outputChannels = filterShape[0];
                filterDepthIn = filterShape[3];
                break;
            case wnn::Conv2dFilterOperandLayout::Ihwo:
                filterHeight = filterShape[1];
                filterWidth = filterShape[2];
                outputChannels = filterShape[3];
                filterDepthIn = filterShape[0];
                break;
            case wnn::Conv2dFilterOperandLayout::Oihw:
                filterHeight = filterShape[2];
                filterWidth = filterShape[3];
                outputChannels = filterShape[0];
                filterDepthIn = filterShape[1];
                break;
            default:
                return DAWN_VALIDATION_ERROR("The filter layout is unsupported");
        }
        MaybeError maybeError = ValidateGroup(filterDepthIn, inputChannels);
        if (maybeError.IsError()) {
            return maybeError;
        }

        int32_t outputHeight, outputWidth;
        calculateOutputSize(inputHeight, inputWidth, filterHeight, filterWidth, outputHeight,
                            outputWidth);
        std::vector<int32_t> outputShape;
        if (nchw) {
            outputShape = {batchSize, outputChannels, outputHeight, outputWidth};
        } else {
            outputShape = {batchSize, outputHeight, outputWidth, outputChannels};
        }
        mOutputs[0]->SetShape(std::move(outputShape));

        return {};
    }

    MaybeError Conv2d::ValidateAndInferOutputInfo() {
        MaybeError maybeError = Conv2dBase::ValidateBase();
        if (maybeError.IsError()) {
            return maybeError;
        }

        return CalculateShape();
    }

}  // namespace webnn::native::op
