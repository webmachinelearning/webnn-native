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

#include "webnn/native/ops/Pool2d.h"

#include <math.h>

#include "webnn/native/Error.h"
#include "webnn/native/Utils.h"

namespace webnn::native::op {

    Pool2d::Pool2d(GraphBuilderBase* builder,
                   Pool2dType opType,
                   OperandBase* input,
                   Pool2dOptions const* options)
        : OperatorBase(builder, {input}), mOpType(opType) {
        if (options != nullptr && options->windowDimensionsCount != 0) {
            mWindowDimensions.assign(options->windowDimensions,
                                     options->windowDimensions + options->windowDimensionsCount);
            mOptions.windowDimensions = mWindowDimensions.data();
            mOptions.windowDimensionsCount = mWindowDimensions.size();
        }

        if (options == nullptr || options->paddingCount == 0) {
            mPadding = std::vector<int32_t>(4, 0);
        } else {
            mPadding.assign(options->padding, options->padding + options->paddingCount);
        }
        mOptions.padding = mPadding.data();
        mOptions.paddingCount = mPadding.size();

        if (options == nullptr || options->stridesCount == 0) {
            mStride = std::vector<int32_t>(2, 1);
        } else {
            mStride.assign(options->strides, options->strides + options->stridesCount);
        }
        mOptions.strides = mStride.data();
        mOptions.stridesCount = mStride.size();

        if (options == nullptr || options->dilationsCount == 0) {
            mDilations = std::vector<int32_t>(2, 1);
        } else {
            mDilations.assign(options->dilations, options->dilations + options->dilationsCount);
        }
        mOptions.dilations = mDilations.data();
        mOptions.dilationsCount = mDilations.size();

        mOptions.autoPad = options == nullptr ? wnn::AutoPad::Explicit : options->autoPad;
        mOptions.layout = options == nullptr ? wnn::InputOperandLayout::Nchw : options->layout;
        mOptions.roundingType =
            options == nullptr ? wnn::RoundingType::Floor : options->roundingType;

        if (options != nullptr && options->outputSizesCount != 0) {
            mOutputSizes.assign(options->outputSizes,
                                options->outputSizes + options->outputSizesCount);
            mOptions.outputSizes = mOutputSizes.data();
            mOptions.outputSizesCount = mOutputSizes.size();
        }
    }

    MaybeError Pool2d::AddToGraph(GraphBase* graph) const {
        return graph->AddPool2d(this);
    }

    Pool2dOptions const* Pool2d::GetOptions() const {
        return &mOptions;
    }

    Pool2dType Pool2d::GetType() const {
        return mOpType;
    }

    const std::vector<int32_t>& Pool2d::GetOutputSizes() const {
        return mOutputSizes;
    }

    MaybeError Pool2d::CalculateShape() {
        auto inputShape = mInputs[0]->Shape();
        bool nchw = mOptions.layout == wnn::InputOperandLayout::Nchw;
        int32_t inputHeight = nchw ? inputShape[2] : inputShape[1];
        int32_t inputWidth = nchw ? inputShape[3] : inputShape[2];
        int32_t windowHeight =
            mOptions.windowDimensions == nullptr ? inputHeight : mWindowDimensions[0];
        int32_t windowWidth =
            mOptions.windowDimensions == nullptr ? inputWidth : mWindowDimensions[1];

        int32_t paddingBeginningHeight = mPadding[0], paddingEndingHeight = mPadding[1],
                paddingBeginningWidth = mPadding[2], paddingEndingWidth = mPadding[3];
        if (mOptions.autoPad != wnn::AutoPad::Explicit) {
            utils::ComputeImplicitPaddingForAutoPad(mOptions.autoPad, mOptions.dilations[0],
                                                    inputHeight, windowHeight, mOptions.strides[0],
                                                    paddingBeginningHeight, paddingEndingHeight);
            utils::ComputeImplicitPaddingForAutoPad(mOptions.autoPad, mOptions.dilations[1],
                                                    inputWidth, windowWidth, mOptions.strides[1],
                                                    paddingBeginningWidth, paddingEndingWidth);
        }

        // TODO(mingming): We may need to consider dilations when calculating output sizes.
        int32_t outputHeight, outputWidth;
        int32_t floorOutputHeight =
            1 + (inputHeight - windowHeight + paddingBeginningHeight + paddingEndingHeight) /
                    mStride[0];
        int32_t floorOutputWidth =
            1 +
            (inputWidth - windowWidth + paddingBeginningWidth + paddingEndingWidth) / mStride[1];
        int32_t ceilOutputHeight = ceil(
            1 + 1.0 * (inputHeight - windowHeight + paddingBeginningHeight + paddingEndingHeight) /
                    mStride[0]);
        int32_t ceilOutputWidth =
            ceil(1 + 1.0 * (inputWidth - windowWidth + paddingBeginningWidth + paddingEndingWidth) /
                         mStride[1]);
        if (mOptions.outputSizes == nullptr) {
            outputHeight = mOptions.roundingType == wnn::RoundingType::Floor ? floorOutputHeight
                                                                             : ceilOutputHeight;
            outputWidth = mOptions.roundingType == wnn::RoundingType::Floor ? floorOutputWidth
                                                                            : ceilOutputWidth;
        } else {
            outputHeight = mOptions.outputSizes[0];
            outputWidth = mOptions.outputSizes[1];
            // Predict and reset the implicit rounding type by the explicitly specified outputSizes
            // which should match either floor or ceil rounding type.
            if (outputHeight == floorOutputHeight && outputWidth == floorOutputWidth) {
                mOptions.roundingType = wnn::RoundingType::Floor;
            } else if (outputHeight == ceilOutputHeight && outputWidth == ceilOutputWidth) {
                mOptions.roundingType = wnn::RoundingType::Ceil;
            } else {
                return DAWN_VALIDATION_ERROR("Invalid output sizes.");
            }
        }

        std::vector<int32_t> outputShape;
        int32_t batches = inputShape[0];
        int32_t channels = nchw ? inputShape[1] : inputShape[3];
        if (nchw) {
            outputShape = {batches, channels, outputHeight, outputWidth};
        } else {
            outputShape = {batches, outputHeight, outputWidth, channels};
        }
        mOutputs[0]->SetShape(std::move(outputShape));
        mOutputSizes = {outputHeight, outputWidth};
        return {};
    }

    MaybeError Pool2d::ValidateAndInferOutputInfo() {
        MaybeError maybeError = OperatorBase::ValidateAndInferOutputInfo();
        if (maybeError.IsError()) {
            return maybeError;
        }

        auto input = mInputs[0];
        // The input 4-D tensor
        if (input->Shape().size() != 4) {
            return DAWN_VALIDATION_ERROR("Argument input is not a 4D tensor.");
        }
        // windowDimensions: a sequence of long of length 2
        if (mOptions.windowDimensionsCount != 2 && mOptions.windowDimensionsCount != 0) {
            return DAWN_VALIDATION_ERROR("windowDimensionsCount is incorrect.");
        }
        // padding: a sequence of long of length 4
        if (mOptions.paddingCount != 4) {
            return DAWN_VALIDATION_ERROR("paddingCount is incorrect.");
        }
        // strides: a sequence of long of length 2
        if (mOptions.stridesCount != 2) {
            return DAWN_VALIDATION_ERROR("stridesCount is incorrect.");
        }
        // dilations: a sequence of long of length 2.
        if (mOptions.dilationsCount != 2) {
            return DAWN_VALIDATION_ERROR("dilationsCount is incorrect.");
        }

        return CalculateShape();
    }

}  // namespace webnn::native::op
