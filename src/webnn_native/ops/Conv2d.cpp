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

#include "webnn_native/ops/Conv2d.h"

#include "common/Log.h"
#include "webnn_native/Error.h"
#include "webnn_native/Operator.h"

namespace webnn_native { namespace op {

    Conv2d::Conv2d(GraphBuilderBase* builder,
                   OperandBase* input,
                   OperandBase* filter,
                   Conv2dOptions const* options)
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

        if (options == nullptr || options->outputPadding == nullptr) {
            mOutputPadding = std::vector<int32_t>(2, 0);
        } else {
            mOutputPadding.assign(options->outputPadding,
                                  options->outputPadding + options->outputPaddingCount);
        }
        mOptions.outputPadding = mOutputPadding.data();
        mOptions.outputPaddingCount = mOutputPadding.size();

        if (options != nullptr && options->outputSizes != nullptr) {
            mOutputSizes.assign(options->outputSizes,
                                options->outputSizes + options->outputSizesCount);
            mOptions.outputSizes = mOutputSizes.data();
            mOptions.outputSizesCount = mOutputSizes.size();
        }

        if (options != nullptr) {
            mOptions.transpose = options->transpose;
            mOptions.groups = options->groups;
            mOptions.inputLayout = options->inputLayout;
            mOptions.filterLayout = options->filterLayout;
            mOptions.autoPad = options->autoPad;
            mOptions.bias = options->bias;
            mOptions.activation = options->activation;
        }
        mActivation = Ref<OperatorBase>(mOptions.activation);
    }

    MaybeError Conv2d::AddToGraph(GraphBase* graph) const {
        return graph->AddConv2d(this);
    }

    // Reorder filter shape to oihw.
    void Conv2d::ReorderFilterShapeToOihw(ml::FilterOperandLayout layout,
                                          std::vector<int32_t>& shape) {
        switch (layout) {
            case ml::FilterOperandLayout::Hwio:
                shape = {shape[3], shape[2], shape[0], shape[1]};
                break;
            case ml::FilterOperandLayout::Ohwi:
                shape = {shape[0], shape[3], shape[1], shape[2]};
                break;
            case ml::FilterOperandLayout::Ihwo:
                shape = {shape[3], shape[0], shape[1], shape[2]};
                break;
            case ml::FilterOperandLayout::Oihw:
                break;
            default:
                dawn::ErrorLog() << "The filter layout is unsupported";
                DAWN_ASSERT(0);
                break;
        }
    }

    Conv2dOptions const* Conv2d::GetOptions() const {
        return &mOptions;
    }

    MaybeError Conv2d::CalculateShape() {
        auto inputShape = mInputs[0]->Shape();
        auto filterShape = mInputs[1]->Shape();
        ReorderFilterShapeToOihw(mOptions.filterLayout, filterShape);
        bool nchw = mOptions.inputLayout == ml::InputOperandLayout::Nchw;
        int32_t inputH = nchw ? inputShape[2] : inputShape[1];
        int32_t filterH = filterShape[2];
        int32_t inputW = nchw ? inputShape[3] : inputShape[2];
        int32_t filterW = filterShape[3];

        std::vector<int32_t> inputPadding;
        if (mOptions.autoPad == ml::AutoPad::Explicit) {
            inputPadding = mPadding;
        } else {
            ComputeImplicitPaddingForAutoPad(mOptions.autoPad, mOptions.dilations[0], inputH,
                                             filterH, mOptions.strides[0], inputPadding);
            ComputeImplicitPaddingForAutoPad(mOptions.autoPad, mOptions.dilations[1], inputW,
                                             filterW, mOptions.strides[1], inputPadding);
        }

        int32_t outputH, outputW;
        auto dilatedFilterH = mDilations[0] * (filterH - 1) + 1;
        auto dilatedFilterW = mDilations[1] * (filterW - 1) + 1;
        if (mOptions.transpose) {
            if (mOptions.outputSizes == nullptr) {
                outputH = (inputH - 1) * mStride[0] + dilatedFilterH - inputPadding[0] -
                          inputPadding[1] + mOutputPadding[0];
                outputW = (inputW - 1) * mStride[1] + dilatedFilterW - inputPadding[2] -
                          inputPadding[3] + mOutputPadding[1];
            } else {
                outputH = mOptions.outputSizes[0];
                outputW = mOptions.outputSizes[1];
            }
        } else {
            outputH =
                1 + (inputH - dilatedFilterH + inputPadding[0] + inputPadding[1]) / mStride[0];
            outputW =
                1 + (inputW - dilatedFilterW + inputPadding[2] + inputPadding[3]) / mStride[1];
        }

        std::vector<int32_t> outputShape;
        if (nchw) {
            outputShape = {inputShape[0], filterShape[0], outputH, outputW};
        } else {
            outputShape = {inputShape[0], outputH, outputW, filterShape[0]};
        }
        mOutputs[0]->SetShape(outputShape);
        return {};
    }

    MaybeError Conv2d::Validate() {
        MaybeError maybeError = OperatorBase::Validate();
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

        maybeError = CalculateShape();
        if (maybeError.IsError()) {
            return maybeError;
        }
        return {};
    }

}}  // namespace webnn_native::op
