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

#include "webnn_native/ops/Pool2d.h"

#include "common/Log.h"
#include "webnn_native/Error.h"

namespace webnn_native { namespace op {

    Pool2d::Pool2d(GraphBuilderBase* builder,
                   Pool2dType opType,
                   OperandBase* input,
                   Pool2dOptions const* options)
        : OperandBase(builder, {input}), mOpType(opType) {
        if (options != nullptr && options->windowDimensions != nullptr) {
            mWindowDimensions.assign(options->windowDimensions,
                                     options->windowDimensions + options->windowDimensionsCount);
            mOptions.windowDimensions = mWindowDimensions.data();
            mOptions.windowDimensionsCount = mWindowDimensions.size();
        } else {
            // If options or windowDimensions is not present, the backend should assume the window
            // dimensions to be the height and width dimensions of the input shape.
            mOptions.windowDimensions = nullptr;
            mOptions.windowDimensionsCount = 0;
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

        mOptions.autoPad = options == nullptr ? ml::AutoPad::Explicit : options->autoPad;
        mOptions.layout = options == nullptr ? ml::InputOperandLayout::Nchw : options->layout;
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

    MaybeError Pool2d::ValidateAndInferTypes() {
        MaybeError maybeError = OperandBase::ValidateAndInferTypes();
        if (maybeError.IsError()) {
            return maybeError;
        }

        auto input = mInputs[0];
        // The input 4-D tensor
        if (input->Rank() != 4) {
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

        return {};
    }

}}  // namespace webnn_native::op
