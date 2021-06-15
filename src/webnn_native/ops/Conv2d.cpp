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

namespace webnn_native { namespace op {

    Conv2d::Conv2d(GraphBuilderBase* builder,
                   OperandBase* input,
                   OperandBase* filter,
                   Conv2dOptions const* options)
        : OperandBase(builder, {input, filter}) {
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

        mOptions.groups = options == nullptr ? 1 : options->groups;
        mOptions.inputLayout =
            options == nullptr ? ml::InputOperandLayout::Nchw : options->inputLayout;
        mOptions.filterLayout =
            options == nullptr ? ml::FilterOperandLayout::Oihw : options->filterLayout;
        mOptions.autoPad = options == nullptr ? ml::AutoPad::Explicit : options->autoPad;
    }

    MaybeError Conv2d::AddToGraph(GraphBase* graph) const {
        return graph->AddConv2d(this);
    }

    Conv2dOptions const* Conv2d::GetOptions() const {
        return &mOptions;
    }

    MaybeError Conv2d::ValidateAndInferTypes() {
        MaybeError maybeError = OperandBase::ValidateAndInferTypes();
        if (maybeError.IsError()) {
            return maybeError;
        }

        auto input = mInputs[0];
        auto filter = mInputs[1];
        if (input->Type() != filter->Type()) {
            return DAWN_VALIDATION_ERROR("Argument types are inconsistent.");
        }
        // The input 4-D tensor
        if (input->Rank() != 4) {
            return DAWN_VALIDATION_ERROR("Argument input is not a 4D tensor.");
        }
        // The filter 4-D tensor
        if (filter->Rank() != 4) {
            return DAWN_VALIDATION_ERROR("Argument filter is not a 4D tensor.");
        }
        // padding: a sequence of long of length 4
        if (mOptions.paddingCount != 4) {
            return DAWN_VALIDATION_ERROR("PaddingCount is incorrect.");
        }
        // strides: a sequence of long of length 2
        if (mOptions.stridesCount != 2) {
            return DAWN_VALIDATION_ERROR("windowDimensionsCount is incorrect.");
        }
        // dilations: a sequence of long of length 2
        if (mOptions.dilationsCount != 2) {
            return DAWN_VALIDATION_ERROR("windowDimensionsCount is incorrect.");
        }

        return {};
    }

}}  // namespace webnn_native::op
