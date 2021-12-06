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

#include "webnn_native/ops/Resample2d.h"

#include <algorithm>

#include "webnn_native/Error.h"

namespace webnn_native { namespace op {
    Resample2d::Resample2d(GraphBuilderBase* builder,
                           OperandBase* input,
                           Resample2dOptions const* options)
        : OperatorBase(builder, {input}), mScales({1.0, 1.0}), mSizes({}), mAxes({2, 3}) {
        mOptions.mode = options == nullptr ? ml::InterpolationMode::NearestNeighbor : options->mode;
        if (options != nullptr && options->scales != nullptr) {
            mScales.assign(options->scales, options->scales + options->scalesCount);
        }
        if (options != nullptr && options->sizes != nullptr) {
            mSizes.assign(options->sizes, options->sizes + options->sizesCount);
        }
        if (options != nullptr && options->axes != nullptr) {
            mAxes.assign(options->axes, options->axes + options->axesCount);
        }
    }

    MaybeError Resample2d::CalculateShape() {
        auto inputShape = mInputs[0]->Shape();
        auto outputShape = inputShape;
        // When the target sizes are specified, the options.scales argument is ignored as the
        // scaling factor values are derived from the target sizes of each spatial dimension of
        // input.
        for (size_t i = 0; i < mAxes.size(); i++) {
            if (!mSizes.empty()) {
                outputShape[mAxes[i]] = mSizes[i];
            } else {
                outputShape[mAxes[i]] *= mScales[i];
            }
        }
        mOutputs[0]->SetShape(std::move(outputShape));
        return {};
    }

    MaybeError Resample2d::ValidateAndInferOutputInfo() {
        MaybeError maybeError = OperatorBase::ValidateAndInferOutputInfo();
        if (maybeError.IsError()) {
            return maybeError;
        }

        // The input is 4-D tensor.
        if (mInputs[0]->Shape().size() != 4) {
            return DAWN_VALIDATION_ERROR("Input is not a 4D tensor.");
        }
        // The scales is 2-D tensor.
        if (mOptions.scales != nullptr && mOptions.scalesCount != 2) {
            return DAWN_VALIDATION_ERROR("Argument scales is not a 2D tensor.");
        }
        // The sizes is 2-D tensor.
        if (mOptions.sizes != nullptr && mOptions.sizesCount != 2) {
            return DAWN_VALIDATION_ERROR("Argument scales is not a 2D tensor.");
        }
        // The axes is 2-D tensor and the valid values of axes in the sequence are [0, 1], [1, 2] or
        // [2, 3].
        if (mOptions.axes != nullptr) {
            if (mOptions.axesCount != 2) {
                return DAWN_VALIDATION_ERROR("Argument axes is not a 2D tensor.");
            }
            if (mOptions.axes[0] > 2 || mOptions.axes[1] != mOptions.axes[0] + 1) {
                return DAWN_VALIDATION_ERROR("The values of axes is invalid.");
            }
        }

        return CalculateShape();
    }

}}  // namespace webnn_native::op
