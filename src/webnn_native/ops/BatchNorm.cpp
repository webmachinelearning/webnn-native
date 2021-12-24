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

#include "webnn_native/ops/BatchNorm.h"

#include <algorithm>

#include "webnn_native/Error.h"

namespace webnn_native { namespace op {
    BatchNorm::BatchNorm(GraphBuilderBase* builder,
                         OperandBase* input,
                         OperandBase* mean,
                         OperandBase* variance,
                         BatchNormOptions const* options)
        : OperatorBase(builder, {input, mean, variance}) {
        if (options != nullptr) {
            mOptions = *options;
            if (options->scale != nullptr) {
                mInputs.push_back(options->scale);
            }
            if (options->bias != nullptr) {
                mInputs.push_back(options->bias);
            }
        } else {
            mOptions.scale = nullptr;
            mOptions.bias = nullptr;
            mOptions.activation = nullptr;
        }
        mActivation = Ref<FusionOperatorBase>(mOptions.activation);
    }

    MaybeError BatchNorm::ValidateAndInferOutputInfo() {
        MaybeError maybeError = OperatorBase::ValidateAndInferOutputInfo();
        if (maybeError.IsError()) {
            return maybeError;
        }

        // The input is 4-D tensor.
        if (mInputs[0]->Shape().size() != 4) {
            return DAWN_VALIDATION_ERROR("Input is not a 4D tensor.");
        }

        // The mean is 1-D tensor.
        auto mean = mInputs[1];
        if (mean->Shape().size() != 1) {
            return DAWN_VALIDATION_ERROR("Argument mean is not a 1D tensor.");
        }
        // The variance is 1-D tensor.
        auto variance = mInputs[2];
        if (variance->Shape().size() != 1) {
            return DAWN_VALIDATION_ERROR("Argument variance is not a 1D tensor.");
        }
        // The scale is 1-D tensor.
        if (mOptions.scale != nullptr) {
            auto scale = mInputs[3];
            if (scale->Shape().size() != 1) {
                return DAWN_VALIDATION_ERROR("Argument scale is not a 1D tensor.");
            }
        }
        // The bias is 1-D tensor.
        if (mOptions.bias != nullptr) {
            size_t biasIndex = mOptions.scale != nullptr ? 4 : 3;
            auto bias = mInputs[biasIndex];
            if (bias->Shape().size() != 1) {
                return DAWN_VALIDATION_ERROR("Argument bias is not a 1D tensor.");
            }
        }
        // When input is a 4-D tensor of the "nchw" or "nhwc" layout, options.axis should be set to
        // 1 or 3 respectively.
        if (mOptions.axis != 1 && mOptions.axis != 3) {
            return DAWN_VALIDATION_ERROR("Argument axis is not supported.");
        }

        mOutputs[0]->SetShape(mInputs[0]->Shape());

        return {};
    }

}}  // namespace webnn_native::op
