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

#include "webnn_native/ops/InstanceNorm.h"

#include <algorithm>

#include "webnn_native/Error.h"

namespace webnn_native { namespace op {
    InstanceNorm::InstanceNorm(GraphBuilderBase* builder,
                               OperandBase* input,
                               InstanceNormOptions const* options)
        : OperatorBase(builder, {input}) {
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
        }
    }

    MaybeError InstanceNorm::ValidateAndInferOutputInfo() {
        MaybeError maybeError = OperatorBase::ValidateAndInferOutputInfo();
        if (maybeError.IsError()) {
            return maybeError;
        }

        // The input is 4-D tensor.
        if (mInputs[0]->Shape().size() != 4) {
            return DAWN_VALIDATION_ERROR("Input is not a 4D tensor.");
        }

        // The scale is 1-D tensor.
        if (mOptions.scale != nullptr) {
            auto scale = mInputs[1];
            if (scale->Shape().size() != 1) {
                return DAWN_VALIDATION_ERROR("Argument scale is not a 1D tensor.");
            }
        }
        // The bias is 1-D tensor.
        if (mOptions.bias != nullptr) {
            size_t biasIndex = mOptions.scale != nullptr ? 2 : 1;
            auto bias = mInputs[biasIndex];
            if (bias->Shape().size() != 1) {
                return DAWN_VALIDATION_ERROR("Argument bias is not a 1D tensor.");
            }
        }

        mOutputs[0]->SetShape(mInputs[0]->Shape());

        return {};
    }

}}  // namespace webnn_native::op
