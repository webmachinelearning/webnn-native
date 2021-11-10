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

#include "webnn_native/ops/Concat.h"

#include "webnn_native/Error.h"

namespace webnn_native { namespace op {
    MaybeError Concat::CalculateShape() {
        auto outputShape = mInputs[0]->Shape();
        // The size of the dimension along axis is computed as the sum of all the input sizes of
        // the same dimension.
        outputShape[mAxis] = 0;
        for (auto& input : mInputs) {
            outputShape[mAxis] += input->Shape()[mAxis];
        }
        mOutputs[0]->SetShape(std::move(outputShape));
        return {};
    }

    MaybeError Concat::ValidateAndInferOutputInfo() {
        MaybeError maybeError = OperatorBase::ValidateAndInferOutputInfo();
        if (maybeError.IsError()) {
            return maybeError;
        }

        if (mInputs.empty()) {
            return DAWN_VALIDATION_ERROR("Empty inputs is not supported.");
        }

        auto inputType = mInputs[0]->Type();
        auto inputShape = mInputs[0]->Shape();
        auto inputRank = inputShape.size();
        for (auto& input : mInputs) {
            if (input->Type() != inputType) {
                return DAWN_VALIDATION_ERROR("Argument types are inconsistent.");
            }

            auto shape = input->Shape();
            if (shape.size() != inputShape.size()) {
                return DAWN_VALIDATION_ERROR("The input tensors must have the same rank.");
            }

            for (size_t i = 0; i < inputShape.size(); ++i) {
                if (uint32_t(i) != mAxis && shape[i] != inputShape[i]) {
                    return DAWN_VALIDATION_ERROR(
                        "Argument inputs must have same shape except for the size of the dimension "
                        "to "
                        "concatenate on.");
                }
            }
        }

        if (mAxis >= inputRank) {
            return DAWN_VALIDATION_ERROR("The axis is out of rank range.");
        }

        return CalculateShape();
    }

}}  // namespace webnn_native::op
