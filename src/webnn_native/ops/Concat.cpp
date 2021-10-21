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
        auto inputShape =
            mInputs[0]->Shape().empty() ? std::vector<int32_t>{1} : mInputs[0]->Shape();
        int32_t axisShape = 0;
        for (auto& input : mInputs) {
            auto shape = input->Shape();
            if (shape.size() != inputShape.size()) {
                return DAWN_VALIDATION_ERROR("The input shapes are incompatible.");
            }
            axisShape += shape[mAxis];

            for (size_t i = 0; i < inputShape.size(); ++i) {
                if (uint32_t(i) != mAxis && shape[i] != inputShape[i]) {
                    return DAWN_VALIDATION_ERROR(
                        "All input tensors must have the same shape, except for the size of the "
                        "dimension to concatenate on.");
                }
            }
        }
        auto outputShape = inputShape;
        outputShape[mAxis] = axisShape;
        mOutputs[0]->SetShape(outputShape);
        return {};
    }

    MaybeError Concat::Validate() {
        MaybeError maybeError = OperatorBase::Validate();
        if (maybeError.IsError()) {
            return maybeError;
        }

        auto inputType = mInputs[0]->Type();
        for (auto& input : mInputs) {
            if (input->Type() != inputType) {
                return DAWN_VALIDATION_ERROR("Argument types are inconsistent.");
            }
        }
        auto inputRank = mInputs[0]->Shape().empty() ? 1 : mInputs[0]->Shape().size();
        if (mAxis >= inputRank) {
            return DAWN_VALIDATION_ERROR("The axis is out of rank range.");
        }
        maybeError = CalculateShape();
        if (maybeError.IsError()) {
            return maybeError;
        }
        return {};
    }

}}  // namespace webnn_native::op
