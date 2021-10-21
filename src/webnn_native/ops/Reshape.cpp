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

#include "webnn_native/ops/Reshape.h"

#include "common/Log.h"
#include "webnn_native/Error.h"

namespace webnn_native { namespace op {

    MaybeError Reshape::CalculateShape() {
        auto inputShape =
            mInputs[0]->Shape().empty() ? std::vector<int32_t>{1} : mInputs[0]->Shape();
        uint32_t inputSize = 1, capacity = 1;
        for (auto dim : inputShape) {
            inputSize *= dim;
        }
        int unkDimIdx = -1;
        bool hasMinus1 = false;
        std::vector<int32_t> outputShape(mNewShape.size());
        for (size_t i = 0; i < mNewShape.size(); ++i) {
            int32_t dim = mNewShape[i];
            if (dim < -1) {
                return DAWN_VALIDATION_ERROR(
                    "The component of newShape should not be smaller than -1.");
            }
            if (dim == -1) {
                if (hasMinus1) {
                    return DAWN_VALIDATION_ERROR(
                        "Only one component of newShape can be the special value of -1.");
                }
                unkDimIdx = i;
                hasMinus1 = true;
            } else {
                capacity *= dim;
                outputShape[i] = dim;
            }
        }

        if (hasMinus1) {
            outputShape[unkDimIdx] = inputSize / capacity;
        } else {
            if (inputSize != capacity) {
                return DAWN_VALIDATION_ERROR("Total size should keep consistent.");
            }
        }
        mOutputs[0]->SetShape(outputShape);
        return {};
    }

    MaybeError Reshape::Validate() {
        MaybeError maybeError = OperatorBase::Validate();
        if (maybeError.IsError()) {
            return maybeError;
        }
        maybeError = CalculateShape();
        if (maybeError.IsError()) {
            return maybeError;
        }
        return {};
    }

}}  // namespace webnn_native::op
