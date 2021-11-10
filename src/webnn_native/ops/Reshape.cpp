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

#include "webnn_native/Error.h"

namespace webnn_native { namespace op {

    MaybeError Reshape::CalculateShape() {
        auto inputShape = mInputs[0]->Shape();
        uint32_t inputSize = 1, capacity = 1;
        for (auto dim : inputShape) {
            inputSize *= dim;
        }
        int minus1DimIdx = -1;
        bool hasMinus1 = false;
        std::vector<int32_t> outputShape(mNewShape.size());
        for (size_t i = 0; i < mNewShape.size(); ++i) {
            int32_t dim = mNewShape[i];
            if (dim == -1) {
                minus1DimIdx = i;
                hasMinus1 = true;
            } else {
                capacity *= dim;
                outputShape[i] = dim;
            }
        }

        // The size of the dimension with the value -1 is computed so that the total size remains
        // constant.
        if (hasMinus1) {
            outputShape[minus1DimIdx] = inputSize / capacity;
        } else {
            // The number of elements implied by newShape must be the same as the number of elements
            // in the input tensor.
            if (inputSize != capacity) {
                return DAWN_VALIDATION_ERROR("Total size should keep consistent.");
            }
        }
        mOutputs[0]->SetShape(std::move(outputShape));
        return {};
    }

    MaybeError Reshape::ValidateAndInferOutputInfo() {
        MaybeError maybeError = OperatorBase::ValidateAndInferOutputInfo();
        if (maybeError.IsError()) {
            return maybeError;
        }

        bool hasMinus1 = false;
        // Only one component of newShape can be the special value of -1
        for (auto i : mNewShape) {
            if (i < -1 || i == 0) {
                return DAWN_VALIDATION_ERROR("Argument newShape is invalid.");
            } else if (i == -1) {
                if (hasMinus1) {
                    return DAWN_VALIDATION_ERROR("Argument newShape is invalid.");
                }
                hasMinus1 = true;
            }
        }

        return CalculateShape();
    }

}}  // namespace webnn_native::op
