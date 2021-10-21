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

#include "webnn_native/ops/Binary.h"

#include "common/Log.h"
#include "webnn_native/Error.h"

namespace webnn_native { namespace op {

    MaybeError Binary::CalculateShape() {
        auto inputShape1 =
            mInputs[0]->Shape().empty() ? std::vector<int32_t>{1} : mInputs[0]->Shape();
        auto inputShape2 =
            mInputs[1]->Shape().empty() ? std::vector<int32_t>{1} : mInputs[1]->Shape();

        auto l1 = inputShape1.size(), l2 = inputShape2.size();
        bool shape1IsBigger = l1 >= l2;
        auto maxShape = shape1IsBigger ? inputShape1 : inputShape2;
        auto minShape = shape1IsBigger ? inputShape2 : inputShape1;
        std::vector<int32_t> outputShape;
        if (mOpType == kMatMul) {
            if (l1 == 1 && l2 == 1) {
                if (inputShape1 != inputShape2) {
                    return DAWN_VALIDATION_ERROR(
                        "The two 1D inputs of Matmul should have the same shape.");
                }
                outputShape = {1};
            }
            if (l1 == 2 && l2 == 1) {
                if (inputShape1[1] != inputShape2[0]) {
                    return DAWN_VALIDATION_ERROR("The input shapes are incompatible.");
                }
                outputShape = {inputShape1[0], 1};
            }
            if (l1 == 1 && l2 == 2) {
                if (inputShape1[0] != inputShape2[0]) {
                    return DAWN_VALIDATION_ERROR("The input shapes are incompatible.");
                }
                outputShape = {1, inputShape2[1]};
            }
            if (l1 >= 2 && l2 >= 2) {
                if (inputShape1[l1 - 1] != inputShape2[l2 - 2]) {
                    return DAWN_VALIDATION_ERROR("The input shapes are incompatible.");
                }
                // broadcasting support
                for (int32_t i = (int32_t)maxShape.size() - 3, j = (int32_t)minShape.size() - 3;
                     i >= 0 && j >= 0; --i, --j) {
                    auto maxDim = maxShape[i], minDim = minShape[j];
                    if (maxDim != minDim && maxDim != 1 && minDim != 1) {
                        return DAWN_VALIDATION_ERROR(
                            "Shapes are not compatible for Matmul, broadcasting failed.");
                    }
                    if (maxDim < minDim) {
                        maxShape[i] = minDim;
                    }
                }
                outputShape = maxShape;
                outputShape[outputShape.size() - 1] = inputShape2[l2 - 1];
                outputShape[outputShape.size() - 2] = inputShape1[l1 - 2];
            }
        } else {
            // broadcasting support
            for (int32_t i = (int32_t)maxShape.size() - 1, j = (int32_t)minShape.size() - 1;
                 i >= 0 && j >= 0; --i, --j) {
                auto maxDim = maxShape[i], minDim = minShape[j];
                if (maxDim != minDim && maxDim != 1 && minDim != 1) {
                    return DAWN_VALIDATION_ERROR(
                        "Shapes are incompatible for Matmul, broadcasting failed.");
                }
                if (maxDim < minDim) {
                    maxShape[i] = minDim;
                }
            }
            outputShape = maxShape;
        }
        mOutputs[0]->SetShape(outputShape);
        return {};
    }

    MaybeError Binary::Validate() {
        MaybeError maybeError = OperatorBase::Validate();
        if (maybeError.IsError()) {
            return maybeError;
        }

        Ref<OperandBase> a = mInputs[0];
        Ref<OperandBase> b = mInputs[1];
        if (a->Type() != b->Type()) {
            return DAWN_VALIDATION_ERROR("Argument types are inconsistent.");
        }
        maybeError = CalculateShape();
        if (maybeError.IsError()) {
            return maybeError;
        }
        return {};
    }

}}  // namespace webnn_native::op
