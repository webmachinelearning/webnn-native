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

#include "webnn_native/ops/Gemm.h"

#include <algorithm>

#include "webnn_native/Error.h"

namespace webnn_native { namespace op {
    Gemm::Gemm(GraphBuilderBase* builder,
               OperandBase* a,
               OperandBase* b,
               GemmOptions const* options)
        : OperatorBase(builder, {a, b}) {
        mOptions.alpha = options == nullptr ? 1.0 : options->alpha;
        mOptions.beta = options == nullptr ? 1.0 : options->beta;
        mOptions.aTranspose = options == nullptr ? false : options->aTranspose;
        mOptions.bTranspose = options == nullptr ? false : options->bTranspose;
        if (options != nullptr && options->c) {
            mInputs.push_back(options->c);
        }
    }

    MaybeError Gemm::CalculateShape() {
        // The first input 2-D tensor with shape [M, K] if aTranspose is false, or [K, M] if
        // aTranspose is true. The second input 2-D tensor with shape [K, N] if bTranspose is false,
        // or [N, K] if bTranspose is true.
        auto inputAShape = mInputs[0]->Shape();
        auto inputBShape = mInputs[1]->Shape();
        bool matMulSupported = (mOptions.aTranspose ? inputAShape[0] : inputAShape[1]) ==
                               (mOptions.bTranspose ? inputBShape[1] : inputBShape[0]);
        if (!matMulSupported) {
            return DAWN_VALIDATION_ERROR(
                "Matrix multiplication failed, K should be same in the two input tensors.");
        }
        std::vector<int32_t> outputShape = {mOptions.aTranspose ? inputAShape[1] : inputAShape[0],
                                            mOptions.bTranspose ? inputBShape[0] : inputBShape[1]};
        // The third input tensor c is either a scalar, or of the shape that is unidirectionally
        // broadcastable to the shape [M, N].
        if (mInputs.size() == 3) {
            auto cShape = mInputs[2]->Shape();
            if (cShape.size() > 2) {
                return DAWN_VALIDATION_ERROR(
                    "The specified third input is either a scalar, or of the shape that is "
                    "unidirectionally broadcastable.");
            }

            for (int32_t i = cShape.size() - 1, j = outputShape.size() - 1; i >= 0 && j >= 0;
                 --i, --j) {
                if (cShape[i] != outputShape[j] && cShape[i] != 1) {
                    return DAWN_VALIDATION_ERROR(
                        "The specified third input is either a scalar, or of the shape that is "
                        "unidirectionally broadcastable.");
                }
            }
        }
        mOutputs[0]->SetShape(std::move(outputShape));
        return {};
    }

    MaybeError Gemm::ValidateAndInferOutputInfo() {
        MaybeError maybeError = OperatorBase::ValidateAndInferOutputInfo();
        if (maybeError.IsError()) {
            return maybeError;
        }
        if (mInputs[0]->Shape().size() != 2) {
            return DAWN_VALIDATION_ERROR("The first input is not 2D.");
        }
        if (mInputs[1]->Shape().size() != 2) {
            return DAWN_VALIDATION_ERROR("The second input is not 2D.");
        }

        return CalculateShape();
    }

}}  // namespace webnn_native::op
