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

#include "common/Log.h"
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
        if (options->c) {
            mInputs.push_back(options->c);
        }
    }

    MaybeError Gemm::CalculateShape() {
        auto inputAShape = mInputs[0]->Shape();
        auto inputBShape = mInputs[1]->Shape();
        std::vector<int32_t> outputShape(2);
        outputShape[0] = mOptions.aTranspose ? inputAShape[1] : inputAShape[0];
        outputShape[1] = mOptions.bTranspose ? inputBShape[0] : inputBShape[1];
        mOutputs[0]->SetShape(outputShape);
        return {};
    }

    MaybeError Gemm::Validate() {
        MaybeError maybeError = OperatorBase::Validate();
        if (maybeError.IsError()) {
            return maybeError;
        }
        if (mInputs[0]->Shape().size() != 2) {
            return DAWN_VALIDATION_ERROR("The first input is not 2D.");
        }
        if (mInputs[1]->Shape().size() != 2) {
            return DAWN_VALIDATION_ERROR("The second input is not 2D.");
        }
        if (mInputs.size() == 3) {
            if (mInputs[2]->Shape().size() > 2) {
                return DAWN_VALIDATION_ERROR(
                    "The specified third input is either a scalar, or of the shape that is "
                    "unidirectionally broadcastable.");
            }
        }
        maybeError = CalculateShape();
        if (maybeError.IsError()) {
            return maybeError;
        }
        return {};
    }

}}  // namespace webnn_native::op
