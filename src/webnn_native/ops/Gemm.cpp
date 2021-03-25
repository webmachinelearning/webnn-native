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
        : OperandBase(builder, {a, b}) {
        mOptions.alpha = options == nullptr ? 1.0 : options->alpha;
        mOptions.beta = options == nullptr ? 1.0 : options->beta;
        mOptions.aTranspose = options == nullptr ? false : options->aTranspose;
        mOptions.bTranspose = options == nullptr ? false : options->bTranspose;
        if (options->c) {
            mInputs.push_back(options->c);
        }
    }

    MaybeError Gemm::ValidateAndInferTypes() {
        for (auto input : mInputs) {
            if (input->IsError()) {
                return DAWN_VALIDATION_ERROR("Argument input is invalid.");
            }
        }
        if (mInputs[0]->Rank() != 2) {
            return DAWN_VALIDATION_ERROR("The first input is not 2D.");
        }
        if (mInputs[1]->Rank() != 2) {
            return DAWN_VALIDATION_ERROR("The second input is not 2D.");
        }
        if (mInputs.size() == 3) {
            if (mInputs[2]->Rank() > 2) {
                return DAWN_VALIDATION_ERROR(
                    "The specified third input is either a scalar, or of the shape that is "
                    "unidirectionally broadcastable.");
            }
        }
        mRank = mInputs[0]->Rank();
        mType = mInputs[0]->Type();

        return {};
    }

}}  // namespace webnn_native::op
