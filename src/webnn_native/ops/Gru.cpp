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

#include "webnn_native/ops/Gru.h"

#include <algorithm>

#include "common/Log.h"
#include "webnn_native/Error.h"

namespace webnn_native { namespace op {
    Gru::Gru(GraphBuilderBase* builder,
             OperandBase* input,
             OperandBase* weight,
             OperandBase* recurrentWeight,
             int32_t steps,
             int32_t hiddenSize,
             GruOptions const* options)
        : OperatorBase(builder, {input, weight, recurrentWeight}, options->returnSequence ? 2 : 1),
          mSteps(steps),
          mHiddenSize(hiddenSize) {
        if (options != nullptr) {
            mOptions = *options;
            if (options->bias != nullptr) {
                mInputs.push_back(options->bias);
            }
            if (options->recurrentBias != nullptr) {
                mInputs.push_back(options->recurrentBias);
            }
            if (options->initialHiddenState != nullptr) {
                mInputs.push_back(options->initialHiddenState);
            }
            // The first element of the output sequence is a 3-D tensor. If returnSequence is set to
            // true, the second element is the 4-D output tensor.
            if (options->returnSequence) {
                // The rank of the mOutputs[0] is set to 3 by default.
                mOutputs[1]->SetRank(4);
            }
        }
        if (options == nullptr || options->activations == nullptr) {
            mActivations = AcquireRef(new OperatorArrayBase());
            mActivations->Set(AcquireRef(new OperatorBase(builder, FusedOperator::Sigmoid)).Get());
            mActivations->Set(AcquireRef(new OperatorBase(builder, FusedOperator::Tanh)).Get());
        } else {
            mActivations = Ref<OperatorArrayBase>(mOptions.activations);
        }
    }

    MaybeError Gru::Validate() {
        MaybeError maybeError = OperatorBase::Validate();
        if (maybeError.IsError()) {
            return maybeError;
        }
        // The input 3-D tensor
        if (mInputs[0]->Rank() != 3) {
            return DAWN_VALIDATION_ERROR("Argument input is not a 3D tensor.");
        }
        // The weight 3-D tensor
        if (mInputs[1]->Rank() != 3) {
            return DAWN_VALIDATION_ERROR("Argument weight is not a 3D tensor.");
        }
        // The recurrentWeight 3-D tensor
        if (mInputs[2]->Rank() != 3) {
            return DAWN_VALIDATION_ERROR("Argument recurrentWeight is not a 3D tensor.");
        }
        // The steps parameter
        if (GetSteps() <= 0) {
            return DAWN_VALIDATION_ERROR("Argument steps value must be greater than 0.");
        }
        // The hiddenSize parameter
        if (GetHiddenSize() <= 0) {
            return DAWN_VALIDATION_ERROR("Argument hiddenSize value must be a positive integer.");
        }
        int n = 3;
        // The bias 2-D tensor
        if (mOptions.bias != nullptr) {
            if (mInputs[n++]->Rank() != 2) {
                return DAWN_VALIDATION_ERROR("Argument bias is not a 2D tensor.");
            }
        }
        // The recurrentBias 2-D tensor
        if (mOptions.recurrentBias != nullptr) {
            if (mInputs[n++]->Rank() != 2) {
                return DAWN_VALIDATION_ERROR("Argument recurrentBias is not a 2D tensor.");
            }
        }
        // The initialHiddenState 3-D tensor
        if (mOptions.initialHiddenState != nullptr) {
            if (mInputs[n++]->Rank() != 3) {
                return DAWN_VALIDATION_ERROR("Argument initialHiddenState is not a 3D tensor.");
            }
        }
        // The activations parameter
        if (GetActivations().Get()->Size() != 2) {
            return DAWN_VALIDATION_ERROR("Argument activations is not a sequence of length 2.");
        }

        return {};
    }

}}  // namespace webnn_native::op
