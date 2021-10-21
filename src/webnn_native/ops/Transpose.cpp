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

#include "webnn_native/ops/Transpose.h"

#include <algorithm>

#include "common/Log.h"
#include "webnn_native/Error.h"

namespace webnn_native { namespace op {

    MaybeError Transpose::CalculateShape() {
        auto inputShape =
            mInputs[0]->Shape().empty() ? std::vector<int32_t>{1} : mInputs[0]->Shape();
        size_t rank = inputShape.size();
        std::vector<int32_t> outputShape(rank);
        for (size_t i = 0; i < rank; i++) {
            outputShape[i] = inputShape[mPermutation[i]];
        }
        mOutputs[0]->SetShape(outputShape);
        return {};
    }

    MaybeError Transpose::Validate() {
        MaybeError maybeError = OperatorBase::Validate();
        if (maybeError.IsError()) {
            return maybeError;
        }

        // the number of values in the sequence must be the same as the rank of the input tensor
        auto inputRank = mInputs[0]->Shape().empty() ? 1 : mInputs[0]->Shape().size();
        if (mPermutation.size() != size_t(inputRank)) {
            return DAWN_VALIDATION_ERROR("permutation size is invalid.");
        }

        // the values in the sequence must be within the range from 0 to N-1
        // with no two or more same values found in the sequence.
        std::vector<uint32_t> newPermutation;
        newPermutation.assign(mPermutation.begin(), mPermutation.end());
        std::sort(newPermutation.begin(), newPermutation.end());
        for (uint32_t i = 0; i < inputRank - 1; i++) {
            if (newPermutation[i] != i) {
                return DAWN_VALIDATION_ERROR("permutation value is invalid.");
            }
        }
        maybeError = CalculateShape();
        if (maybeError.IsError()) {
            return maybeError;
        }
        return {};
    }

}}  // namespace webnn_native::op
