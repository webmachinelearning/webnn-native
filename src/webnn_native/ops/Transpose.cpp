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

    MaybeError Transpose::ValidateAndInferTypes() {
        MaybeError maybeError = OperandBase::ValidateAndInferTypes();
        if (maybeError.IsError()) {
            return maybeError;
        }

        // the number of values in the sequence must be the same as the rank of the input tensor
        if (mPermutation.size() != size_t(mRank)) {
            return DAWN_VALIDATION_ERROR("permutation size is invalid.");
        }

        // the values in the sequence must be within the range from 0 to N-1
        // with no two or more same values found in the sequence.
        std::vector<uint32_t> newPermutation;
        newPermutation.assign(mPermutation.begin(), mPermutation.end());
        std::sort(newPermutation.begin(), newPermutation.end());
        for (uint32_t i = 0; i < mRank - 1; i++) {
            if (newPermutation[i] != i) {
                return DAWN_VALIDATION_ERROR("permutation value is invalid.");
            }
        }

        return {};
    }

}}  // namespace webnn_native::op
