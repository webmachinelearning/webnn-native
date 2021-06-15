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

    MaybeError Reshape::ValidateAndInferTypes() {
        MaybeError maybeError = OperandBase::ValidateAndInferTypes();
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
        mRank = mNewShape.size();

        return {};
    }

}}  // namespace webnn_native::op
