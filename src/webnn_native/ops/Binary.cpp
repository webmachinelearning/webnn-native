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

    MaybeError Binary::ValidateAndInferTypes() {
        MaybeError maybeError = OperandBase::ValidateAndInferTypes();
        if (maybeError.IsError()) {
            return maybeError;
        }

        Ref<OperandBase> a = mInputs[0];
        Ref<OperandBase> b = mInputs[1];
        if (a->Type() != b->Type()) {
            return DAWN_VALIDATION_ERROR("Argument types are inconsistent.");
        }

        // For element-wise binary ops, The rank of the output tensor
        // is the maximum rank of the input tensors.
        // According to
        // [numpy-broadcasting-rule](https://webmachinelearning.github.io/webnn/#biblio-numpy-broadcasting-rule)
        // For matmul
        // 1. if a->Rank() == 2 && b->Rank() == 2, rank_ = 2;
        // 2. if a->Rank() > 2 || b->Rank() > 2, rank_ = std::max(a->Rank(), b->Rank());
        // 3. if a->Rank() == 1 && b->Rank() == 1, rank_ = 0;
        // 4. if a->Rank() == 1 && b->Rank() == 2, rank_ = 2;
        // 5. if a->Rank() == 2 && b->Rank() == 1, rank_ = 2;
        if (mOpType == kMatMul && a->Rank() == 1 && b->Rank() == 1) {
            mRank = 0;
        } else {
            mRank = std::max(a->Rank(), b->Rank());
        }
        return {};
    }

}}  // namespace webnn_native::op
