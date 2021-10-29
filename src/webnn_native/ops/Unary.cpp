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

#include "webnn_native/ops/Unary.h"

#include "webnn_native/Error.h"

namespace webnn_native { namespace op {

    MaybeError Unary::ValidateAndInferOutputInfo() {
        MaybeError maybeError = OperatorBase::ValidateAndInferOutputInfo();
        if (maybeError.IsError()) {
            return maybeError;
        }
        if (mOpType == UnaryOpType::kSoftmax) {
            auto input = mInputs[0];
            if (input->Shape().size() != 2) {
                return DAWN_VALIDATION_ERROR("Input dimensions is incorrect.");
            }
        }

        if (!mInputs.empty()) {
            mOutputs[0]->SetShape(mInputs[0]->Shape());
        }

        return {};
    }

}}  // namespace webnn_native::op
