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

#include "webnn_native/Operator.h"

#include "common/Assert.h"
#include "webnn_native/GraphBuilder.h"

namespace webnn_native {
    OperatorBase::OperatorBase(GraphBuilderBase* graphBuilder,
                               std::vector<Ref<OperandBase>> inputs,
                               size_t outputSize)
        : ObjectBase(graphBuilder->GetContext()), mInputs(std::move(inputs)) {
        mOutputs.reserve(outputSize);
        for (size_t i = 0; i < outputSize; ++i) {
            mOutputs.push_back(new OperandBase(graphBuilder, this));
        }
    }

    OperatorBase::OperatorBase(GraphBuilderBase* graphBuilder, ObjectBase::ErrorTag tag)
        : ObjectBase(graphBuilder->GetContext(), tag) {
    }

    const std::vector<Ref<OperandBase>>& OperatorBase::Inputs() const {
        return mInputs;
    }

    const std::vector<Ref<OperandBase>>& OperatorBase::Outputs() const {
        return mOutputs;
    }

    OperandBase* OperatorBase::PrimaryOutput() const {
        return mOutputs[0].Get();
    }

    MaybeError OperatorBase::AddToGraph(GraphBase* graph) const {
        DAWN_UNREACHABLE();
    }

    MaybeError OperatorBase::ValidateAndInferOutputInfo() {
        for (auto& input : mInputs) {
            if (input->IsError()) {
                return DAWN_VALIDATION_ERROR("Argument inputs are invalid.");
            }
        }

        // The type is the same as input[0] by default.
        if (!mInputs.empty()) {
            mOutputs[0]->SetType(mInputs[0]->Type());
        }
        return {};
    }

    // static
    OperatorBase* OperatorBase::MakeError(GraphBuilderBase* graphBuilder) {
        return new OperatorBase(graphBuilder, ObjectBase::kError);
    }

}  // namespace webnn_native
