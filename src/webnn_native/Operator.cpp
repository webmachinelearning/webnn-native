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
#include "common/Log.h"
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

    OperatorBase::OperatorBase(GraphBuilderBase* graphBuilder, FusedOperator fusedOperator)
        : ObjectBase(graphBuilder->GetContext()), mFusedOperator(fusedOperator) {
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

    MaybeError OperatorBase::Validate() {
        for (auto& input : mInputs) {
            if (input->IsError()) {
                return DAWN_VALIDATION_ERROR("Argument inputs are invalid.");
            }
        }
        return {};
    }

    MaybeError OperatorBase::CalculateShape() {
        if (mInputs.empty()) {
            return {};
        }
        for (auto& output : mOutputs) {
            output->SetShape(mInputs[0]->Shape());
        }
        return {};
    }

    FusedOperator OperatorBase::GetFusedOperator() const {
        return mFusedOperator;
    }

    // static
    OperatorBase* OperatorBase::MakeError(GraphBuilderBase* graphBuilder) {
        return new OperatorBase(graphBuilder, ObjectBase::kError);
    }

    void ComputeImplicitPaddingForAutoPad(ml::AutoPad autoPad,
                                          int32_t dilation,
                                          int32_t inputSize,
                                          int32_t filterSize,
                                          int32_t stride,
                                          std::vector<int32_t>& padding) {
        int32_t outSize = (inputSize + stride - 1) / stride;
        int32_t dilatedFilter = (filterSize - 1) * dilation + 1;
        int32_t neededInput = (outSize - 1) * stride + dilatedFilter;
        int32_t totalPadding = neededInput > inputSize ? neededInput - inputSize : 0;
        int32_t paddingBegin = 0;
        int32_t paddingEnd = 0;
        switch (autoPad) {
            case ml::AutoPad::SameUpper:
                paddingBegin = totalPadding / 2;
                paddingEnd = (totalPadding + 1) / 2;
                break;
            case ml::AutoPad::SameLower:
                paddingBegin = (totalPadding + 1) / 2;
                paddingEnd = totalPadding / 2;
                break;
            default:
                DAWN_ASSERT(0);
                break;
        }
        padding.push_back(paddingBegin);
        padding.push_back(paddingEnd);
    }
}  // namespace webnn_native
