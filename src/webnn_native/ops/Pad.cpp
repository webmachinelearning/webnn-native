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

#include "webnn_native/ops/Pad.h"

#include "webnn_native/Error.h"

namespace webnn_native { namespace op {

    Pad::Pad(GraphBuilderBase* builder,
             OperandBase* input,
             OperandBase* padding,
             PadOptions const* options)
        : OperatorBase(builder, {input, padding}) {
        mOptions.mode = options == nullptr ? ml::PaddingMode::Constant : options->mode;
        mOptions.value = options == nullptr ? 0 : options->value;
    }

    MaybeError Pad::CalculateShape() {
        auto inputShape = mInputs[0]->Shape();
        auto paddingShape = mInputs[1]->Shape();
        std::vector<int32_t> outputShape(inputShape.size());
        // TODO(mingming): Need to check whether padding is a constant.
        const op::Constant* padding = reinterpret_cast<const op::Constant*>(mInputs[1]->Operator());
        const uint32_t* padBuffer = static_cast<const uint32_t*>(padding->GetBuffer());
        // For each dimension D of input, padding[D, 0] indicates how many values to add before the
        // content in that dimension, and padding[D, 1] indicates how many values to add after the
        // content in that dimension.
        for (size_t i = 0; i < inputShape.size(); ++i) {
            outputShape[i] = inputShape[i] + padBuffer[2 * i] + padBuffer[2 * i + 1];
        }
        mOutputs[0]->SetShape(std::move(outputShape));
        return {};
    }

    MaybeError Pad::ValidateAndInferOutputInfo() {
        MaybeError maybeError = OperatorBase::ValidateAndInferOutputInfo();
        if (maybeError.IsError()) {
            return maybeError;
        }

        auto inputShape = mInputs[0]->Shape();
        auto paddingShape = mInputs[1]->Shape();

        if (paddingShape.size() != 2 || inputShape.size() != size_t(paddingShape[0]) ||
            paddingShape[1] != 2) {
            return DAWN_VALIDATION_ERROR(
                "The padding tensor should has shape [n, 2] where n is the rank of the input "
                "tensor.");
        }

        return CalculateShape();
    }

}}  // namespace webnn_native::op