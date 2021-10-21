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

#include "webnn_native/ops/Reduce.h"

#include "common/Log.h"
#include "webnn_native/Error.h"

namespace webnn_native { namespace op {

    Reduce::Reduce(GraphBuilderBase* builder,
                   ReduceType opType,
                   OperandBase* input,
                   ReduceOptions const* options)
        : OperatorBase(builder, {input}), mOpType(opType) {
        if (options == nullptr || options->axes == nullptr) {
            int32_t rank = input->Shape().size();
            mAxes.resize(rank);
            for (auto i = 0; i < rank; ++i) {
                mAxes[i] = i;
            }
        } else {
            mAxes.assign(options->axes, options->axes + options->axesCount);
        }
        mOptions.axes = mAxes.data();
        mOptions.axesCount = mAxes.size();
        if (options) {
            mOptions.keepDimensions = options->keepDimensions;
        }
    }

    MaybeError Reduce::CalculateShape() {
        auto inputShape =
            mInputs[0]->Shape().empty() ? std::vector<int32_t>{1} : mInputs[0]->Shape();
        std::vector<int32_t> tempShape(inputShape.size()), outputShape;
        tempShape = inputShape;
        for (size_t i = 0; i < mAxes.size(); ++i) {
            if (mAxes[i] == -1) {
                mAxes[i] = inputShape.size() - 1;
            }
            tempShape[mAxes[i]] = 1;
        }
        if (!mOptions.keepDimensions) {
            std::vector<int32_t> reducedShape(inputShape.size() - mAxes.size());
            for (size_t i = 0; i < inputShape.size(); ++i) {
                if (!(inputShape[i] != 1 && tempShape[i] == 1)) {
                    outputShape.push_back(inputShape[i]);
                }
            }
            if (outputShape.size() == 0) {
                outputShape = {1};
            }
        } else {
            outputShape = tempShape;
        }
        mOutputs[0]->SetShape(outputShape);
        return {};
    }

    MaybeError Reduce::Validate() {
        MaybeError maybeError = OperatorBase::Validate();
        if (maybeError.IsError()) {
            return maybeError;
        }

        // The number of values in the sequence must be smaller than the rank of the input tensor.
        auto inputRank = mInputs[0]->Shape().empty() ? 1 : mInputs[0]->Shape().size();
        if (mOptions.axesCount > inputRank) {
            return DAWN_VALIDATION_ERROR("Axes size is invalid.");
        }

        // The values in the sequence must be within the range from 0 to N-1,
        // with no two or more same values found in the sequence.
        // Besides, axis can also be -1 to represent the last dimension.
        std::map<int32_t, size_t> axesMap;
        for (size_t i = 0; i < mAxes.size(); ++i) {
            if (mAxes[i] > static_cast<int32_t>(inputRank - 1) || mAxes[i] < -1) {
                return DAWN_VALIDATION_ERROR("Axes value is invalid.");
            }

            if (axesMap.find(mAxes[i]) != axesMap.end()) {
                return DAWN_VALIDATION_ERROR("All axes must be unique");
            }
            axesMap[mAxes[i]] = i;
        }
        maybeError = CalculateShape();
        if (maybeError.IsError()) {
            return maybeError;
        }
        return {};
    }

}}  // namespace webnn_native::op
