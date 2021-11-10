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

#include <algorithm>

#include "webnn_native/Error.h"

namespace webnn_native { namespace op {

    Reduce::Reduce(GraphBuilderBase* builder,
                   ReduceType opType,
                   OperandBase* input,
                   ReduceOptions const* options)
        : OperatorBase(builder, {input}), mOpType(opType) {
        // If axes are not present, all dimensions are reduced.
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
        auto inputShape = mInputs[0]->Shape();
        std::vector<int32_t> reducedShape = inputShape, outputShape;
        std::vector<int32_t> axes = mAxes;
        for (size_t i = 0; i < axes.size(); ++i) {
            // The dimensions to reduce where -1 means the last dimension.
            if (axes[i] == -1) {
                axes[i] = inputShape.size() - 1;
            }
            reducedShape[axes[i]] = 1;
        }

        if (!mOptions.keepDimensions) {
            for (size_t i = 0; i < inputShape.size(); ++i) {
                // The axes may be in unexpected order, use push_back to keep dimensions which
                // haven't been reduced.
                if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
                    outputShape.push_back(inputShape[i]);
                }
            }
            if (outputShape.size() == 0) {
                outputShape = {1};
            }
        } else {
            outputShape = std::move(reducedShape);
        }
        mOutputs[0]->SetShape(std::move(outputShape));
        return {};
    }

    MaybeError Reduce::ValidateAndInferOutputInfo() {
        MaybeError maybeError = OperatorBase::ValidateAndInferOutputInfo();
        if (maybeError.IsError()) {
            return maybeError;
        }

        auto inputShape = mInputs[0]->Shape();
        // The number of values in the sequence must be smaller than the rank of the input tensor.
        if (mAxes.size() > inputShape.size()) {
            return DAWN_VALIDATION_ERROR("Axes size is invalid.");
        }

        // The values in the sequence must be within the range from 0 to N-1,
        // with no two or more same values found in the sequence.
        // Besides, axis can also be -1 to represent the last dimension.
        std::map<int32_t, size_t> axesMap;
        for (size_t i = 0; i < mAxes.size(); ++i) {
            if (mAxes[i] > static_cast<int32_t>(inputShape.size() - 1) || mAxes[i] < -1) {
                return DAWN_VALIDATION_ERROR("axes value is invalid.");
            }

            if (axesMap.find(mAxes[i]) != axesMap.end()) {
                return DAWN_VALIDATION_ERROR("all axes must be unique");
            }
            axesMap[mAxes[i]] = i;
        }

        return CalculateShape();
    }

}}  // namespace webnn_native::op
