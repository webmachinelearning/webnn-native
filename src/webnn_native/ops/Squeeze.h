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

#ifndef WEBNN_NATIVE_OPS_SQUEEZE_H_
#define WEBNN_NATIVE_OPS_SQUEEZE_H_

#include <unordered_set>

#include "webnn_native/Graph.h"
#include "webnn_native/Operand.h"
#include "webnn_native/Operator.h"

namespace webnn::native::op {

    class Squeeze final : public OperatorBase {
      public:
        Squeeze(GraphBuilderBase* builder, OperandBase* input, SqueezeOptions const* options)
            : OperatorBase(builder, {input}) {
            if (options && options->axes) {
                mAxes.assign(options->axes, options->axes + options->axesCount);
            }
        }
        ~Squeeze() override = default;

        MaybeError AddToGraph(GraphBase* graph) const override {
            return graph->AddSqueeze(this);
        }

        MaybeError CalculateShape() {
            auto inputShape = mInputs[0]->Shape();
            auto inputRank = inputShape.size();
            std::vector<int32_t> outputShape;

            //  Axes are indices to the shape dimensions of size 1 to eliminate. When not
            //  specified, every shape dimensions of size 1 in the tensor are eliminated.
            if (mAxes.empty()) {
                for (size_t i = 0; i < inputRank; ++i) {
                    if (inputShape[i] != 1) {
                        outputShape.push_back(inputShape[i]);
                    }
                }
            } else {
                std::unordered_set<int32_t> axesToSqueeze;
                for (const auto& axis : mAxes) {
                    axesToSqueeze.insert(axis);
                }

                for (size_t i = 0; i < inputRank; ++i) {
                    if (axesToSqueeze.find(i) == axesToSqueeze.end()) {
                        outputShape.push_back(inputShape[i]);
                    } else if (inputShape[i] != 1) {
                        return DAWN_VALIDATION_ERROR(
                            "Only shape dimensions of size 1 in the tensor can be eliminated.");
                    }
                }
            }

            if (outputShape.empty()) {
                outputShape = {1};
            }
            mOutputs[0]->SetShape(std::move(outputShape));
            return {};
        }

        MaybeError ValidateAndInferOutputInfo() override {
            MaybeError maybeError = OperatorBase::ValidateAndInferOutputInfo();
            if (maybeError.IsError()) {
                return maybeError;
            }

            auto inputShape = mInputs[0]->Shape();
            for (size_t i = 0; i < mAxes.size(); ++i) {
                if (mAxes[i] >= int32_t(inputShape.size()) || mAxes[i] < 0) {
                    return DAWN_VALIDATION_ERROR("Axes value is invalid.");
                }
            }

            return CalculateShape();
        }

        std::vector<int32_t> GetAxes() const {
            return mAxes;
        }

      private:
        std::vector<int32_t> mAxes;
    };

}  // namespace webnn::native::op

#endif  // WEBNN_NATIVE_OPS_SQUEEZE_H_
