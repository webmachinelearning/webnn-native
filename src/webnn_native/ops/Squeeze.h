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

namespace webnn_native { namespace op {

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

        MaybeError CalculateShape() override {
            auto inputShape =
                mInputs[0]->Shape().empty() ? std::vector<int32_t>{1} : mInputs[0]->Shape();
            auto inputRank = inputShape.size();

            std::unordered_set<int32_t> axesSqueezed;
            if (mAxes.empty()) {
                for (size_t i = 0; i < inputRank; ++i) {
                    if (inputShape[i] == 1) {
                        axesSqueezed.insert(i);
                    }
                }
            } else {
                for (const auto& axis : mAxes) {
                    axesSqueezed.insert(axis);
                }
            }

            std::vector<int32_t> outputShape;
            for (size_t i = 0; i < inputRank; ++i) {
                if (axesSqueezed.find(i) == axesSqueezed.end()) {
                    outputShape.push_back(inputShape[i]);
                }
            }
            if (outputShape.empty()) {
                outputShape = {1};
            }
            mOutputs[0]->SetShape(outputShape);
            return {};
        }

        MaybeError Validate() override {
            MaybeError maybeError = OperatorBase::Validate();
            if (maybeError.IsError()) {
                return maybeError;
            }

            auto inputRank = mInputs[0]->Shape().empty() ? 1 : mInputs[0]->Shape().size();
            for (const auto& axis : mAxes) {
                if (axis > int32_t(inputRank - 1) || axis < 0) {
                    return DAWN_VALIDATION_ERROR("Axes value is invalid.");
                }
            }

            maybeError = CalculateShape();
            if (maybeError.IsError()) {
                return maybeError;
            }
            return {};
        }

        std::vector<int32_t> GetAxes() const {
            return mAxes;
        }

      private:
        std::vector<int32_t> mAxes;
    };

}}  // namespace webnn_native::op

#endif  // WEBNN_NATIVE_OPS_SQUEEZE_H_
