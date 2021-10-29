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

#ifndef WEBNN_NATIVE_OPS_SPLIT_H_
#define WEBNN_NATIVE_OPS_SPLIT_H_

#include <vector>

#include "webnn_native/GraphBuilder.h"
#include "webnn_native/Operand.h"

namespace webnn_native { namespace op {

    class Split final : public OperatorBase {
      public:
        Split(GraphBuilderBase* builder,
              OperandBase* input,
              uint32_t const* splits,
              uint32_t splitsCount,
              SplitOptions const* options)
            : OperatorBase(builder, {input}, splitsCount == 1 ? splits[0] : splitsCount) {
            mAxis = options ? options->axis : 0;
            mSplits.assign(splits, splits + splitsCount);
        }
        ~Split() override = default;

        MaybeError AddToGraph(GraphBase* graph) const override {
            return graph->AddSplit(this);
        }

        MaybeError CalculateShape() {
            auto inputShape = mInputs[0]->Shape();
            auto outputShape = inputShape;
            size_t outputSize;
            auto axis = mAxis;
            if (axis < 0) {
                axis += inputShape.size();
            }

            if (mSplits.size() == 1) {
                outputSize = mSplits[0];
                outputShape[axis] /= outputSize;
            } else {
                outputSize = mSplits.size();
            }

            int32_t dimSumAlongAxis = 0;
            for (size_t i = 0; i < outputSize; ++i) {
                if (mSplits.size() != 1) {
                    outputShape[axis] = mSplits[i];
                }
                dimSumAlongAxis += outputShape[axis];
                mOutputs[i]->SetShape(outputShape);
            }

            // The number of output must evenly divide the dimension size of input along
            // options.axis.
            if (dimSumAlongAxis != inputShape[axis]) {
                return DAWN_VALIDATION_ERROR(
                    "The sum of sizes must equal to the dimension size of input along "
                    "options.axis.");
            }
            return {};
        }

        MaybeError ValidateAndInferOutputInfo() override {
            MaybeError maybeError = OperatorBase::ValidateAndInferOutputInfo();
            if (maybeError.IsError()) {
                return maybeError;
            }

            int32_t inputRank = mInputs[0]->Shape().size();
            if (mAxis >= inputRank || mAxis < (-inputRank)) {
                return DAWN_VALIDATION_ERROR("Argument axis value is invalid.");
            }

            if (mSplits.empty()) {
                return DAWN_VALIDATION_ERROR("Argument splits is invalid.");
            }

            return CalculateShape();
        }

        std::vector<uint32_t> GetSplits() const {
            return mSplits;
        }

        int32_t GetAxis() const {
            return mAxis;
        }

      private:
        std::vector<uint32_t> mSplits;
        int32_t mAxis;
    };

}}  // namespace webnn_native::op

#endif  // WEBNN_NATIVE_OPS_SPLIT_H_
