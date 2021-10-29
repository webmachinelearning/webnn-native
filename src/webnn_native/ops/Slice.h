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

#ifndef WEBNN_NATIVE_OPS_SLICE_H_
#define WEBNN_NATIVE_OPS_SLICE_H_

#include <vector>

#include "webnn_native/Graph.h"
#include "webnn_native/Operand.h"
#include "webnn_native/Operator.h"

namespace webnn_native { namespace op {

    class Slice final : public OperatorBase {
      public:
        Slice(GraphBuilderBase* builder,
              OperandBase* input,
              int32_t const* starts,
              uint32_t startsCount,
              int32_t const* sizes,
              uint32_t sizesCount,
              SliceOptions const* options)
            : OperatorBase(builder, {input}) {
            if (options != nullptr && options->axes != nullptr) {
                mAxes.assign(options->axes, options->axes + options->axesCount);
            }

            mStarts.assign(starts, starts + startsCount);
            mSizes.assign(sizes, sizes + sizesCount);
        }

        ~Slice() override = default;

        MaybeError AddToGraph(GraphBase* graph) const override {
            return graph->AddSlice(this);
        }

        MaybeError CalculateShape() {
            auto inputShape = mInputs[0]->Shape();
            auto outputShape = inputShape;
            std::vector<int32_t> axes;
            if (mAxes.empty()) {
                for (size_t i = 0; i < mSizes.size(); ++i) {
                    axes.push_back(i);
                }
            } else {
                axes = mAxes;
                if (axes.size() != mSizes.size()) {
                    return DAWN_VALIDATION_ERROR("The size of axes is invalid.");
                }
            }

            for (size_t i = 0; i < axes.size(); ++i) {
                if (axes[i] < 0) {
                    axes[i] += inputShape.size();
                }
                if (inputShape[axes[i]] < mSizes[i]) {
                    return DAWN_VALIDATION_ERROR(
                        "The target size should be smaller than the input size.");
                }

                // The values in the starts sequence are either within the [0, r-1] range where r is
                // the dimension size of input shape along the axis, or the [-r, -1] range where
                // negative values mean counting back from the end of that dimension along the axis.
                if (mStarts[i] >= inputShape[axes[i]] || mStarts[i] < -inputShape[axes[i]]) {
                    return DAWN_VALIDATION_ERROR("The values of starts are out of range.");
                }

                // The length value of -1 from size selects all the remaining elements from the
                // starting index of the given axis.
                auto remainingSize =
                    mStarts[i] < 0 ? -mStarts[i] : inputShape[axes[i]] - mStarts[i];
                if (mSizes[i] == -1) {
                    // A negative index value from starts is interpreted as counting back from the
                    // end.
                    outputShape[axes[i]] = remainingSize;
                } else {
                    if (remainingSize < mSizes[i]) {
                        return DAWN_VALIDATION_ERROR(
                            "The target size should be smaller than the number of remaining "
                            "elements from the starting index of the given axis.");
                    }
                    outputShape[axes[i]] = mSizes[i];
                }
            }
            mOutputs[0]->SetShape(std::move(outputShape));
            return {};
        }

        MaybeError ValidateAndInferOutputInfo() override {
            MaybeError maybeError = OperatorBase::ValidateAndInferOutputInfo();
            if (maybeError.IsError()) {
                return maybeError;
            }

            int32_t inputRank = mInputs[0]->Shape().size();
            for (auto axis : mAxes) {
                if (axis >= inputRank || axis < (-inputRank)) {
                    return DAWN_VALIDATION_ERROR("The axes is invalid.");
                }
            }

            if (mStarts.size() != mSizes.size()) {
                return DAWN_VALIDATION_ERROR("The size of starts are invalid.");
            }

            return CalculateShape();
        }

        std::vector<int32_t> GetStarts() const {
            return mStarts;
        }

        std::vector<int32_t> GetSizes() const {
            return mSizes;
        }

        std::vector<int32_t> GetAxes() const {
            return mAxes;
        }

      private:
        std::vector<int32_t> mStarts;
        std::vector<int32_t> mSizes;
        std::vector<int32_t> mAxes;
    };

}}  // namespace webnn_native::op

#endif  // WEBNN_NATIVE_OPS_SLICE_H_
