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

        MaybeError Validate() override {
            MaybeError maybeError = OperatorBase::Validate();
            if (maybeError.IsError()) {
                return maybeError;
            }

            int inputRank = mInputs[0]->Rank();
            for (auto axis : mAxes) {
                if (abs(axis) > inputRank) {
                    return DAWN_VALIDATION_ERROR("The axes is invalid.");
                }
            }

            if (mStarts.size() != mSizes.size()) {
                return DAWN_VALIDATION_ERROR("The size of starts are invalid.");
            }

            return {};
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
