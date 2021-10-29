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

#ifndef WEBNN_NATIVE_OPS_GRU_H_
#define WEBNN_NATIVE_OPS_GRU_H_

#include "webnn_native/Graph.h"
#include "webnn_native/GraphBuilder.h"
#include "webnn_native/Operand.h"
#include "webnn_native/OperatorArray.h"

namespace webnn_native { namespace op {

    class Gru final : public OperatorBase {
      public:
        Gru(GraphBuilderBase* builder,
            OperandBase* input,
            OperandBase* weight,
            OperandBase* recurrentWeight,
            int32_t steps,
            int32_t hiddenSize,
            GruOptions const* options);
        ~Gru() override = default;

        MaybeError AddToGraph(GraphBase* graph) const override {
            return graph->AddGru(this);
        }

        MaybeError ValidateAndInferOutputInfo() override;

        GruOptions const* GetOptions() const {
            return &mOptions;
        }

        size_t GetSteps() const {
            return mSteps;
        }

        size_t GetHiddenSize() const {
            return mHiddenSize;
        }

        Ref<OperatorArrayBase> GetActivations() const {
            return mActivations;
        }

      private:
        MaybeError CalculateShape();
        GruOptions mOptions;
        size_t mSteps;
        size_t mHiddenSize;
        Ref<OperatorArrayBase> mActivations;
    };

}}  // namespace webnn_native::op

#endif  // WEBNN_NATIVE_OPS_GRU_H_
