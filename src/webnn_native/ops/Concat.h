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

#ifndef WEBNN_NATIVE_OPS_CONCAT_H_
#define WEBNN_NATIVE_OPS_CONCAT_H_

#include <vector>

#include "webnn_native/Graph.h"
#include "webnn_native/Operand.h"

namespace webnn_native { namespace op {

    class Concat final : public OperatorBase {
      public:
        Concat(GraphBuilderBase* builder, std::vector<Ref<OperandBase>> inputs, uint32_t axis)
            : OperatorBase(builder, std::move(inputs)), mAxis(axis) {
        }
        ~Concat() override = default;

        MaybeError AddToGraph(GraphBase* graph) const override {
            return graph->AddConcat(this);
        }
        uint32_t GetAxis() const {
            return mAxis;
        }
        MaybeError ValidateAndInferOutputInfo() override;

      private:
        MaybeError CalculateShape();
        uint32_t mAxis;
    };

}}  // namespace webnn_native::op

#endif  // WEBNN_NATIVE_OPS_CONCAT_H_
