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

#ifndef WEBNN_NATIVE_OPS_TRANSPOSE_H_
#define WEBNN_NATIVE_OPS_TRANSPOSE_H_

#include "webnn_native/Graph.h"
#include "webnn_native/Operand.h"
#include "webnn_native/Operator.h"

namespace webnn_native { namespace op {

    class Transpose final : public OperatorBase {
      public:
        Transpose(GraphBuilderBase* builder, OperandBase* input, TransposeOptions const* options)
            : OperatorBase(builder, {input}) {
            if (options == nullptr || options->permutation == nullptr) {
                int32_t rank = input->Shape().size();
                mPermutation.resize(rank);
                for (auto i = 0; i < rank - 1; i++) {
                    mPermutation[i] = rank - 1 - i;
                }
            } else {
                mPermutation.assign(options->permutation,
                                    options->permutation + options->permutationCount);
            }
        }
        ~Transpose() override = default;

        MaybeError AddToGraph(GraphBase* graph) const override {
            return graph->AddTranspose(this);
        }
        MaybeError ValidateAndInferOutputInfo() override;

        std::vector<int32_t> GetPermutation() const {
            return mPermutation;
        }

      private:
        MaybeError CalculateShape();
        std::vector<int32_t> mPermutation;
    };

}}  // namespace webnn_native::op

#endif  // WEBNN_NATIVE_OPS_TRANSPOSE_H_
