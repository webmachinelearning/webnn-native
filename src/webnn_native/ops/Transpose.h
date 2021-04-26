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

namespace webnn_native { namespace op {

    class Transpose final : public OperandBase {
      public:
        Transpose(GraphBuilderBase* builder, OperandBase* input, TransposeOptions const* options)
            : OperandBase(builder, {input}) {
            if (options == nullptr || options->permutation == nullptr) {
                int32_t rank = input->Rank();
                mPermutation.resize(rank);
                for (auto i = 0; i < rank - 1; i++) {
                    mPermutation[i] = rank - 1 - i;
                }
            } else {
                mPermutation.assign(options->permutation,
                                    options->permutation + options->permutationCount);
            }
            mOptions.permutation = mPermutation.data();
            mOptions.permutationCount = mPermutation.size();
        }
        ~Transpose() override = default;

        MaybeError AddToGraph(GraphBase* model) const override {
            return model->AddTranspose(this);
        }
        MaybeError ValidateAndInferTypes() override;

        TransposeOptions const* GetOptions() const {
            return &mOptions;
        }

      private:
        TransposeOptions mOptions;
        std::vector<int32_t> mPermutation;
    };

}}  // namespace webnn_native::op

#endif  // WEBNN_NATIVE_OPS_TRANSPOSE_H_
