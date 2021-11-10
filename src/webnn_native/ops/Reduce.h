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

#ifndef WEBNN_NATIVE_OPS_REDUCE_H_
#define WEBNN_NATIVE_OPS_REDUCE_H_

#include "webnn_native/Graph.h"
#include "webnn_native/Operand.h"
#include "webnn_native/Operator.h"

namespace webnn_native { namespace op {

    enum ReduceType {
        kReduceL1 = 0,
        kReduceL2,
        kReduceMax,
        kReduceMean,
        kReduceMin,
        kReduceProduct,
        kReduceSum,
    };

    class Reduce : public OperatorBase {
      public:
        Reduce(GraphBuilderBase* builder,
               ReduceType opType,
               OperandBase* input,
               ReduceOptions const* options);

        ~Reduce() override = default;

        MaybeError AddToGraph(GraphBase* graph) const override {
            return graph->AddReduce(this);
        }
        MaybeError ValidateAndInferOutputInfo() override;

        ReduceType GetType() const {
            return mOpType;
        }

        ReduceOptions const* GetOptions() const {
            return &mOptions;
        }

      private:
        MaybeError CalculateShape();
        ReduceType mOpType;
        ReduceOptions mOptions;
        std::vector<int32_t> mAxes;
    };

}}  // namespace webnn_native::op

#endif  // WEBNN_NATIVE_OPS_REDUCE_H_
