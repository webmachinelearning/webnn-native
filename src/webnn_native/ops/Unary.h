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

#ifndef WEBNN_NATIVE_OPS_UNARY_H_
#define WEBNN_NATIVE_OPS_UNARY_H_

#include "webnn_native/Graph.h"
#include "webnn_native/Operand.h"

namespace webnn_native { namespace op {

    enum UnaryOpType {
        kRelu = 0,
        kSoftmax,
    };

    class Unary final : public OperandBase {
      public:
        Unary(GraphBuilderBase* builder, UnaryOpType opType, OperandBase* input)
            : OperandBase(builder, {input}), mOpType(opType) {
        }
        ~Unary() override = default;

        MaybeError AddToGraph(GraphBase* model) const override {
            return model->AddUnary(this);
        }
        MaybeError ValidateAndInferTypes() override;
        UnaryOpType GetType() const {
            return mOpType;
        }

      private:
        UnaryOpType mOpType;
    };

}}  // namespace webnn_native::op

#endif  // WEBNN_NATIVE_OPS_UNARY_H_
