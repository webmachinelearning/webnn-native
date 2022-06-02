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

#ifndef WEBNN_NATIVE_OPERAND_ARRAY_H_
#define WEBNN_NATIVE_OPERAND_ARRAY_H_

#include "webnn/native/ObjectBase.h"
#include "webnn/native/Operand.h"

namespace webnn::native {

    class OperandArrayBase : public ObjectBase {
      public:
        OperandArrayBase(GraphBuilderBase* graphBuilder, std::vector<Ref<OperandBase>> operands)
            : ObjectBase(graphBuilder->GetContext()), mOperands(std::move(operands)) {
        }
        virtual ~OperandArrayBase() = default;

        static OperandArrayBase* MakeError(GraphBuilderBase* graphBuilder) {
            return new OperandArrayBase(graphBuilder, ObjectBase::kError);
        }
        // WebNN API
        size_t APISize() {
            return mOperands.size();
        }
        OperandBase* APIGetOperand(size_t index) {
            return mOperands[index].Get();
        }

      private:
        OperandArrayBase(GraphBuilderBase* graphBuilder, ObjectBase::ErrorTag tag)
            : ObjectBase(graphBuilder->GetContext(), tag) {
        }

        std::vector<Ref<OperandBase>> mOperands;
    };
}  // namespace webnn::native

#endif  // WEBNN_NATIVE_OPERAND_ARRAY_H_
