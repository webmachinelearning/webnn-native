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

#ifndef WEBNN_NATIVE_OPERAND_H_
#define WEBNN_NATIVE_OPERAND_H_

#include <string>
#include <vector>

#include "webnn_native/Forward.h"
#include "webnn_native/Graph.h"
#include "webnn_native/ObjectBase.h"
#include "webnn_native/Operator.h"
#include "webnn_native/webnn_platform.h"

namespace webnn_native {

    class OperandBase : public ObjectBase {
      public:
        OperandBase(GraphBuilderBase*, OperatorBase*);
        virtual ~OperandBase() = default;

        const OperatorBase* Operator() const {
            return mOperator.Get();
        }

        ml::OperandType Type() const {
            return mType;
        }
        void SetType(ml::OperandType type) {
            mType = type;
        }

        std::vector<int32_t> Shape() const {
            return mShape;
        }

        void SetShape(std::vector<int32_t> shape) {
            mShape = std::move(shape);
        }

        static OperandBase* MakeError(GraphBuilderBase* modelBuilder);

      private:
        OperandBase(GraphBuilderBase* GraphBuilder, ObjectBase::ErrorTag tag);

      protected:
        // The operator of generating the operand.
        Ref<OperatorBase> mOperator;
        // The operand type.
        ml::OperandType mType;
        // The operand dimensions
        std::vector<int32_t> mShape;
    };
}  // namespace webnn_native

#endif  // WEBNN_NATIVE_OPERAND_H_
