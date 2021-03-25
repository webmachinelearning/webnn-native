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
#include "webnn_native/Model.h"
#include "webnn_native/ObjectBase.h"
#include "webnn_native/webnn_platform.h"

namespace webnn_native {

    class OperandBase : public ObjectBase {
      public:
        explicit OperandBase(ModelBuilderBase* modelBuilder, std::vector<Ref<OperandBase>> = {});
        virtual ~OperandBase() = default;

        // It's used for getting inputs when traversaling model tree.
        const std::vector<Ref<OperandBase>>& Inputs() const;
        // Add the operand to model for specific backend.
        virtual MaybeError AddToModel(ModelBase* model) const;

        webnn::OperandType Type() const {
            return mType;
        }
        int32_t Rank() const {
            return mRank;
        }

        static OperandBase* MakeError(ModelBuilderBase* modelBuilder);
        virtual MaybeError ValidateAndInferTypes() {
            UNREACHABLE();
        }

      private:
        OperandBase(ModelBuilderBase* modelBuilder, ObjectBase::ErrorTag tag);

      protected:
        // The inputs of operand.
        std::vector<Ref<OperandBase>> mInputs;
        // The operand type.
        webnn::OperandType mType;
        // only set rank for dimensions
        int32_t mRank;
    };
}  // namespace webnn_native

#endif  // WEBNN_NATIVE_OPERAND_H_
