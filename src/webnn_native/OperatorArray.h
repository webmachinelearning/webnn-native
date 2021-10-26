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

#ifndef WEBNN_NATIVE_OPERATOR_ARRAY_H_
#define WEBNN_NATIVE_OPERATOR_ARRAY_H_

#include "webnn_native/Operator.h"

namespace webnn_native {

    class OperatorArrayBase : public RefCounted {
      public:
        OperatorArrayBase() = default;
        virtual ~OperatorArrayBase() = default;

        // WebNN API
        size_t Size() {
            return mOperators.size();
        }

        void Set(OperatorBase* mlOperator) {
            mOperators.push_back(Ref<OperatorBase>(mlOperator));
        }

        OperatorBase* Get(size_t index) {
            return mOperators[index].Get();
        }

        std::vector<Ref<OperatorBase>> mOperators;
    };
}  // namespace webnn_native

#endif  // WEBNN_NATIVE_OPERATOR_ARRAY_H_
