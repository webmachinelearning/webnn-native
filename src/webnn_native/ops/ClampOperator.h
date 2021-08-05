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

#ifndef WEBNN_NATIVE_OPS_CLAMPOPERATOR_H_
#define WEBNN_NATIVE_OPS_CLAMPOPERATOR_H_

#include "webnn_native/Graph.h"
#include "webnn_native/Operator.h"

namespace webnn_native { namespace op {

    class ClampOperator final : public OperatorBase {
      public:
        ClampOperator(GraphBuilderBase* builder, ClampOptions const* options);
        ~ClampOperator() override = default;
        ClampOptions const* GetOptions() const {
            return &mOptions;
        }

        OperatorType GetOperatorType() const override {
            return mType;
        }

      private:
        ClampOptions mOptions;
        OperatorType mType;
    };

}}  // namespace webnn_native::op

#endif  // WEBNN_NATIVE_OPS_CLAMPOPERATOR_H_