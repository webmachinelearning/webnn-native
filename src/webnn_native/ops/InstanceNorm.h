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

#ifndef WEBNN_NATIVE_OPS_INSTANCENORM_H_
#define WEBNN_NATIVE_OPS_INSTANCENORM_H_

#include "webnn_native/Graph.h"
#include "webnn_native/Operand.h"

namespace webnn_native { namespace op {

    class InstanceNorm final : public OperatorBase {
      public:
        InstanceNorm(GraphBuilderBase* builder,
                     OperandBase* input,
                     InstanceNormOptions const* options);
        ~InstanceNorm() override = default;

        MaybeError AddToGraph(GraphBase* graph) const override {
            return graph->AddInstanceNorm(this);
        }
        MaybeError ValidateAndInferOutputInfo() override;

        InstanceNormOptions const* GetOptions() const {
            return &mOptions;
        }

      private:
        InstanceNormOptions mOptions;
    };

}}  // namespace webnn_native::op

#endif  // WEBNN_NATIVE_OPS_INSTANCENORM_H_
