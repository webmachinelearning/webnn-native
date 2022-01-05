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

#ifndef WEBNN_NATIVE_OPS_LEAKYRELU_H_
#define WEBNN_NATIVE_OPS_LEAKYRELU_H_

#include "webnn_native/Operator.h"
#include "webnn_native/ops/Unary.h"

namespace webnn_native { namespace op {

    class LeakyReluBase {
      public:
        LeakyReluBase(LeakyReluOptions const* options) {
            mAlpha = options == nullptr ? 0.01 : options->alpha;
        }
        ~LeakyReluBase() = default;

        float GetAlpha() const {
            return mAlpha;
        }

      private:
        float mAlpha;
    };

    class LeakyRelu final : public LeakyReluBase, public Unary {
      public:
        LeakyRelu(GraphBuilderBase* builder, OperandBase* input, LeakyReluOptions const* options)
            : LeakyReluBase(options), Unary(builder, kLeakyRelu, input) {
        }
        ~LeakyRelu() override = default;
    };

    class FusionLeakyRelu final : public LeakyReluBase, public FusionOperatorBase {
      public:
        FusionLeakyRelu(GraphBuilderBase* builder, LeakyReluOptions const* options)
            : LeakyReluBase(options), FusionOperatorBase(builder, FusionType::LeakyRelu) {
        }
    };

}}  // namespace webnn_native::op

#endif  // WEBNN_NATIVE_OPS_LEAKYRELU_H_
