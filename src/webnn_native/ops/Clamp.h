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

#ifndef WEBNN_NATIVE_OPS_CLAMP_H_
#define WEBNN_NATIVE_OPS_CLAMP_H_

#include "webnn_native/Graph.h"
#include "webnn_native/Operand.h"
#include "webnn_native/Operator.h"
#include "webnn_native/FusionOperator.h"

namespace webnn_native { namespace op {

    class ClampBase {
      public:
        ClampBase(ClampOptions const* options) {
            mMinValue =
                options == nullptr ? std::numeric_limits<float>::lowest() : options->minValue;
            mMaxValue = options == nullptr ? std::numeric_limits<float>::max() : options->maxValue;
        }
        ~ClampBase() = default;

        float GetMinValue() const {
            return mMinValue;
        }
        float GetMaxValue() const {
            return mMaxValue;
        }

      private:
        float mMinValue;
        float mMaxValue;
    };

    class Clamp final : public ClampBase, public OperatorBase {
      public:
        Clamp(GraphBuilderBase* builder, OperandBase* input, ClampOptions const* options)
            : ClampBase(options), OperatorBase(builder, {input}) {
        }
        ~Clamp() override = default;

        MaybeError AddToGraph(GraphBase* graph) const override {
            return graph->AddClamp(this);
        }

        MaybeError ValidateAndInferOutputInfo() override {
            MaybeError maybeError = OperatorBase::ValidateAndInferOutputInfo();
            if (maybeError.IsError()) {
                return maybeError;
            }
            if (!mInputs.empty()) {
                mOutputs[0]->SetShape(mInputs[0]->Shape());
            }

            return {};
        }
    };

    class FusionClamp final : public ClampBase, public FusionOperatorBase {
      public:
        FusionClamp(GraphBuilderBase* builder, ClampOptions const* options)
            : ClampBase(options), FusionOperatorBase(builder, FusionType::Clamp) {
        }
    };

}}  // namespace webnn_native::op

#endif  // WEBNN_NATIVE_OPS_CLAMP_H_