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

namespace webnn_native { namespace op {

    class ClampBase {
      public:
        ClampBase(ClampOptions const* options) {
            if (options != nullptr) {
                mOptions = *options;
            } else {
                mOptions.minValue = nullptr;
                mOptions.maxValue = nullptr;
            }
        }
        ~ClampBase() = default;

        ClampOptions const* GetOptions() const {
            return &mOptions;
        }

      private:
        ClampOptions mOptions;
    };

    class Clamp final : public ClampBase, public OperatorBase {
      public:
        Clamp(GraphBuilderBase* builder, OperandBase* input, ClampOptions const* options)
            : ClampBase(options), OperatorBase(builder, {input}) {
            if (options != nullptr) {
                if (options->minValue != nullptr) {
                    mInputs.push_back(options->minValue);
                }
                if (options->maxValue != nullptr) {
                    mInputs.push_back(options->maxValue);
                }
            }
        }
        Clamp(GraphBuilderBase* builder, ClampOptions const* options)
            : ClampBase(options), OperatorBase(builder, FusedOperator::Clamp) {
        }
        ~Clamp() override = default;

        MaybeError AddToGraph(GraphBase* graph) const override {
            return graph->AddClamp(this);
        }
    };

}}  // namespace webnn_native::op

#endif  // WEBNN_NATIVE_OPS_CLAMP_H_