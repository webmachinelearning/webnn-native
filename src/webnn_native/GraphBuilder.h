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

#ifndef WEBNN_NATIVE_MODEL_BUILDER_H_
#define WEBNN_NATIVE_MODEL_BUILDER_H_

#include "common/RefCounted.h"
#include "webnn_native/Forward.h"
#include "webnn_native/NamedOperands.h"
#include "webnn_native/ObjectBase.h"
#include "webnn_native/webnn_platform.h"

#include <functional>
#include <vector>

namespace webnn_native {

    class GraphBuilderBase : public ObjectBase {
      public:
        GraphBuilderBase(ContextBase* context);
        virtual ~GraphBuilderBase() = default;

        // WebNN API
        OperandBase* Constant(OperandDescriptor const* desc, ArrayBufferView const* arrayBuffer);
        OperandBase* Input(char const* name, OperandDescriptor const* desc);
        OperandBase* Matmul(OperandBase* a, OperandBase* b);
        OperandBase* Add(OperandBase*, OperandBase*);
        OperandBase* Div(OperandBase*, OperandBase*);
        OperandBase* Mul(OperandBase*, OperandBase*);
        OperandBase* Sub(OperandBase*, OperandBase*);
        OperandBase* Max(OperandBase*, OperandBase*);
        OperandBase* Min(OperandBase*, OperandBase*);
        OperandBase* Pow(OperandBase*, OperandBase*);
        OperandBase* Conv2d(OperandBase*, OperandBase*, Conv2dOptions const* options);
        OperandBase* AveragePool2d(OperandBase*, Pool2dOptions const* options);
        OperandBase* MaxPool2d(OperandBase*, Pool2dOptions const* options);
        OperandBase* Pad(OperandBase*, OperandBase*, PadOptions const* options);
        OperandBase* ReduceMean(OperandBase*, ReduceMeanOptions const* options);
        OperandBase* Resample(OperandBase*, ResampleOptions const* options);
        OperandBase* Relu(OperandBase*);
        OperandBase* HardSwish(OperandBase*);
        OperatorBase* HardSwishOperator();
        OperatorBase* ReluOperator();
        OperandBase* Reshape(OperandBase*, int32_t const*, size_t);
        OperandBase* Sigmoid(OperandBase*);
        OperatorBase* SigmoidOperator();
        OperandBase* Softmax(OperandBase*);
        OperandBase* Squeeze(OperandBase*, SqueezeOptions const* options);
        OperandArrayBase* Split(OperandBase*,
                                uint32_t const*,
                                uint32_t,
                                SplitOptions const* options);
        OperandBase* Tanh(OperandBase*);
        OperandBase* Transpose(OperandBase*, TransposeOptions const* options);
        OperandBase* LeakyRelu(OperandBase*, LeakyReluOptions const* options);
        OperatorBase* LeakyReluOperator(LeakyReluOptions const* options);
        OperandBase* Concat(uint32_t inputsCount, OperandBase* const* inputs, uint32_t axis);
        OperandBase* Gemm(OperandBase*, OperandBase*, GemmOptions const* options);
        OperandBase* Clamp(OperandBase*, ClampOptions const* options);
        OperatorBase* ClampOperator(ClampOptions const* options);
        OperandBase* BatchNorm(OperandBase*,
                               OperandBase*,
                               OperandBase*,
                               BatchNormOptions const* options);
        OperandBase* InstanceNorm(OperandBase*, InstanceNormOptions const* options);
        GraphBase* Build(NamedOperandsBase const* namedOperands);

      private:
        // Topological sort of nodes needed to compute rootNodes
        std::vector<const OperatorBase*> TopologicalSort(
            std::vector<const OperandBase*>& rootNodes);
    };

}  // namespace webnn_native

#endif  // WEBNN_NATIVE_MODEL_BUILDER_H_
