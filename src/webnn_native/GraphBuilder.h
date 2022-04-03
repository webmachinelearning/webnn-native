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
#include "webnn_native/Operand.h"
#include "webnn_native/Operator.h"
#include "webnn_native/webnn_platform.h"

#include <functional>
#include <vector>

namespace webnn_native {

    class GraphBuilderBase : public ObjectBase {
      public:
        GraphBuilderBase(ContextBase* context);
        virtual ~GraphBuilderBase() = default;

        // WebNN API
        OperandBase* Abs(OperandBase*);
        OperandBase* Add(OperandBase*, OperandBase*);
        OperandBase* AveragePool2d(OperandBase*, Pool2dOptions const* options);
        OperandBase* BatchNorm(OperandBase*,
                               OperandBase*,
                               OperandBase*,
                               BatchNormOptions const* options);
        OperandBase* Clamp(OperandBase*, ClampOptions const* options);
        FusionOperatorBase* ClampOperator(ClampOptions const* options);
        OperandBase* Ceil(OperandBase*);
        OperandBase* Concat(uint32_t inputsCount, OperandBase* const* inputs, uint32_t axis);
        OperandBase* Constant(OperandDescriptor const* desc, ArrayBufferView const* arrayBuffer);
        OperandBase* ConstantWithGpuBuffer(OperandDescriptor const* desc,
                                           GpuBufferView const* arrayBuffer);
        OperandBase* Conv2d(OperandBase*, OperandBase*, Conv2dOptions const* options);
        OperandBase* ConvTranspose2d(OperandBase*,
                                     OperandBase*,
                                     ConvTranspose2dOptions const* options);
        OperandBase* Cos(OperandBase*);
        OperandBase* Div(OperandBase*, OperandBase*);
        OperandBase* Exp(OperandBase*);
        OperandBase* Floor(OperandBase*);
        OperandBase* Gemm(OperandBase*, OperandBase*, GemmOptions const* options);
        OperandArrayBase* Gru(OperandBase*,
                              OperandBase*,
                              OperandBase*,
                              int32_t steps,
                              int32_t hiddenSize,
                              GruOptions const* options);
        OperandBase* HardSwish(OperandBase*);
        FusionOperatorBase* HardSwishOperator();
        OperandBase* Input(char const* name, OperandDescriptor const* desc);
        OperandBase* InstanceNorm(OperandBase*, InstanceNormOptions const* options);
        OperandBase* LeakyRelu(OperandBase*, LeakyReluOptions const* options);
        FusionOperatorBase* LeakyReluOperator(LeakyReluOptions const* options);
        OperandBase* Log(OperandBase*);
        OperandBase* L2Pool2d(OperandBase*, Pool2dOptions const* options);
        OperandBase* Matmul(OperandBase* a, OperandBase* b);
        OperandBase* Max(OperandBase*, OperandBase*);
        OperandBase* MaxPool2d(OperandBase*, Pool2dOptions const* options);
        OperandBase* Min(OperandBase*, OperandBase*);
        OperandBase* Mul(OperandBase*, OperandBase*);
        OperandBase* Neg(OperandBase*);
        // OperandBase* Pad(OperandBase*, uint32_t const*, size_t, PadOptions const*);
        OperandBase* Pad(OperandBase*, OperandBase*, PadOptions const* options);
        OperandBase* Pow(OperandBase*, OperandBase*);
        OperandBase* ReduceArgMax(OperandBase*, ReduceOptions const* options);
        OperandBase* ReduceArgMin(OperandBase*, ReduceOptions const* options);
        OperandBase* ReduceL1(OperandBase*, ReduceOptions const* options);
        OperandBase* ReduceL2(OperandBase*, ReduceOptions const* options);
        OperandBase* ReduceMax(OperandBase*, ReduceOptions const* options);
        OperandBase* ReduceMean(OperandBase*, ReduceOptions const* options);
        OperandBase* ReduceMin(OperandBase*, ReduceOptions const* options);
        OperandBase* ReduceProduct(OperandBase*, ReduceOptions const* options);
        OperandBase* ReduceSum(OperandBase*, ReduceOptions const* options);
        OperandBase* Relu(OperandBase*);
        FusionOperatorBase* ReluOperator();
        OperandBase* Resample2d(OperandBase*, Resample2dOptions const* options);
        OperandBase* Reshape(OperandBase*, int32_t const*, size_t);
        OperandBase* Sigmoid(OperandBase*);
        FusionOperatorBase* SigmoidOperator();
        OperandBase* Sin(OperandBase*);
        OperandBase* Slice(OperandBase*,
                           int32_t const* starts,
                           uint32_t startsCount,
                           int32_t const* sizes,
                           uint32_t sizesCount,
                           SliceOptions const* options);
        OperandBase* Softmax(OperandBase*);
        OperandArrayBase* Split(OperandBase*,
                                uint32_t const*,
                                uint32_t,
                                SplitOptions const* options);
        OperandBase* Squeeze(OperandBase*, SqueezeOptions const* options);
        OperandBase* Sub(OperandBase*, OperandBase*);
        OperandBase* Tan(OperandBase*);
        OperandBase* Tanh(OperandBase*);
        FusionOperatorBase* TanhOperator();
        OperandBase* Transpose(OperandBase*, TransposeOptions const* options);

        GraphBase* Build(NamedOperandsBase const* namedOperands);

      private:
        ResultOrError<Ref<GraphBase>> BuildImpl(NamedOperandsBase const* namedOperands);

        // Topological sort of nodes needed to compute rootNodes
        std::vector<const OperatorBase*> TopologicalSort(
            std::vector<const OperandBase*>& rootNodes);
    };

}  // namespace webnn_native

#endif  // WEBNN_NATIVE_MODEL_BUILDER_H_
