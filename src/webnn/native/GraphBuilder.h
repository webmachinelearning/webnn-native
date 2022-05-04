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
#include "webnn/native/Forward.h"
#include "webnn/native/NamedOperands.h"
#include "webnn/native/ObjectBase.h"
#include "webnn/native/Operand.h"
#include "webnn/native/Operator.h"
#include "webnn/native/webnn_platform.h"

#include <functional>
#include <vector>

namespace webnn::native {

    class GraphBuilderBase : public ObjectBase {
      public:
        GraphBuilderBase(ContextBase* context);
        virtual ~GraphBuilderBase() = default;

        // WebNN API
        OperandBase* APIAbs(OperandBase*);
        OperandBase* APIAdd(OperandBase*, OperandBase*);
        OperandBase* APIAveragePool2d(OperandBase*, Pool2dOptions const* options);
        OperandBase* APIBatchNorm(OperandBase*,
                                  OperandBase*,
                                  OperandBase*,
                                  BatchNormOptions const* options);
        OperandBase* APIClamp(OperandBase*, ClampOptions const* options);
        FusionOperatorBase* APIClampOperator(ClampOptions const* options);
        OperandBase* APICeil(OperandBase*);
        OperandBase* APIConcat(uint32_t inputsCount, OperandBase* const* inputs, uint32_t axis);
        OperandBase* APIConstant(OperandDescriptor const* desc, ArrayBufferView const* arrayBuffer);
        OperandBase* APIConstantWithGpuBuffer(OperandDescriptor const* desc,
                                              GpuBufferView const* arrayBuffer);
        OperandBase* APIConv2d(OperandBase*, OperandBase*, Conv2dOptions const* options);
        OperandBase* APIConvTranspose2d(OperandBase*,
                                        OperandBase*,
                                        ConvTranspose2dOptions const* options);
        OperandBase* APICos(OperandBase*);
        OperandBase* APIDiv(OperandBase*, OperandBase*);
        OperandBase* APIExp(OperandBase*);
        OperandBase* APIFloor(OperandBase*);
        OperandBase* APIGemm(OperandBase*, OperandBase*, GemmOptions const* options);
        OperandArrayBase* APIGru(OperandBase*,
                                 OperandBase*,
                                 OperandBase*,
                                 int32_t steps,
                                 int32_t hiddenSize,
                                 GruOptions const* options);
        OperandBase* APIHardSwish(OperandBase*);
        FusionOperatorBase* APIHardSwishOperator();
        OperandBase* APIInput(char const* name, OperandDescriptor const* desc);
        OperandBase* APIInstanceNorm(OperandBase*, InstanceNormOptions const* options);
        OperandBase* APILeakyRelu(OperandBase*, LeakyReluOptions const* options);
        FusionOperatorBase* APILeakyReluOperator(LeakyReluOptions const* options);
        OperandBase* APILog(OperandBase*);
        OperandBase* APIL2Pool2d(OperandBase*, Pool2dOptions const* options);
        OperandBase* APIMatmul(OperandBase* a, OperandBase* b);
        OperandBase* APIMax(OperandBase*, OperandBase*);
        OperandBase* APIMaxPool2d(OperandBase*, Pool2dOptions const* options);
        OperandBase* APIMin(OperandBase*, OperandBase*);
        OperandBase* APIMul(OperandBase*, OperandBase*);
        OperandBase* APINeg(OperandBase*);
        OperandBase* APIPad(OperandBase*, OperandBase*, PadOptions const* options);
        OperandBase* APIPow(OperandBase*, OperandBase*);
        OperandBase* APIReduceArgMax(OperandBase*, ReduceOptions const* options);
        OperandBase* APIReduceArgMin(OperandBase*, ReduceOptions const* options);
        OperandBase* APIReduceL1(OperandBase*, ReduceOptions const* options);
        OperandBase* APIReduceL2(OperandBase*, ReduceOptions const* options);
        OperandBase* APIReduceMax(OperandBase*, ReduceOptions const* options);
        OperandBase* APIReduceMean(OperandBase*, ReduceOptions const* options);
        OperandBase* APIReduceMin(OperandBase*, ReduceOptions const* options);
        OperandBase* APIReduceProduct(OperandBase*, ReduceOptions const* options);
        OperandBase* APIReduceSum(OperandBase*, ReduceOptions const* options);
        OperandBase* APIRelu(OperandBase*);
        FusionOperatorBase* APIReluOperator();
        OperandBase* APIResample2d(OperandBase*, Resample2dOptions const* options);
        OperandBase* APIReshape(OperandBase*, int32_t const*, size_t);
        OperandBase* APISigmoid(OperandBase*);
        FusionOperatorBase* APISigmoidOperator();
        OperandBase* APISin(OperandBase*);
        OperandBase* APISlice(OperandBase*,
                              int32_t const* starts,
                              uint32_t startsCount,
                              int32_t const* sizes,
                              uint32_t sizesCount,
                              SliceOptions const* options);
        OperandBase* APISoftmax(OperandBase*);
        OperandArrayBase* APISplit(OperandBase*,
                                   uint32_t const*,
                                   uint32_t,
                                   SplitOptions const* options);
        OperandBase* APISqueeze(OperandBase*, SqueezeOptions const* options);
        OperandBase* APISub(OperandBase*, OperandBase*);
        OperandBase* APITan(OperandBase*);
        OperandBase* APITanh(OperandBase*);
        FusionOperatorBase* APITanhOperator();
        OperandBase* APITranspose(OperandBase*, TransposeOptions const* options);

        GraphBase* APIBuild(NamedOperandsBase const* namedOperands);

      private:
        ResultOrError<Ref<GraphBase>> BuildImpl(NamedOperandsBase const* namedOperands);

        std::vector<Ref<OperatorBase>> mOperators;
        // Topological sort of nodes needed to compute rootNodes
        std::vector<const OperatorBase*> TopologicalSort(
            std::vector<const OperandBase*>& rootNodes);
    };

}  // namespace webnn::native

#endif  // WEBNN_NATIVE_MODEL_BUILDER_H_
