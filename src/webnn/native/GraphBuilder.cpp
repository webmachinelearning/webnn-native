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

#include "webnn/native/GraphBuilder.h"

#include <stack>
#include <string>
#include <unordered_set>
#include <vector>

#include "common/Assert.h"
#include "common/Log.h"
#include "common/RefCounted.h"
#include "webnn/native/Context.h"
#include "webnn/native/Graph.h"
#include "webnn/native/Operand.h"
#include "webnn/native/OperandArray.h"
#include "webnn/native/Operator.h"
#include "webnn/native/ops/BatchNorm.h"
#include "webnn/native/ops/Binary.h"
#include "webnn/native/ops/Clamp.h"
#include "webnn/native/ops/Concat.h"
#include "webnn/native/ops/Constant.h"
#include "webnn/native/ops/Conv2d.h"
#include "webnn/native/ops/Gemm.h"
#include "webnn/native/ops/Gru.h"
#include "webnn/native/ops/Input.h"
#include "webnn/native/ops/InstanceNorm.h"
#include "webnn/native/ops/LeakyRelu.h"
#include "webnn/native/ops/Pad.h"
#include "webnn/native/ops/Pool2d.h"
#include "webnn/native/ops/Reduce.h"
#include "webnn/native/ops/Resample2d.h"
#include "webnn/native/ops/Reshape.h"
#include "webnn/native/ops/Slice.h"
#include "webnn/native/ops/Split.h"
#include "webnn/native/ops/Squeeze.h"
#include "webnn/native/ops/Transpose.h"
#include "webnn/native/ops/Unary.h"

#define WEBNN_VALIDATE(ptr, objectBase)                                  \
    Ref<OperatorBase> op = AcquireRef(ptr);                              \
    if (GetContext()->ConsumedError(op->ValidateAndInferOutputInfo())) { \
        return objectBase::MakeError(this);                              \
    }                                                                    \
    mOperators.push_back(op);                                            \
    for (;;)                                                             \
    break

#define VALIDATE_FOR_OPERAND(ptr)     \
    WEBNN_VALIDATE(ptr, OperandBase); \
    return op->PrimaryOutput()
#define VALIDATE_ARRAY_OPERAND(ptr)        \
    WEBNN_VALIDATE(ptr, OperandArrayBase); \
    return new OperandArrayBase(this, op->Outputs())

namespace webnn::native {

    GraphBuilderBase::GraphBuilderBase(ContextBase* context) : ObjectBase(context) {
    }

    OperandBase* GraphBuilderBase::APIAbs(OperandBase* input) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kAbs, input));
    }

    OperandBase* GraphBuilderBase::APIAdd(OperandBase* a, OperandBase* b) {
        VALIDATE_FOR_OPERAND(new op::Binary(this, op::BinaryOpType::kAdd, a, b));
    }

    OperandBase* GraphBuilderBase::APIAveragePool2d(OperandBase* input,
                                                    Pool2dOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Pool2d(this, op::Pool2dType::kAveragePool2d, input, options));
    }

    OperandBase* GraphBuilderBase::APIBatchNorm(OperandBase* input,
                                                OperandBase* mean,
                                                OperandBase* variance,
                                                BatchNormOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::BatchNorm(this, input, mean, variance, options));
    }

    OperandBase* GraphBuilderBase::APIClamp(OperandBase* input, ClampOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Clamp(this, input, options));
    }

    FusionOperatorBase* GraphBuilderBase::APIClampOperator(ClampOptions const* options) {
        return new op::FusionClamp(this, options);
    }

    OperandBase* GraphBuilderBase::APICeil(OperandBase* input) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kCeil, input));
    }

    OperandBase* GraphBuilderBase::APIConcat(uint32_t inputsCount,
                                             OperandBase* const* inputs,
                                             uint32_t axis) {
        std::vector<Ref<OperandBase>> operandInputs;
        operandInputs.reserve(inputsCount);
        for (uint32_t i = 0; i < inputsCount; ++i) {
            operandInputs.push_back(inputs[i]);
        }
        VALIDATE_FOR_OPERAND(new op::Concat(this, std::move(operandInputs), axis));
    }

    OperandBase* GraphBuilderBase::APIConstant(OperandDescriptor const* desc,
                                               ArrayBufferView const* arrayBuffer) {
        VALIDATE_FOR_OPERAND(new op::Constant(this, desc, arrayBuffer));
    }

    OperandBase* GraphBuilderBase::APIConstantWithGpuBuffer(OperandDescriptor const* desc,
                                                            GpuBufferView const* gpuBuffer) {
#if defined(WEBNN_ENABLE_GPU_BUFFER)
        VALIDATE_FOR_OPERAND(new op::Constant(this, desc, gpuBuffer));
#endif
        UNREACHABLE();
        return nullptr;
    }

    OperandBase* GraphBuilderBase::APIConv2d(OperandBase* input,
                                             OperandBase* filter,
                                             Conv2dOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Conv2d(this, input, filter, options));
    }

    OperandBase* GraphBuilderBase::APIConvTranspose2d(OperandBase* input,
                                                      OperandBase* filter,
                                                      ConvTranspose2dOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::ConvTranspose2d(this, input, filter, options));
    }

    OperandBase* GraphBuilderBase::APICos(OperandBase* input) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kCos, input));
    }

    OperandBase* GraphBuilderBase::APIDiv(OperandBase* a, OperandBase* b) {
        VALIDATE_FOR_OPERAND(new op::Binary(this, op::BinaryOpType::kDiv, a, b));
    }

    OperandBase* GraphBuilderBase::APIExp(OperandBase* input) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kExp, input));
    }

    OperandBase* GraphBuilderBase::APIFloor(OperandBase* input) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kFloor, input));
    }

    OperandBase* GraphBuilderBase::APIGemm(OperandBase* a,
                                           OperandBase* b,
                                           GemmOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Gemm(this, a, b, options));
    }

    OperandArrayBase* GraphBuilderBase::APIGru(OperandBase* input,
                                               OperandBase* weight,
                                               OperandBase* recurrentWeight,
                                               int32_t steps,
                                               int32_t hiddenSize,
                                               GruOptions const* options) {
        VALIDATE_ARRAY_OPERAND(
            new op::Gru(this, input, weight, recurrentWeight, steps, hiddenSize, options));
    }

    OperandBase* GraphBuilderBase::APIHardSwish(OperandBase* input) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kHardSwish, input));
    }

    FusionOperatorBase* GraphBuilderBase::APIHardSwishOperator() {
        return new op::FusionUnary(this, FusionType::HardSwish);
    }

    OperandBase* GraphBuilderBase::APIInput(char const* name, OperandDescriptor const* desc) {
        VALIDATE_FOR_OPERAND(new op::Input(this, std::string(name), desc));
    }

    OperandBase* GraphBuilderBase::APIInstanceNorm(OperandBase* input,
                                                   InstanceNormOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::InstanceNorm(this, input, options));
    }

    OperandBase* GraphBuilderBase::APILeakyRelu(OperandBase* input,
                                                LeakyReluOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::LeakyRelu(this, input, options));
    }

    FusionOperatorBase* GraphBuilderBase::APILeakyReluOperator(LeakyReluOptions const* options) {
        return new op::FusionLeakyRelu(this, options);
    }

    OperandBase* GraphBuilderBase::APILog(OperandBase* input) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kLog, input));
    }

    OperandBase* GraphBuilderBase::APIL2Pool2d(OperandBase* input, Pool2dOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Pool2d(this, op::Pool2dType::kL2Pool2d, input, options));
    }

    OperandBase* GraphBuilderBase::APIMatmul(OperandBase* a, OperandBase* b) {
        VALIDATE_FOR_OPERAND(new op::Binary(this, op::BinaryOpType::kMatMul, a, b));
    }

    OperandBase* GraphBuilderBase::APIMax(OperandBase* a, OperandBase* b) {
        VALIDATE_FOR_OPERAND(new op::Binary(this, op::BinaryOpType::kMax, a, b));
    }

    OperandBase* GraphBuilderBase::APIMaxPool2d(OperandBase* input, Pool2dOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Pool2d(this, op::Pool2dType::kMaxPool2d, input, options));
    }

    OperandBase* GraphBuilderBase::APIMin(OperandBase* a, OperandBase* b) {
        VALIDATE_FOR_OPERAND(new op::Binary(this, op::BinaryOpType::kMin, a, b));
    }

    OperandBase* GraphBuilderBase::APIMul(OperandBase* a, OperandBase* b) {
        VALIDATE_FOR_OPERAND(new op::Binary(this, op::BinaryOpType::kMul, a, b));
    }

    OperandBase* GraphBuilderBase::APINeg(OperandBase* input) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kNeg, input));
    }

    // OperandBase* GraphBuilderBase::Pad(OperandBase* input,
    //                                    uint32_t const* padding,
    //                                    size_t padding_count,
    //                                    PadOptions const* options) {
    //     VALIDATE_FOR_OPERAND(new op::Pad(this, input, padding, padding_count, options));
    // }
    OperandBase* GraphBuilderBase::APIPad(OperandBase* input,
                                          OperandBase* padding,
                                          PadOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Pad(this, input, padding, options));
    }

    OperandBase* GraphBuilderBase::APIPow(OperandBase* a, OperandBase* b) {
        VALIDATE_FOR_OPERAND(new op::Binary(this, op::BinaryOpType::kPower, a, b));
    }

    OperandBase* GraphBuilderBase::APIReduceArgMax(OperandBase* input,
                                                   ReduceOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Reduce(this, op::ReduceType::kReduceArgMax, input, options));
    }

    OperandBase* GraphBuilderBase::APIReduceArgMin(OperandBase* input,
                                                   ReduceOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Reduce(this, op::ReduceType::kReduceArgMin, input, options));
    }

    OperandBase* GraphBuilderBase::APIReduceL2(OperandBase* input, ReduceOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Reduce(this, op::ReduceType::kReduceL2, input, options));
    }

    OperandBase* GraphBuilderBase::APIReduceL1(OperandBase* input, ReduceOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Reduce(this, op::ReduceType::kReduceL1, input, options));
    }

    OperandBase* GraphBuilderBase::APIReduceMax(OperandBase* input, ReduceOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Reduce(this, op::ReduceType::kReduceMax, input, options));
    }

    OperandBase* GraphBuilderBase::APIReduceMean(OperandBase* input, ReduceOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Reduce(this, op::ReduceType::kReduceMean, input, options));
    }

    OperandBase* GraphBuilderBase::APIReduceMin(OperandBase* input, ReduceOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Reduce(this, op::ReduceType::kReduceMin, input, options));
    }

    OperandBase* GraphBuilderBase::APIReduceProduct(OperandBase* input,
                                                    ReduceOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Reduce(this, op::ReduceType::kReduceProduct, input, options));
    }

    OperandBase* GraphBuilderBase::APIReduceSum(OperandBase* input, ReduceOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Reduce(this, op::ReduceType::kReduceSum, input, options));
    }

    OperandBase* GraphBuilderBase::APIRelu(OperandBase* input) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kRelu, input));
    }

    FusionOperatorBase* GraphBuilderBase::APIReluOperator() {
        return new op::FusionUnary(this, FusionType::Relu);
    }

    OperandBase* GraphBuilderBase::APIResample2d(OperandBase* input,
                                                 Resample2dOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Resample2d(this, input, options));
    }

    OperandBase* GraphBuilderBase::APIReshape(OperandBase* input,
                                              int32_t const* new_shape,
                                              size_t new_shape_count) {
        VALIDATE_FOR_OPERAND(new op::Reshape(this, input, new_shape, new_shape_count));
    }

    OperandBase* GraphBuilderBase::APISigmoid(OperandBase* input) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kSigmoid, input));
    }

    FusionOperatorBase* GraphBuilderBase::APISigmoidOperator() {
        return new op::FusionUnary(this, FusionType::Sigmoid);
    }

    OperandBase* GraphBuilderBase::APISin(OperandBase* input) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kSin, input));
    }

    OperandBase* GraphBuilderBase::APISlice(OperandBase* input,
                                            int32_t const* starts,
                                            uint32_t startsCount,
                                            int32_t const* sizes,
                                            uint32_t sizesCount,
                                            SliceOptions const* options) {
        VALIDATE_FOR_OPERAND(
            new op::Slice(this, input, starts, startsCount, sizes, sizesCount, options));
    }

    OperandBase* GraphBuilderBase::APISoftmax(OperandBase* input) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kSoftmax, input));
    }

    OperandArrayBase* GraphBuilderBase::APISplit(OperandBase* input,
                                                 uint32_t const* splits,
                                                 uint32_t splitsCount,
                                                 SplitOptions const* options) {
        VALIDATE_ARRAY_OPERAND(new op::Split(this, input, splits, splitsCount, options));
    }

    OperandBase* GraphBuilderBase::APISqueeze(OperandBase* input, SqueezeOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Squeeze(this, input, options));
    }

    OperandBase* GraphBuilderBase::APISub(OperandBase* a, OperandBase* b) {
        VALIDATE_FOR_OPERAND(new op::Binary(this, op::BinaryOpType::kSub, a, b));
    }

    OperandBase* GraphBuilderBase::APITan(OperandBase* input) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kTan, input));
    }

    OperandBase* GraphBuilderBase::APITanh(OperandBase* input) {
        VALIDATE_FOR_OPERAND(new op::Unary(this, op::UnaryOpType::kTanh, input));
    }

    FusionOperatorBase* GraphBuilderBase::APITanhOperator() {
        return new op::FusionUnary(this, FusionType::Tanh);
    }

    OperandBase* GraphBuilderBase::APITranspose(OperandBase* input,
                                                TransposeOptions const* options) {
        VALIDATE_FOR_OPERAND(new op::Transpose(this, input, options));
    }

    ResultOrError<Ref<GraphBase>> GraphBuilderBase::BuildImpl(
        NamedOperandsBase const* namedOperands) {
        DAWN_INVALID_IF(this->IsError(), "The GraphBuilderBase is an error object.");
        DAWN_INVALID_IF(namedOperands->GetRecords().empty(), "The namedOperands are empty.");

        std::vector<const OperandBase*> outputs;
        for (auto& namedOutput : namedOperands->GetRecords()) {
            outputs.push_back(namedOutput.second);
        }
        std::vector<const OperatorBase*> sorted_operands = TopologicalSort(outputs);
        DAWN_INVALID_IF(sorted_operands.empty(), "The graph can't be built.");
        Ref<GraphBase> graph = AcquireRef(GetContext()->CreateGraph());
        for (auto& op : sorted_operands) {
            DAWN_INVALID_IF(op->IsError(), "The operand is an error object.");
            DAWN_TRY(op->AddToGraph(graph.Get()));
        }
        for (auto& [name, output] : namedOperands->GetRecords()) {
            DAWN_TRY(graph->AddOutput(name, output));
        }
        DAWN_TRY(graph->Finish());
        DAWN_TRY(graph->Compile());

        return std::move(graph);
    }

    GraphBase* GraphBuilderBase::APIBuild(NamedOperandsBase const* namedOperands) {
        Ref<GraphBase> result = nullptr;
        if (GetContext()->ConsumedError(BuildImpl(namedOperands), &result)) {
            ASSERT(result == nullptr);
            return GraphBase::MakeError(this->GetContext());
        }
        return result.Detach();
    }

    // The implementation derives from nGraph topological_sort in
    // https://github.com/openvinotoolkit/openvino/blob/master/ngraph/core/include/ngraph/graph_util.hpp
    //
    //*****************************************************************************
    // Copyright 2017-2020 Intel Corporation
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
    //*****************************************************************************
    std::vector<const OperatorBase*> GraphBuilderBase::TopologicalSort(
        std::vector<const OperandBase*>& rootNodes) {
        std::stack<const OperatorBase*> nodesToDo;
        std::unordered_set<const OperatorBase*> nodesDone;
        std::vector<const OperatorBase*> result;

        for (auto node : rootNodes) {
            if (node->IsError()) {
                return {};
            }
            nodesToDo.push(const_cast<OperandBase*>(node)->Operator());
        }
        while (nodesToDo.size() > 0) {
            const OperatorBase* node = nodesToDo.top();
            if (nodesDone.count(node) == 0) {
                bool can_add = true;
                for (auto& dep : node->Inputs()) {
                    if (nodesDone.count(dep->Operator()) == 0) {
                        can_add = false;
                        nodesToDo.push(dep->Operator());
                    }
                }
                if (can_add) {
                    result.push_back(node);
                    nodesToDo.pop();
                    nodesDone.insert(node);
                }
            } else {
                nodesToDo.pop();
            }
        }
        return result;
    }

}  // namespace webnn::native
