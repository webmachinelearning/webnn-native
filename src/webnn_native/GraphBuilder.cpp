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

#include "webnn_native/GraphBuilder.h"

#include <stack>
#include <string>
#include <unordered_set>
#include <vector>

#include "common/Assert.h"
#include "common/Log.h"
#include "common/RefCounted.h"
#include "webnn_native/Context.h"
#include "webnn_native/Graph.h"
#include "webnn_native/Operand.h"
#include "webnn_native/ops/BatchNorm.h"
#include "webnn_native/ops/Binary.h"
#include "webnn_native/ops/Clamp.h"
#include "webnn_native/ops/Concat.h"
#include "webnn_native/ops/Constant.h"
#include "webnn_native/ops/Conv2d.h"
#include "webnn_native/ops/Gemm.h"
#include "webnn_native/ops/Input.h"
#include "webnn_native/ops/InstanceNorm.h"
#include "webnn_native/ops/LeakyRelu.h"
#include "webnn_native/ops/Pad.h"
#include "webnn_native/ops/Pool2d.h"
#include "webnn_native/ops/ReduceMean.h"
#include "webnn_native/ops/Resample.h"
#include "webnn_native/ops/Reshape.h"
#include "webnn_native/ops/Transpose.h"
#include "webnn_native/ops/Unary.h"

#define DAWN_VALIDATE_AND_INFER_TYPES(ptr)                          \
    Ref<OperandBase> op = AcquireRef(ptr);                          \
    if (GetContext()->ConsumedError(op->ValidateAndInferTypes())) { \
        return OperandBase::MakeError(this);                        \
    }                                                               \
    return op.Detach();                                             \
    for (;;)                                                        \
    break

namespace webnn_native {

    GraphBuilderBase::GraphBuilderBase(ContextBase* context) : ObjectBase(context) {
    }

    OperandBase* GraphBuilderBase::Constant(OperandDescriptor const* desc,
                                            ArrayBufferView const* arrayBuffer) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Constant(this, desc, arrayBuffer));
    }

    OperandBase* GraphBuilderBase::Input(char const* name, OperandDescriptor const* desc) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Input(this, std::string(name), desc));
    }

    OperandBase* GraphBuilderBase::Matmul(OperandBase* a, OperandBase* b) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Binary(this, op::BinaryOpType::kMatMul, a, b));
    }

    OperandBase* GraphBuilderBase::Add(OperandBase* a, OperandBase* b) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Binary(this, op::BinaryOpType::kAdd, a, b));
    }

    OperandBase* GraphBuilderBase::Div(OperandBase* a, OperandBase* b) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Binary(this, op::BinaryOpType::kDiv, a, b));
    }

    OperandBase* GraphBuilderBase::Mul(OperandBase* a, OperandBase* b) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Binary(this, op::BinaryOpType::kMul, a, b));
    }

    OperandBase* GraphBuilderBase::Sub(OperandBase* a, OperandBase* b) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Binary(this, op::BinaryOpType::kSub, a, b));
    }

    OperandBase* GraphBuilderBase::Max(OperandBase* a, OperandBase* b) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Binary(this, op::BinaryOpType::kMax, a, b));
    }

    OperandBase* GraphBuilderBase::Conv2d(OperandBase* input,
                                          OperandBase* filter,
                                          Conv2dOptions const* options) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Conv2d(this, input, filter, options));
    }

    OperandBase* GraphBuilderBase::AveragePool2d(OperandBase* input, Pool2dOptions const* options) {
        DAWN_VALIDATE_AND_INFER_TYPES(
            new op::Pool2d(this, op::Pool2dType::kAveragePool2d, input, options));
    }

    OperandBase* GraphBuilderBase::MaxPool2d(OperandBase* input, Pool2dOptions const* options) {
        DAWN_VALIDATE_AND_INFER_TYPES(
            new op::Pool2d(this, op::Pool2dType::kMaxPool2d, input, options));
    }

    OperandBase* GraphBuilderBase::ReduceMean(OperandBase* input,
                                              ReduceMeanOptions const* options) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::ReduceMean(this, input, options));
    }

    OperandBase* GraphBuilderBase::Relu(OperandBase* input) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Unary(this, op::UnaryOpType::kRelu, input));
    }

    OperandBase* GraphBuilderBase::Resample(OperandBase* input, ResampleOptions const* options) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Resample(this, input, options));
    }

    OperandBase* GraphBuilderBase::Reshape(OperandBase* input,
                                           int32_t const* new_shape,
                                           size_t new_shape_count) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Reshape(this, input, new_shape, new_shape_count));
    }

    OperandBase* GraphBuilderBase::Softmax(OperandBase* input) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Unary(this, op::UnaryOpType::kSoftmax, input));
    }

    OperandBase* GraphBuilderBase::Sigmoid(OperandBase* input) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Unary(this, op::UnaryOpType::kSigmoid, input));
    }

    OperandBase* GraphBuilderBase::Tanh(OperandBase* input) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Unary(this, op::UnaryOpType::kTanh, input));
    }

    OperandBase* GraphBuilderBase::Transpose(OperandBase* input, TransposeOptions const* options) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Transpose(this, input, options));
    }

    OperandBase* GraphBuilderBase::LeakyRelu(OperandBase* input, LeakyReluOptions const* options) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::LeakyRelu(this, input, options));
    }

    OperandBase* GraphBuilderBase::Concat(uint32_t inputsCount,
                                          OperandBase* const* inputs,
                                          uint32_t axis) {
        std::vector<Ref<OperandBase>> operandInputs;
        operandInputs.reserve(inputsCount);
        for (uint32_t i = 0; i < inputsCount; ++i) {
            operandInputs.push_back(inputs[i]);
        }
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Concat(this, std::move(operandInputs), axis));
    }

    OperandBase* GraphBuilderBase::Gemm(OperandBase* a,
                                        OperandBase* b,
                                        GemmOptions const* options) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Gemm(this, a, b, options));
    }

    OperandBase* GraphBuilderBase::Clamp(OperandBase* input, ClampOptions const* options) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Clamp(this, input, options));
    }

    OperandBase* GraphBuilderBase::BatchNorm(OperandBase* input,
                                             OperandBase* mean,
                                             OperandBase* variance,
                                             BatchNormOptions const* options) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::BatchNorm(this, input, mean, variance, options));
    }

    OperandBase* GraphBuilderBase::Pad(OperandBase* input,
                                       OperandBase* padding,
                                       PadOptions const* options) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Pad(this, input, padding, options));
    }

    OperandBase* GraphBuilderBase::InstanceNorm(OperandBase* input,
                                                InstanceNormOptions const* options) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::InstanceNorm(this, input, options));
    }

    GraphBase* GraphBuilderBase::Build(NamedOperandsBase const* namedOperands) {
        if (DAWN_UNLIKELY(this->IsError())) {
            dawn::ErrorLog() << "This Graph object is an error";
            return nullptr;
        }

        std::vector<const OperandBase*> outputs;
        if (namedOperands->GetRecords().empty()) {
            dawn::ErrorLog() << "The output named operands are empty.";
            return nullptr;
        }
        for (auto& namedOutput : namedOperands->GetRecords()) {
            outputs.push_back(namedOutput.second);
        }
        std::vector<const OperandBase*> sorted_operands = TopologicalSort(outputs);
        Ref<GraphBase> graph = AcquireRef(GetContext()->CreateGraph());
        for (auto& op : sorted_operands) {
            if (op->IsError() || GetContext()->ConsumedError(op->AddToGraph(graph.Get()))) {
                dawn::ErrorLog() << "Failed to add the operand when building graph.";
                return nullptr;
            }
        }
        for (auto& namedOutput : namedOperands->GetRecords()) {
            if (GetContext()->ConsumedError(
                    graph->AddOutput(namedOutput.first, namedOutput.second))) {
                dawn::ErrorLog() << "Failed to add output when building graph.";
                return nullptr;
            }
        }
        if (GetContext()->ConsumedError(graph->Finish())) {
            dawn::ErrorLog() << "Failed to finish building graph.";
            return nullptr;
        }

        if (GetContext()->ConsumedError(graph->Compile())) {
            dawn::ErrorLog() << "Failed to compile the graph.";
            return nullptr;
        }

        return graph.Detach();
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
    std::vector<const OperandBase*> GraphBuilderBase::TopologicalSort(
        std::vector<const OperandBase*>& rootNodes) {
        std::stack<const OperandBase*> nodesToDo;
        std::unordered_set<const OperandBase*> nodesDone;
        std::vector<const OperandBase*> result;

        for (auto& node : rootNodes) {
            nodesToDo.push(node);
        }
        while (nodesToDo.size() > 0) {
            const OperandBase* node = nodesToDo.top();
            if (nodesDone.count(node) == 0) {
                bool can_add = true;
                for (auto& dep : node->Inputs()) {
                    if (nodesDone.count(dep.Get()) == 0) {
                        can_add = false;
                        nodesToDo.push(dep.Get());
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

}  // namespace webnn_native
