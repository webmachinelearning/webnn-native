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
#include "webnn_native/ops/Binary.h"
#include "webnn_native/ops/Constant.h"
#include "webnn_native/ops/Conv2d.h"
#include "webnn_native/ops/Input.h"
#include "webnn_native/ops/Pool2d.h"
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

#define BUILD_ERROR_AND_CALLBACK(message)                          \
    do {                                                           \
        callback(MLBuildStatus_Error, nullptr, message, userdata); \
        return;                                                    \
    } while (0)

namespace webnn_native {

    GraphBuilderBase::GraphBuilderBase(ContextBase* context) : ObjectBase(context) {
    }

    OperandBase* GraphBuilderBase::Constant(OperandDescriptor const* desc,
                                            void const* value,
                                            size_t size) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Constant(this, desc, value, size));
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

    OperandBase* GraphBuilderBase::Mul(OperandBase* a, OperandBase* b) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Binary(this, op::BinaryOpType::kMul, a, b));
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

    OperandBase* GraphBuilderBase::Relu(OperandBase* input) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Unary(this, op::UnaryOpType::kRelu, input));
    }

    OperandBase* GraphBuilderBase::Reshape(OperandBase* input,
                                           int32_t const* new_shape,
                                           size_t new_shape_count) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Reshape(this, input, new_shape, new_shape_count));
    }

    OperandBase* GraphBuilderBase::Softmax(OperandBase* input) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Unary(this, op::UnaryOpType::kSoftmax, input));
    }

    OperandBase* GraphBuilderBase::Transpose(OperandBase* input, TransposeOptions const* options) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Transpose(this, input, options));
    }

    void GraphBuilderBase::Build(NamedOperandsBase const* namedOperands,
                                 MLBuildCallback callback,
                                 void* userdata) {
        if (DAWN_UNLIKELY(this->IsError())) {
            BUILD_ERROR_AND_CALLBACK("This Graph object is an error");
        }

        std::vector<const OperandBase*> outputs;
        if (namedOperands->GetRecords().empty()) {
            BUILD_ERROR_AND_CALLBACK("The output named operands are empty.");
        }
        for (auto& namedOutput : namedOperands->GetRecords()) {
            outputs.push_back(namedOutput.second);
        }
        std::vector<const OperandBase*> sorted_operands = TopologicalSort(outputs);
        Ref<GraphBase> graph = AcquireRef(GetContext()->CreateGraph());
        for (auto& op : sorted_operands) {
            if (op->IsError() || GetContext()->ConsumedError(op->AddToGraph(graph.Get()))) {
                BUILD_ERROR_AND_CALLBACK("Failed to add the operand when building graph.");
            }
        }
        for (auto& namedOutput : namedOperands->GetRecords()) {
            if (GetContext()->ConsumedError(
                    graph->AddOutput(namedOutput.first, namedOutput.second))) {
                BUILD_ERROR_AND_CALLBACK("Failed to add output when building graph.");
            }
        }
        if (GetContext()->ConsumedError(graph->Finish())) {
            BUILD_ERROR_AND_CALLBACK("Failed to finish building graph.");
        }
        callback(MLBuildStatus_Success, reinterpret_cast<MLGraph>(graph.Detach()), nullptr,
                 userdata);
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
