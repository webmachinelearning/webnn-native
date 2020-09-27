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

#include "webnn_native/ModelBuilder.h"

#include <stack>
#include <string>
#include <unordered_set>
#include <vector>

#include "common/Assert.h"
#include "common/RefCounted.h"
#include "webnn_native/Model.h"
#include "webnn_native/NeuralNetworkContext.h"
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

namespace webnn_native {

    ModelBuilderBase::ModelBuilderBase(NeuralNetworkContextBase* context) : ObjectBase(context) {
    }

    OperandBase* ModelBuilderBase::Constant(OperandDescriptor const* desc,
                                            void const* value,
                                            size_t size) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Constant(this, desc, value, size));
    }

    OperandBase* ModelBuilderBase::Input(char const* name, OperandDescriptor const* desc) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Input(this, std::string(name), desc));
    }

    OperandBase* ModelBuilderBase::Matmul(OperandBase* a, OperandBase* b) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Binary(this, op::BinaryOpType::kMatMul, a, b));
    }

    OperandBase* ModelBuilderBase::Add(OperandBase* a, OperandBase* b) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Binary(this, op::BinaryOpType::kAdd, a, b));
    }

    OperandBase* ModelBuilderBase::Mul(OperandBase* a, OperandBase* b) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Binary(this, op::BinaryOpType::kMul, a, b));
    }

    OperandBase* ModelBuilderBase::Conv2d(OperandBase* input,
                                          OperandBase* filter,
                                          Conv2dOptions const* options) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Conv2d(this, input, filter, options));
    }

    OperandBase* ModelBuilderBase::AveragePool2d(OperandBase* input, Pool2dOptions const* options) {
        DAWN_VALIDATE_AND_INFER_TYPES(
            new op::Pool2d(this, op::Pool2dType::kAveragePool2d, input, options));
    }

    OperandBase* ModelBuilderBase::MaxPool2d(OperandBase* input, Pool2dOptions const* options) {
        DAWN_VALIDATE_AND_INFER_TYPES(
            new op::Pool2d(this, op::Pool2dType::kMaxPool2d, input, options));
    }

    OperandBase* ModelBuilderBase::Relu(OperandBase* input) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Unary(this, op::UnaryOpType::kRelu, input));
    }

    OperandBase* ModelBuilderBase::Reshape(OperandBase* input,
                                           int32_t const* new_shape,
                                           size_t new_shape_count) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Reshape(this, input, new_shape, new_shape_count));
    }

    OperandBase* ModelBuilderBase::Softmax(OperandBase* input) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Unary(this, op::UnaryOpType::kSoftmax, input));
    }

    OperandBase* ModelBuilderBase::Transpose(OperandBase* input, TransposeOptions const* options) {
        DAWN_VALIDATE_AND_INFER_TYPES(new op::Transpose(this, input, options));
    }

    ModelBase* ModelBuilderBase::CreateModel(NamedOperandsBase const* namedOperands) {
        Ref<ModelBase> model = AcquireRef(CreateModelImpl());
        std::vector<const OperandBase*> outputs;
        if (namedOperands->GetRecords().empty()) {
            return ModelBase::MakeError(this);
        }
        for (auto& namedOutput : namedOperands->GetRecords()) {
            outputs.push_back(namedOutput.second);
        }
        std::vector<const OperandBase*> sorted_operands = TopologicalSort(outputs);
        for (auto& op : sorted_operands) {
            if (op->IsError() || GetContext()->ConsumedError(op->AddToModel(model.Get()))) {
                return ModelBase::MakeError(this);
            }
        }
        for (auto& namedOutput : namedOperands->GetRecords()) {
            if (GetContext()->ConsumedError(
                    model->AddOutput(namedOutput.first, namedOutput.second))) {
                return ModelBase::MakeError(this);
            }
        }
        if (GetContext()->ConsumedError(model->Finish())) {
            return ModelBase::MakeError(this);
        }
        return model.Detach();
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
    std::vector<const OperandBase*> ModelBuilderBase::TopologicalSort(
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
