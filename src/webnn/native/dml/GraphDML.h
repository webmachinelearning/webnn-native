// Copyright 2022 The WebNN-native Authors
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

#ifndef WEBNN_NATIVE_DML_GRAPH_DML_H_
#define WEBNN_NATIVE_DML_GRAPH_DML_H_

#include <unordered_set>

#include "DeviceDML.h"
#include "webnn/native/Graph.h"
#include "webnn/native/Operand.h"
#include "webnn/native/Operator.h"
#include "webnn/native/dml/ContextDML.h"
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

namespace webnn::native::dml {

    class Graph : public GraphBase {
      public:
        explicit Graph(Context* context);
        ~Graph() override = default;

        virtual MaybeError AddConstant(const op::Constant* constant) override;
        virtual MaybeError AddInput(const op::Input* input) override;
        virtual MaybeError AddOutput(std::string_view name, const OperandBase* output) override;
        virtual MaybeError AddBatchNorm(const op::BatchNorm* batchNorm) override;
        virtual MaybeError AddBinary(const op::Binary* binary) override;
        virtual MaybeError AddConv2d(const op::Conv2d* conv2d) override;
        virtual MaybeError AddConvTranspose2d(const op::ConvTranspose2d* convTranspose2d) override;
        virtual MaybeError AddPad(const op::Pad* pad) override;
        virtual MaybeError AddPool2d(const op::Pool2d* pool2d) override;
        virtual MaybeError AddReduce(const op::Reduce* reduce) override;
        virtual MaybeError AddResample2d(const op::Resample2d* resample2d) override;
        virtual MaybeError AddReshape(const op::Reshape* reshape) override;
        virtual MaybeError AddSlice(const op::Slice* slice) override;
        virtual MaybeError AddSplit(const op::Split* split) override;
        virtual MaybeError AddSqueeze(const op::Squeeze* squeeze) override;
        virtual MaybeError AddTranspose(const op::Transpose* transpose) override;
        virtual MaybeError AddUnary(const op::Unary* unary) override;
        virtual MaybeError AddGemm(const op::Gemm* Gemm) override;
        virtual MaybeError AddGru(const op::Gru* Gru) override;
        virtual MaybeError AddConcat(const op::Concat* concat) override;
        virtual MaybeError AddClamp(const op::Clamp* clamp) override;
        virtual MaybeError AddInstanceNorm(const op::InstanceNorm* instanceNorm) override;
        virtual MaybeError Finish() override;

        MaybeError CreateDmlTensorDesc(DML_TENSOR_DESC& createdTensorDesc,
                                       const std::vector<UINT>& dimensions,
                                       const std::vector<UINT>& strides = {},
                                       DML_TENSOR_DATA_TYPE dataType = DML_TENSOR_DATA_TYPE_FLOAT32,
                                       DML_TENSOR_FLAGS tensorFlag = DML_TENSOR_FLAG_NONE);
        MaybeError CreateDmlTensorDesc(
            DML_TENSOR_DESC& createdTensorDesc,
            OperandDescriptor const* desc,
            DML_TENSOR_FLAGS tensorFlag = DML_TENSOR_FLAGS::DML_TENSOR_FLAG_NONE);
        MaybeError CreateDmlTensorDesc(DML_TENSOR_DESC& createdTensorDesc,
                                       DML_TENSOR_DESC const* tensorDesc,
                                       std::vector<UINT> dimensions = {},
                                       std::vector<UINT> strides = {},
                                       bool useDefaultFlags = false);
        MaybeError AppendIdentity(DML_TENSOR_DESC& outputTensorDesc,
                                  const DML_TENSOR_DESC& inputTensorDesc);
        MaybeError CreateConstantInput(std::shared_ptr<InputNode>& inputNode,
                                       void const* value,
                                       size_t size,
                                       const std::vector<UINT>& dimensions,
                                       const std::vector<UINT>& strides = {},
                                       DML_TENSOR_DATA_TYPE dataType = DML_TENSOR_DATA_TYPE_FLOAT32,
                                       DML_TENSOR_FLAGS tensorFlag = DML_TENSOR_FLAG_OWNED_BY_DML);
        std::shared_ptr<NodeBase> Clamp(const op::ClampBase* clamp,
                                        std::shared_ptr<NodeBase>& inputNode);
        MaybeError HardSwish(std::shared_ptr<NodeBase>& inputNode,
                             const std::vector<UINT>& inputDims);
        MaybeError EmulateFusedOperator(FusionOperatorBase* activation,
                                        std::shared_ptr<NodeBase>& inputNode,
                                        const std::vector<UINT>& inputDims);
        MaybeError TransposeOutputToNhwc(std::shared_ptr<NodeBase>& inputNode,
                                         const std::vector<UINT>& nchwOutputDims);

      private:
        MaybeError CompileImpl() override;
        MaybeError ComputeImpl(NamedInputsBase* inputs, NamedOutputsBase* outputs) override;
        std::unique_ptr<Device> mDevice;
        std::vector<std::shared_ptr<InputNode>> mInputs;
        std::vector<Node> mOutputs;
        std::unique_ptr<GraphBuilder> mGraphBuilder;
        ComPtr<IDMLCompiledOperator> mCompiledGraph;
        std::map<const OperandBase*, std::shared_ptr<NodeBase>> mGraphNodesMap;
        // Keep the input tensors description here to avoid releasing too early.
        std::unordered_set<const OperandBase*> mConstantSet;
        std::vector<std::unique_ptr<char>> mConstantsBuffer;
        std::vector<std::shared_ptr<TensorDesc>> mTensorsDesc;
    };

}  // namespace webnn::native::dml

#endif  // WEBNN_NATIVE_DML_GRAPH_DML_H_
