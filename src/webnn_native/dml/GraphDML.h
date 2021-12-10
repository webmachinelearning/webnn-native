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

#ifndef WEBNN_NATIVE_DML_MODEL_DML_H_
#define WEBNN_NATIVE_DML_MODEL_DML_H_

#define DML_TARGET_VERSION_USE_LATEST 1

#include <dxgi1_4.h>
#include <wrl\client.h>

#include "DirectML.h"
#include "webnn_native/Graph.h"
#include "webnn_native/Operand.h"
#include "webnn_native/Operator.h"
#include "webnn_native/dml/ContextDML.h"
#include "webnn_native/ops/BatchNorm.h"
#include "webnn_native/ops/Binary.h"
#include "webnn_native/ops/Clamp.h"
#include "webnn_native/ops/Concat.h"
#include "webnn_native/ops/Constant.h"
#include "webnn_native/ops/Conv2d.h"
#include "webnn_native/ops/Gemm.h"
#include "webnn_native/ops/Gru.h"
#include "webnn_native/ops/Input.h"
#include "webnn_native/ops/InstanceNorm.h"
#include "webnn_native/ops/LeakyRelu.h"
#include "webnn_native/ops/Pad.h"
#include "webnn_native/ops/Pool2d.h"
#include "webnn_native/ops/Reduce.h"
#include "webnn_native/ops/Resample2d.h"
#include "webnn_native/ops/Reshape.h"
#include "webnn_native/ops/Slice.h"
#include "webnn_native/ops/Split.h"
#include "webnn_native/ops/Squeeze.h"
#include "webnn_native/ops/Transpose.h"
#include "webnn_native/ops/Unary.h"
namespace webnn_native { namespace dml {

    using namespace Microsoft::WRL;

    enum class NodeType {
        Invalid,
        Input,
        IntermediateNode,
    };

    //  Represent the DirectML tensor description.
    struct TensorDesc {
        std::vector<UINT> dimensions;
        DML_BUFFER_TENSOR_DESC bufferDesc = {};
    };

    //  Only represent the information for input nodes.
    struct InputInfo {
        // Indicate the input index in the graph.
        size_t inputIndex = 0;
        void const* buffer = nullptr;
        size_t byteLength = 0;
        // Indicate if the input is a constant buffer which need to be
        // uploaded in the stage of initialization.
        bool isConstant = false;
    };

    //  Represent the information of each node added into the graph.
    struct NodeInfo {
        NodeType nodeType = NodeType::Invalid;
        // Indicate the index of the output node produced by this node.
        uint32_t outputNodeIndex = 0;
        DML_TENSOR_DESC outputDESC = {};
        // Indicate the index of the intermediate node in the graph.
        uint32_t nodeIndex = 0;
        std::string name = "";
        InputInfo inputInfo = {};
    };

    class Graph : public GraphBase {
      public:
        explicit Graph(Context* context);
        ~Graph() override = default;

        virtual MaybeError AddConstant(const op::Constant* constant) override;
        virtual MaybeError AddInput(const op::Input* input) override;
        virtual MaybeError AddOutput(const std::string& name, const OperandBase* output) override;
        virtual MaybeError AddBatchNorm(const op::BatchNorm* batchNorm) override;
        virtual MaybeError AddBinary(const op::Binary* binary) override;
        virtual MaybeError AddConv2d(const op::Conv2d* conv2d) override;
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

        bool GetDMLTensorDesc(OperandDescriptor const* desc,
                              TensorDesc& dmlBufferTensorDesc,
                              DML_TENSOR_FLAGS tensorFlag = DML_TENSOR_FLAGS::DML_TENSOR_FLAG_NONE);
        void InitializeDirect3D12();
        DML_GRAPH_DESC CreateGraphDesc();
        void CloseExecuteResetWait();
        MaybeError AddToGraph(std::vector<NodeInfo> inputNodes);

      private:
        MaybeError CompileImpl() override;
        MLComputeGraphStatus ComputeImpl(NamedInputsBase* inputs,
                                         NamedOutputsBase* outputs) override;

        // Represents a DirectML device, which is used to create operators, binding tables, command
        // recorders, and other objects.
        Microsoft::WRL::ComPtr<IDMLDevice> mDevice;
        // The IDMLDevice1 interface inherits from IDMLDevice.
        Microsoft::WRL::ComPtr<IDMLDevice1> mDevice1;
        // Represents a virtual adapter; it is used to create command allocators, command lists,
        // command queues, fences, resources, pipeline state objects, heaps, root signatures,
        // samplers, and many resource views.
        Microsoft::WRL::ComPtr<ID3D12Device> mD3D12Device;

        Microsoft::WRL::ComPtr<IDMLCommandRecorder> mCommandRecorder;
        Microsoft::WRL::ComPtr<ID3D12CommandQueue> mCommandQueue;
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> mCommandAllocator;
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> mCommandList;

        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> mDescriptorHeap;
        DML_BINDING_TABLE_DESC mBindingTableDesc{};
        Microsoft::WRL::ComPtr<IDMLBindingTable> mBindingTable;

        Microsoft::WRL::ComPtr<ID3D12Resource> mOutputBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> mUploadBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> mInputBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> mTemporaryBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> mPersistentBuffer;

        // Describe a graph of DirectML operators used to compile a combined, optimized operator.
        std::vector<NodeInfo> mInputNodes;
        std::vector<NodeInfo> mOutputNodes;
        std::vector<std::unique_ptr<DML_OPERATOR_GRAPH_NODE_DESC>> mIntermediateNodesDesc;
        std::vector<std::unique_ptr<DML_INPUT_GRAPH_EDGE_DESC>> mInputEdgesDesc;
        std::vector<std::unique_ptr<DML_OUTPUT_GRAPH_EDGE_DESC>> mOutputEdgesDesc;
        std::vector<std::unique_ptr<DML_INTERMEDIATE_GRAPH_EDGE_DESC>> mIntermediateEdgesDesc;
        std::vector<DML_GRAPH_NODE_DESC> mIntermediateNodes;
        std::vector<DML_GRAPH_EDGE_DESC> mInputEdges;
        std::vector<DML_GRAPH_EDGE_DESC> mOutputEdges;
        std::vector<DML_GRAPH_EDGE_DESC> mIntermediateEdges;

        // IDMLCompiledOperator represents the DirectML graph's output which need to be initialized
        // by IDMLOperatorInitializer.
        Microsoft::WRL::ComPtr<IDMLCompiledOperator> mCompiledOperator;
        Microsoft::WRL::ComPtr<IDMLOperatorInitializer> mCompiledOperatorInitializer;

        std::map<const OperandBase*, NodeInfo> mGraphNodesMap;
        // Indicate the index of the intermediate node in the graph.
        uint32_t mNodeIndex = 0;
        // Keep intermediate nodes here to avoid releasing too early.
        std::map<uint32_t, Microsoft::WRL::ComPtr<IDMLOperator>> mIntermediateNodesMap;

        // Keep the input tensors description here to avoid releasing too early.
        std::map<uint32_t, TensorDesc> mTensorDescMap;
    };

}}  // namespace webnn_native::dml

#endif  // WEBNN_NATIVE_DML_MODEL_DML_H_
