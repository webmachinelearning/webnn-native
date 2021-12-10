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

#include "webnn_native/dml/GraphDML.h"

#include <algorithm>

#include "DMLUtils.h"
#include "common/Assert.h"
#include "common/Log.h"
#include "webnn_native/ErrorData.h"
#include "webnn_native/NamedInputs.h"
#include "webnn_native/NamedOutputs.h"
#include "webnn_native/Utils.h"
#include "webnn_native/dml/ContextDML.h"

#define WEBNN_CHECK(hr)                             \
    if (((HRESULT)(hr)) < 0) {                      \
        dawn::ErrorLog() << "Failed to do " << #hr; \
        DAWN_ASSERT(0);                             \
    }

namespace webnn_native { namespace dml {

    void Graph::InitializeDirect3D12() {
#if defined(_DEBUG)
        Microsoft::WRL::ComPtr<ID3D12Debug> d3D12Debug;
        if (FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(&d3D12Debug)))) {
            // The D3D12 debug layer is missing - you must install the Graphics Tools optional
            // feature
            DAWN_ASSERT(0);
        }
        d3D12Debug->EnableDebugLayer();
#endif
        Microsoft::WRL::ComPtr<IDXGIFactory4> dxgiFactory;
        WEBNN_CHECK(CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory)));
        Microsoft::WRL::ComPtr<IDXGIAdapter> dxgiAdapter;
        UINT adapterIndex{};
        HRESULT hr{};
        do {
            dxgiAdapter = nullptr;
            IDXGIAdapter* dxgiAdapterPtr = dxgiAdapter.Get();
            WEBNN_CHECK(dxgiFactory->EnumAdapters(adapterIndex, &dxgiAdapterPtr));
            ++adapterIndex;

            hr = ::D3D12CreateDevice(dxgiAdapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                     IID_PPV_ARGS(&mD3D12Device));
            if (hr < 0) {
                dawn::ErrorLog() << "Failed to do ::D3D12CreateDevice.";
            }

            if (hr == DXGI_ERROR_UNSUPPORTED)
                continue;
        } while (hr != S_OK);

        D3D12_COMMAND_QUEUE_DESC commandQueueDesc{};
        commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        commandQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        WEBNN_CHECK(
            mD3D12Device->CreateCommandQueue(&commandQueueDesc, IID_PPV_ARGS(&mCommandQueue)));
        WEBNN_CHECK(mD3D12Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                         IID_PPV_ARGS(&mCommandAllocator)));
        WEBNN_CHECK(mD3D12Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                    mCommandAllocator.Get(), nullptr,
                                                    IID_PPV_ARGS(&mCommandList)));
    }

    void Graph::CloseExecuteResetWait() {
        mCommandList->Close();
        ID3D12CommandList* commandLists[] = {mCommandList.Get()};
        mCommandQueue->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);
        mCommandQueue.Get()->GetDevice(IID_PPV_ARGS(mD3D12Device.GetAddressOf()));
        Microsoft::WRL::ComPtr<ID3D12Fence> fence;
        mD3D12Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.GetAddressOf()));
        mCommandQueue.Get()->Signal(fence.Get(), 1);
        fence->SetEventOnCompletion(1, nullptr);
        mCommandAllocator->Reset();
        mCommandList->Reset(mCommandAllocator.Get(), nullptr);
    }

    MaybeError Graph::AddToGraph(std::vector<NodeInfo> inputNodes) {
        std::unique_ptr<DML_OPERATOR_GRAPH_NODE_DESC> nodeDesc(new DML_OPERATOR_GRAPH_NODE_DESC);
        nodeDesc->Operator = mIntermediateNodesMap[mNodeIndex].Get();
        mIntermediateNodes.push_back({DML_GRAPH_NODE_TYPE_OPERATOR, nodeDesc.get()});
        mIntermediateNodesDesc.push_back(std::move(nodeDesc));

        for (size_t i = 0; i < inputNodes.size(); ++i) {
            NodeInfo inputNode = inputNodes[i];
            if (inputNode.nodeType == NodeType::Input) {
                std::unique_ptr<DML_INPUT_GRAPH_EDGE_DESC> inputEdgeDesc(
                    new DML_INPUT_GRAPH_EDGE_DESC);
                inputEdgeDesc->GraphInputIndex = inputNode.inputInfo.inputIndex;
                inputEdgeDesc->ToNodeIndex = mNodeIndex;
                inputEdgeDesc->ToNodeInputIndex = i;
                mInputEdges.push_back({DML_GRAPH_EDGE_TYPE_INPUT, inputEdgeDesc.get()});
                mInputEdgesDesc.push_back(std::move(inputEdgeDesc));
            } else if (inputNode.nodeType == NodeType::IntermediateNode) {
                std::unique_ptr<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdgeDesc(
                    new DML_INTERMEDIATE_GRAPH_EDGE_DESC);
                intermediateEdgeDesc->FromNodeIndex = inputNode.nodeIndex;
                intermediateEdgeDesc->FromNodeOutputIndex = inputNode.outputNodeIndex;
                intermediateEdgeDesc->ToNodeIndex = mNodeIndex;
                intermediateEdgeDesc->ToNodeInputIndex = i;
                mIntermediateEdges.push_back(
                    {DML_GRAPH_EDGE_TYPE_INTERMEDIATE, intermediateEdgeDesc.get()});
                mIntermediateEdgesDesc.push_back(std::move(intermediateEdgeDesc));
            } else {
                return DAWN_INTERNAL_ERROR("Invalid node type.");
            }
        }
        mNodeIndex++;
        return {};
    }

    DML_GRAPH_DESC Graph::CreateGraphDesc() {
        DML_GRAPH_DESC graphDesc = {};
        graphDesc.InputCount = static_cast<UINT>(mInputNodes.size());
        graphDesc.OutputCount = static_cast<UINT>(mOutputNodes.size());

        graphDesc.NodeCount = static_cast<UINT>(mIntermediateNodes.size());
        graphDesc.Nodes = mIntermediateNodes.data();

        graphDesc.InputEdgeCount = static_cast<UINT>(mInputEdges.size());
        graphDesc.InputEdges = mInputEdges.data();

        graphDesc.OutputEdgeCount = static_cast<UINT>(mOutputEdges.size());
        graphDesc.OutputEdges = mOutputEdges.data();

        graphDesc.IntermediateEdgeCount = static_cast<UINT>(mIntermediateEdges.size());
        graphDesc.IntermediateEdges = mIntermediateEdges.data();

        return graphDesc;
    }

    bool Graph::GetDMLTensorDesc(OperandDescriptor const* desc,
                                 TensorDesc& tensorDesc,
                                 DML_TENSOR_FLAGS tensorFlag) {
        size_t typeLength = 4;
        if (desc->type == ml::OperandType::Float32) {
            tensorDesc.bufferDesc.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
            typeLength = 4;
        } else if (desc->type == ml::OperandType::Float16) {
            tensorDesc.bufferDesc.DataType = DML_TENSOR_DATA_TYPE_FLOAT16;
            typeLength = 2;
        } else if (desc->type == ml::OperandType::Int32) {
            tensorDesc.bufferDesc.DataType = DML_TENSOR_DATA_TYPE_INT32;
        } else if (desc->type == ml::OperandType::Uint32) {
            tensorDesc.bufferDesc.DataType = DML_TENSOR_DATA_TYPE_UINT32;
        } else {
            return false;
        }

        size_t bufferLength = typeLength;
        if (desc->dimensionsCount > DML_TENSOR_DIMENSION_COUNT_MAX) {
            dawn::ErrorLog() << "Tensor dimension count " << desc->dimensionsCount
                             << " is greater than DML_TENSOR_DIMENSION_COUNT_MAX "
                             << DML_TENSOR_DIMENSION_COUNT_MAX;
            return false;
        }

        if (desc->dimensionsCount == 0) {
            tensorDesc.dimensions.resize(1);
            tensorDesc.dimensions[0] = 1;
        } else {
            tensorDesc.dimensions.resize(desc->dimensionsCount);
            for (uint32_t i = 0; i < desc->dimensionsCount; ++i) {
                int32_t d = desc->dimensions[i];
                if (d < 0) {
                    dawn::ErrorLog() << "DML doesn't support the negative dimension value";
                    return false;
                }
                tensorDesc.dimensions[i] = d;
                bufferLength *= d;
            }
        }
        tensorDesc.bufferDesc.Flags = tensorFlag;
        tensorDesc.bufferDesc.DimensionCount = desc->dimensionsCount;
        tensorDesc.bufferDesc.Sizes = tensorDesc.dimensions.data();
        tensorDesc.bufferDesc.Strides = nullptr;
        tensorDesc.bufferDesc.TotalTensorSizeInBytes = bufferLength;
        tensorDesc.bufferDesc.GuaranteedBaseOffsetAlignment = 0;
        return true;
    }

    Graph::Graph(Context* context) : GraphBase(context) {
        // Set up Direct3D 12.
        InitializeDirect3D12();

        // Create the DirectML device.
        DML_CREATE_DEVICE_FLAGS dmlCreateDeviceFlags = DML_CREATE_DEVICE_FLAG_NONE;
        WEBNN_CHECK(
            DMLCreateDevice(mD3D12Device.Get(), dmlCreateDeviceFlags, IID_PPV_ARGS(&mDevice)));
    }

    MaybeError Graph::AddConstant(const op::Constant* constant) {
        const OperandDescriptor* desc = constant->GetOperandDescriptor();
        mTensorDescMap.insert(std::make_pair(mInputNodes.size(), TensorDesc{}));
        if (!GetDMLTensorDesc(desc, mTensorDescMap[mInputNodes.size()],
                              DML_TENSOR_FLAG_OWNED_BY_DML)) {
            return DAWN_INTERNAL_ERROR("Failed to get DML buffer tensor description.");
        }
        DML_TENSOR_DESC dmlTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                         &mTensorDescMap[mInputNodes.size()].bufferDesc};
        InputInfo constantInfo = {mInputNodes.size(), constant->GetBuffer(),
                                  constant->GetByteLength(), true};
        NodeInfo node = {NodeType::Input,
                         0,
                         dmlTensorDesc,
                         0,
                         "Input_Constant_" + std::to_string(mInputNodes.size()),
                         constantInfo};
        mGraphNodesMap[constant->PrimaryOutput()] = node;
        mInputNodes.push_back(node);
        return {};
    }

    MaybeError Graph::AddInput(const op::Input* input) {
        const OperandDescriptor* desc = input->GetOperandDescriptor();
        mTensorDescMap.insert(std::make_pair(mInputNodes.size(), TensorDesc{}));
        if (!GetDMLTensorDesc(desc, mTensorDescMap[mInputNodes.size()])) {
            return DAWN_INTERNAL_ERROR("Failed to get DML buffer tensor description.");
        }
        DML_TENSOR_DESC dmlTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                         &mTensorDescMap[mInputNodes.size()].bufferDesc};
        InputInfo constantInfo = {mInputNodes.size(), nullptr, 0, false};
        NodeInfo node = {NodeType::Input, 0, dmlTensorDesc, 0, input->GetName(), constantInfo};
        mGraphNodesMap[input->PrimaryOutput()] = node;
        mInputNodes.push_back(node);
        return {};
    }

    MaybeError Graph::AddBinary(const op::Binary* binary) {
        switch (binary->GetType()) {
            case op::BinaryOpType::kAdd: {
                DAWN_ASSERT(mGraphNodesMap.find(binary->Inputs()[0].Get()) != mGraphNodesMap.end());
                DAWN_ASSERT(mGraphNodesMap.find(binary->Inputs()[1].Get()) != mGraphNodesMap.end());
                NodeInfo inputNodeA = mGraphNodesMap[binary->Inputs()[0].Get()];
                NodeInfo inputNodeB = mGraphNodesMap[binary->Inputs()[1].Get()];
                DML_ELEMENT_WISE_ADD_OPERATOR_DESC dmlAddOperatorDesc{};
                dmlAddOperatorDesc.ATensor = &inputNodeA.outputDESC;
                dmlAddOperatorDesc.BTensor = &inputNodeB.outputDESC;

                DML_TENSOR_DESC outputTensorDesc = inputNodeA.outputDESC;
                dmlAddOperatorDesc.OutputTensor = &outputTensorDesc;

                DML_OPERATOR_DESC dmlOperatorDesc = {};
                dmlOperatorDesc.Type = DML_OPERATOR_ELEMENT_WISE_ADD;
                dmlOperatorDesc.Desc = &dmlAddOperatorDesc;

                Microsoft::WRL::ComPtr<IDMLOperator> dmlOperator;
                WEBNN_CHECK(mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));
                mIntermediateNodesMap[mNodeIndex] = dmlOperator;

                NodeInfo node = {NodeType::IntermediateNode, 0, outputTensorDesc, mNodeIndex};
                mGraphNodesMap[binary->PrimaryOutput()] = node;

                return AddToGraph({inputNodeA, inputNodeB});
            }
            default:
                return DAWN_UNIMPLEMENTED_ERROR(" Binary op is not implemented.");
        }
    }

    MaybeError Graph::AddUnary(const op::Unary* unary) {
        switch (unary->GetType()) {
            case op::UnaryOpType::kSigmoid: {
                Microsoft::WRL::ComPtr<IDMLOperator> dmlOperator;
                DAWN_ASSERT(mGraphNodesMap.find(unary->Inputs()[0].Get()) != mGraphNodesMap.end());
                NodeInfo inputNode = mGraphNodesMap[unary->Inputs()[0].Get()];
                DML_TENSOR_DESC inputTensorDesc = inputNode.outputDESC;
                DML_ACTIVATION_SIGMOID_OPERATOR_DESC dmlSigmoidOperatorDesc{};
                dmlSigmoidOperatorDesc.InputTensor = &inputTensorDesc;
                dmlSigmoidOperatorDesc.OutputTensor = &inputTensorDesc;

                DML_OPERATOR_DESC dmlOperatorDesc = {};
                dmlOperatorDesc.Type = DML_OPERATOR_ACTIVATION_SIGMOID;
                dmlOperatorDesc.Desc = &dmlSigmoidOperatorDesc;

                WEBNN_CHECK(mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));
                mIntermediateNodesMap[mNodeIndex] = dmlOperator;

                NodeInfo node = {NodeType::IntermediateNode, 0, inputTensorDesc, mNodeIndex};
                mGraphNodesMap[unary->PrimaryOutput()] = node;

                return AddToGraph({inputNode});
                break;
            }
            default:
                return DAWN_UNIMPLEMENTED_ERROR(" Unary op is not implemented.");
        }
        return {};
    }

    MaybeError Graph::AddOutput(const std::string& name, const OperandBase* output) {
        auto node = mGraphNodesMap[output];
        if (node.nodeType == NodeType::Input) {
            return DAWN_INTERNAL_ERROR("Graph for input = output is invalid.");
        }
        node.name = name;
        std::unique_ptr<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdgeDesc(new DML_OUTPUT_GRAPH_EDGE_DESC);
        outputEdgeDesc->FromNodeIndex = node.nodeIndex;
        outputEdgeDesc->FromNodeOutputIndex = node.outputNodeIndex;
        outputEdgeDesc->GraphOutputIndex = mOutputNodes.size();
        mOutputEdges.push_back({DML_GRAPH_EDGE_TYPE_OUTPUT, outputEdgeDesc.get()});
        mOutputEdgesDesc.push_back(std::move(outputEdgeDesc));
        mOutputNodes.push_back(node);
        return {};
    }

    MaybeError Graph::AddBatchNorm(const op::BatchNorm* batchNorm) {
        return DAWN_UNIMPLEMENTED_ERROR("BatchNorm hasn't been supported on DirectML.");
    }

    MaybeError Graph::AddConv2d(const op::Conv2d* conv2d) {
        return DAWN_UNIMPLEMENTED_ERROR("Conv2d hasn't been supported on DirectML.");
    }

    MaybeError Graph::AddGru(const op::Gru* gru) {
        return DAWN_UNIMPLEMENTED_ERROR("Gru hasn't been supported on DirectML.");
    }

    MaybeError Graph::AddPad(const op::Pad* pad) {
        return DAWN_UNIMPLEMENTED_ERROR("Pad hasn't been supported on DirectML.");
    }

    MaybeError Graph::AddPool2d(const op::Pool2d* pool2d) {
        return DAWN_UNIMPLEMENTED_ERROR("Pool2d hasn't been supported on DirectML.");
    }

    MaybeError Graph::AddClamp(const op::Clamp* clamp) {
        return DAWN_UNIMPLEMENTED_ERROR("Clamp hasn't been supported on DirectML.");
    }

    MaybeError Graph::AddReduce(const op::Reduce* reduce) {
        return DAWN_UNIMPLEMENTED_ERROR("Reduce hasn't been supported on DirectML.");
    }

    MaybeError Graph::AddResample2d(const op::Resample2d* resample) {
        return DAWN_UNIMPLEMENTED_ERROR("Resample2d hasn't been supported on DirectML.");
    }

    MaybeError Graph::AddReshape(const op::Reshape* reshape) {
        return DAWN_UNIMPLEMENTED_ERROR("Reshape hasn't been supported on DirectML.");
    }

    MaybeError Graph::AddSlice(const op::Slice* slice) {
        return DAWN_UNIMPLEMENTED_ERROR("Slice hasn't been supported on DirectML.");
    }

    MaybeError Graph::AddSplit(const op::Split* split) {
        return DAWN_UNIMPLEMENTED_ERROR("Split hasn't been supported on DirectML.");
    }

    MaybeError Graph::AddSqueeze(const op::Squeeze* squeeze) {
        return DAWN_UNIMPLEMENTED_ERROR("Squeeze hasn't been supported on DirectML.");
    }

    MaybeError Graph::AddTranspose(const op::Transpose* transpose) {
        return DAWN_UNIMPLEMENTED_ERROR("Transpose hasn't been supported on DirectML.");
    }

    MaybeError Graph::AddInstanceNorm(const op::InstanceNorm* instanceNorm) {
        return DAWN_UNIMPLEMENTED_ERROR("InstanceNorm hasn't been supported on DirectML.");
    }

    MaybeError Graph::AddConcat(const op::Concat* concat) {
        return DAWN_UNIMPLEMENTED_ERROR("Concat hasn't been supported on DirectML.");
    }

    MaybeError Graph::AddGemm(const op::Gemm* gemm) {
        return DAWN_UNIMPLEMENTED_ERROR("Gemm hasn't been supported on DirectML.");
    }

    MaybeError Graph::Finish() {
        if (mInputNodes.empty()) {
            return DAWN_VALIDATION_ERROR("Model inputs must be set.");
        }

        WEBNN_CHECK(mDevice.Get()->QueryInterface(IID_PPV_ARGS(&mDevice1)));
        // Compiles a graph of DirectML operators into an object that can be dispatched to the GPU.
        WEBNN_CHECK(mDevice1->CompileGraph(&CreateGraphDesc(), DML_EXECUTION_FLAG_NONE,
                                           IID_PPV_ARGS(&mCompiledOperator)));
        return {};
    }

    MaybeError Graph::CompileImpl() {
        IDMLCompiledOperator* dmlCompiledOperators[] = {mCompiledOperator.Get()};
        WEBNN_CHECK(mDevice->CreateOperatorInitializer(
            ARRAYSIZE(dmlCompiledOperators), dmlCompiledOperators,
            IID_PPV_ARGS(&mCompiledOperatorInitializer)));

        DML_BINDING_PROPERTIES initializeBindingProperties =
            mCompiledOperatorInitializer->GetBindingProperties();
        DML_BINDING_PROPERTIES executeBindingProperties = mCompiledOperator->GetBindingProperties();
        UINT descriptorCount = std::max(initializeBindingProperties.RequiredDescriptorCount,
                                        executeBindingProperties.RequiredDescriptorCount);

        // Create descriptor heaps.
        D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc{};
        descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        descriptorHeapDesc.NumDescriptors = descriptorCount;
        descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        WEBNN_CHECK(mD3D12Device->CreateDescriptorHeap(&descriptorHeapDesc,
                                                       IID_PPV_ARGS(&mDescriptorHeap)));

        // Set the descriptor heap(s).
        ID3D12DescriptorHeap* d3D12DescriptorHeaps[] = {mDescriptorHeap.Get()};
        mCommandList->SetDescriptorHeaps(ARRAYSIZE(d3D12DescriptorHeaps), d3D12DescriptorHeaps);

        // Create a binding table over the descriptor heap we just created.
        mBindingTableDesc.Dispatchable = mCompiledOperatorInitializer.Get();
        mBindingTableDesc.CPUDescriptorHandle =
            mDescriptorHeap->GetCPUDescriptorHandleForHeapStart();
        mBindingTableDesc.GPUDescriptorHandle =
            mDescriptorHeap->GetGPUDescriptorHandleForHeapStart();
        mBindingTableDesc.SizeInDescriptors = descriptorCount;

        WEBNN_CHECK(mDevice->CreateBindingTable(&mBindingTableDesc, IID_PPV_ARGS(&mBindingTable)));

        UINT64 temporaryResourceSize = std::max(initializeBindingProperties.TemporaryResourceSize,
                                                executeBindingProperties.TemporaryResourceSize);
        UINT64 persistentResourceSize = executeBindingProperties.PersistentResourceSize;

        // Bind and initialize the operator on the GPU.

        if (temporaryResourceSize != 0) {
            mD3D12Device->CreateCommittedResource(
                &utils::CreateHeapProperties(), D3D12_HEAP_FLAG_NONE,
                &utils::CreateResourceDesc(temporaryResourceSize,
                                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&mTemporaryBuffer));

            if (initializeBindingProperties.TemporaryResourceSize != 0) {
                DML_BUFFER_BINDING bufferBinding{mTemporaryBuffer.Get(), 0, temporaryResourceSize};
                DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
                mBindingTable->BindTemporaryResource(&bindingDesc);
            }
        }

        if (persistentResourceSize != 0) {
            mD3D12Device->CreateCommittedResource(
                &utils::CreateHeapProperties(), D3D12_HEAP_FLAG_NONE,
                &utils::CreateResourceDesc(persistentResourceSize,
                                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&mPersistentBuffer));

            // The persistent resource should be bound as the output to the
            // IDMLOperatorInitializer.
            DML_BUFFER_BINDING bufferBinding{mPersistentBuffer.Get(), 0, persistentResourceSize};
            DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
            mBindingTable->BindOutputs(1, &bindingDesc);
        }

        // Initialize constant inputs.
        uint64_t constantInputsResourceSize = 0;
        for (auto& input : mInputNodes) {
            if (input.inputInfo.isConstant) {
                uint64_t offset = utils::RoundUpToMultiple(
                    constantInputsResourceSize, (uint64_t)DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);
                constantInputsResourceSize = offset + input.inputInfo.byteLength;
            }
        }

        if (constantInputsResourceSize) {
            std::vector<DML_BUFFER_BINDING> bufferBinding(mInputNodes.size());
            DML_BUFFER_ARRAY_BINDING dmlBufferArrayBinding = {};

            WEBNN_CHECK(mD3D12Device->CreateCommittedResource(
                &utils::CreateHeapProperties(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE,
                &utils::CreateResourceDesc(constantInputsResourceSize),
                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&mUploadBuffer)));

            WEBNN_CHECK(mD3D12Device->CreateCommittedResource(
                &utils::CreateHeapProperties(), D3D12_HEAP_FLAG_NONE,
                &utils::CreateResourceDesc(constantInputsResourceSize,
                                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&mInputBuffer)));

            D3D12_RANGE constantBufferRange{0, constantInputsResourceSize};
            int8_t* constantBuffer;
            WEBNN_CHECK(mUploadBuffer->Map(0, &constantBufferRange,
                                           reinterpret_cast<void**>(&constantBuffer)));
            uint64_t offset = 0;
            for (size_t i = 0; i < mInputNodes.size(); ++i) {
                NodeInfo input = mInputNodes[i];
                if (input.inputInfo.isConstant) {
                    uint32_t requiredAlignment = DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT;
                    offset = utils::RoundUpToMultiple(offset, (uint64_t)requiredAlignment);
                    bufferBinding[i].Buffer = mInputBuffer.Get();
                    bufferBinding[i].Offset = offset;
                    bufferBinding[i].SizeInBytes = input.inputInfo.byteLength;

                    void* dest = constantBuffer + offset;
                    const void* src = input.inputInfo.buffer;
                    memcpy(dest, src, static_cast<size_t>(input.inputInfo.byteLength));
                    offset = offset + input.inputInfo.byteLength;
                }
            }
            dmlBufferArrayBinding.BindingCount = bufferBinding.size();
            dmlBufferArrayBinding.Bindings = bufferBinding.data();
            mUploadBuffer->Unmap(0, nullptr);
            D3D12_RESOURCE_BARRIER inputResourceBarrier = {};
            inputResourceBarrier.Transition.pResource = mInputBuffer.Get();
            inputResourceBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            inputResourceBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
            inputResourceBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            inputResourceBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            inputResourceBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            mCommandList->ResourceBarrier(1, &inputResourceBarrier);
            mCommandList->CopyBufferRegion(mInputBuffer.Get(), 0, mUploadBuffer.Get(), 0,
                                           constantInputsResourceSize);
            inputResourceBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            inputResourceBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            mCommandList->ResourceBarrier(1, &inputResourceBarrier);

            DML_BINDING_DESC inputBindingDesc{DML_BINDING_TYPE_BUFFER_ARRAY,
                                              &dmlBufferArrayBinding};
            mBindingTable->BindInputs(1, &inputBindingDesc);
        }

        // Record execution of the operator initializer.
        // The command recorder is a stateless object that records Dispatches into an existing
        // Direct3D 12 command list.
        WEBNN_CHECK(mDevice->CreateCommandRecorder(IID_PPV_ARGS(&mCommandRecorder)));
        mCommandRecorder->RecordDispatch(mCommandList.Get(), mCompiledOperatorInitializer.Get(),
                                         mBindingTable.Get());
        CloseExecuteResetWait();

        // Bind and execute the operator on the GPU.
        mCommandList->SetDescriptorHeaps(ARRAYSIZE(d3D12DescriptorHeaps), d3D12DescriptorHeaps);
        // Reset the binding table to bind for the operator we want to execute (it was
        // previously used to bind for the initializer).
        mBindingTableDesc.Dispatchable = mCompiledOperator.Get();
        mBindingTable->Reset(&mBindingTableDesc);

        if (temporaryResourceSize != 0) {
            DML_BUFFER_BINDING bufferBinding{mTemporaryBuffer.Get(), 0, temporaryResourceSize};
            DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
            mBindingTable->BindTemporaryResource(&bindingDesc);
        }

        if (persistentResourceSize != 0) {
            DML_BUFFER_BINDING bufferBinding{mPersistentBuffer.Get(), 0, persistentResourceSize};
            DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
            mBindingTable->BindPersistentResource(&bindingDesc);
        }
        return {};
    }

    MLComputeGraphStatus Graph::ComputeImpl(NamedInputsBase* inputs, NamedOutputsBase* outputs) {
        auto namedInputs = inputs->GetRecords();

        // Initialize common inputs.
        uint64_t inputsResourceSize = 0;
        for (auto& input : mInputNodes) {
            // All the inputs must be set.

            if (!input.inputInfo.isConstant && namedInputs.find(input.name) == namedInputs.end()) {
                dawn::ErrorLog() << "The input must be set.";
                return MLComputeGraphStatus_Error;
            }

            if (!input.inputInfo.isConstant) {
                auto& resource = namedInputs[input.name]->resource;
                uint64_t offset = utils::RoundUpToMultiple(
                    inputsResourceSize, (uint64_t)DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);
                inputsResourceSize = offset + resource.byteLength;
            }
        }

        if (inputsResourceSize) {
            WEBNN_CHECK(mD3D12Device->CreateCommittedResource(
                &utils::CreateHeapProperties(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE,
                &utils::CreateResourceDesc(inputsResourceSize), D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr, IID_PPV_ARGS(&mUploadBuffer)));

            WEBNN_CHECK(mD3D12Device->CreateCommittedResource(
                &utils::CreateHeapProperties(), D3D12_HEAP_FLAG_NONE,
                &utils::CreateResourceDesc(inputsResourceSize,
                                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&mInputBuffer)));

            std::vector<DML_BINDING_DESC> bindingDesc(mInputNodes.size());
            std::vector<DML_BUFFER_BINDING> bufferBinding(mInputNodes.size());

            D3D12_RANGE inputBufferRange{0, inputsResourceSize};
            int8_t* inputBuffer;
            WEBNN_CHECK(
                mUploadBuffer->Map(0, &inputBufferRange, reinterpret_cast<void**>(&inputBuffer)));

            uint64_t offset = 0;
            for (size_t i = 0; i < mInputNodes.size(); ++i) {
                NodeInfo input = mInputNodes[i];
                if (!input.inputInfo.isConstant) {
                    uint32_t requiredAlignment = DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT;
                    offset = utils::RoundUpToMultiple(offset, (uint64_t)requiredAlignment);
                    auto& resource = namedInputs[input.name]->resource;
                    bufferBinding[i].Buffer = mInputBuffer.Get();
                    bufferBinding[i].Offset = offset;
                    bufferBinding[i].SizeInBytes = resource.byteLength;
                    bindingDesc[i] = {DML_BINDING_TYPE_BUFFER, &bufferBinding[i]};
                    void* dest = inputBuffer + offset;
                    memcpy(dest, static_cast<int8_t*>(resource.buffer) + resource.byteOffset,
                           resource.byteLength);
                    offset = offset + bufferBinding[i].SizeInBytes;
                }
            }
            mUploadBuffer->Unmap(0, nullptr);

            D3D12_RESOURCE_BARRIER inputResourceBarrier = {};
            inputResourceBarrier.Transition.pResource = mInputBuffer.Get();
            inputResourceBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            inputResourceBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
            inputResourceBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            inputResourceBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            inputResourceBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            mCommandList->ResourceBarrier(1, &inputResourceBarrier);
            mCommandList->CopyBufferRegion(mInputBuffer.Get(), 0, mUploadBuffer.Get(), 0,
                                           inputsResourceSize);
            inputResourceBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            inputResourceBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            mCommandList->ResourceBarrier(1, &inputResourceBarrier);
            mBindingTable->BindInputs(bindingDesc.size(), bindingDesc.data());
        }

        // Prepare for outputs and read back buffer from Gpu.
        uint64_t outputsResourceSize = 0;
        auto namedOutputs = outputs->GetRecords();
        for (auto namedOutput : outputs->GetRecords()) {
            const ArrayBufferView* output = namedOutput.second;
            DAWN_ASSERT(output->buffer != nullptr && output->byteLength != 0);
            outputsResourceSize += output->byteLength;
        }

        WEBNN_CHECK(mD3D12Device->CreateCommittedResource(
            &utils::CreateHeapProperties(), D3D12_HEAP_FLAG_NONE,
            &utils::CreateResourceDesc(outputsResourceSize,
                                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&mOutputBuffer)));

        DML_BUFFER_BINDING outputBufferBinding{mOutputBuffer.Get(), 0, outputsResourceSize};
        DML_BINDING_DESC outputBindingDesc{DML_BINDING_TYPE_BUFFER, &outputBufferBinding};
        mBindingTable->BindOutputs(1, &outputBindingDesc);

        // Record execution of the compiled operator.
        mCommandRecorder->RecordDispatch(mCommandList.Get(), mCompiledOperator.Get(),
                                         mBindingTable.Get());
        CloseExecuteResetWait();

        Microsoft::WRL::ComPtr<ID3D12Resource> readbackBuffer;
        mD3D12Device->CreateCommittedResource(
            &utils::CreateHeapProperties(D3D12_HEAP_TYPE_READBACK), D3D12_HEAP_FLAG_NONE,
            &utils::CreateResourceDesc(outputsResourceSize), D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr, IID_PPV_ARGS(&readbackBuffer));

        D3D12_RESOURCE_BARRIER outputResourceBarrier = {};
        outputResourceBarrier.Transition.pResource = mOutputBuffer.Get();
        outputResourceBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        outputResourceBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        outputResourceBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        outputResourceBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        outputResourceBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        mCommandList->ResourceBarrier(1, &outputResourceBarrier);
        mCommandList->CopyResource(readbackBuffer.Get(), mOutputBuffer.Get());

        CloseExecuteResetWait();

        D3D12_RANGE tensorBufferRange{0, outputsResourceSize};
        int8_t* outputBuffer;
        WEBNN_CHECK(
            readbackBuffer->Map(0, &tensorBufferRange, reinterpret_cast<void**>(&outputBuffer)));

        std::vector<std::string> outputNames;
        for (auto& output : mOutputNodes) {
            outputNames.push_back(output.name);
        }

        for (size_t i = 0; i < outputNames.size(); ++i) {
            std::string outputName = outputNames[i];
            auto namedOutputs = outputs->GetRecords();
            if (namedOutputs.find(outputName) != namedOutputs.end()) {
                const ArrayBufferView* output = namedOutputs[outputName];
                memcpy(static_cast<int8_t*>(output->buffer) + output->byteOffset,
                       outputBuffer + output->byteOffset, output->byteLength);
            }
        }

        readbackBuffer->Unmap(0, nullptr);
        return MLComputeGraphStatus_Success;
    }
}}  // namespace webnn_native::dml
