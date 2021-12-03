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

#define CHECK_ERROR(hr)                             \
    if (((HRESULT)(hr)) < 0) {                      \
        dawn::ErrorLog() << "Failed to do " << #hr; \
    }

namespace webnn_native { namespace dml {

    MaybeError Graph::AddToGraph(std::vector<OperatorNode> inputNodes) {
        for (size_t i = 0; i < inputNodes.size(); ++i) {
            OperatorNode inputNode = inputNodes[i];
            if (inputNode.nodeType == NodeType::Input) {
                DML_INPUT_GRAPH_EDGE_DESC inputEdge = {};
                inputEdge.GraphInputIndex = inputNode.inputInfo.inputIndex;
                inputEdge.ToNodeIndex = mNodeIndex;
                inputEdge.ToNodeInputIndex = i;
                inputEdge.Name = "";
                mInputEdges.push_back(inputEdge);
            } else if (inputNode.nodeType == NodeType::Operator) {
                DML_INTERMEDIATE_GRAPH_EDGE_DESC intermediateEdge = {};
                intermediateEdge.FromNodeIndex = inputNode.nodeIndex;
                intermediateEdge.FromNodeOutputIndex = inputNode.outputNodeIndex;
                intermediateEdge.ToNodeIndex = mNodeIndex;
                intermediateEdge.ToNodeInputIndex = i;
                mIntermediateEdges.push_back(intermediateEdge);
            } else {
                return DAWN_INTERNAL_ERROR("Invalid node type.");
            }
        }
        mNodeIndex++;
        return {};
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

    void Graph::InitializeDirect3D12() {
#if defined(_DEBUG)
        Microsoft::WRL::ComPtr<ID3D12Debug> d3D12Debug;
        if (D3D12GetDebugInterface(IID_PPV_ARGS(&d3D12Debug)) < 0) {
            // The D3D12 debug layer is missing - you must install the Graphics Tools optional
            // feature
            DAWN_ASSERT(0);
        }
        d3D12Debug->EnableDebugLayer();
#endif
        Microsoft::WRL::ComPtr<IDXGIFactory4> dxgiFactory;
        CHECK_ERROR(CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory)));
        Microsoft::WRL::ComPtr<IDXGIAdapter> dxgiAdapter;
        UINT adapterIndex{};
        HRESULT hr{};
        do {
            dxgiAdapter = nullptr;
            IDXGIAdapter* dxgiAdapterPtr = dxgiAdapter.Get();
            CHECK_ERROR(dxgiFactory->EnumAdapters(adapterIndex, &dxgiAdapterPtr));
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
        CHECK_ERROR(
            mD3D12Device->CreateCommandQueue(&commandQueueDesc, IID_PPV_ARGS(&mCommandQueue)));

        CHECK_ERROR(mD3D12Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                         IID_PPV_ARGS(&mCommandAllocator)));

        CHECK_ERROR(mD3D12Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                    mCommandAllocator.Get(), nullptr,
                                                    IID_PPV_ARGS(&mCommandList)));
    }

    Graph::Graph(Context* context) : GraphBase(context) {
        // Set up Direct3D 12.
        InitializeDirect3D12();

        // Create the DirectML device.
        DML_CREATE_DEVICE_FLAGS dmlCreateDeviceFlags = DML_CREATE_DEVICE_FLAG_NONE;
        CHECK_ERROR(
            DMLCreateDevice(mD3D12Device.Get(), dmlCreateDeviceFlags, IID_PPV_ARGS(&mDevice)));
    }

    MaybeError Graph::AddConstant(const op::Constant* constant) {
        const OperandDescriptor* desc = constant->GetOperandDescriptor();
        mTensorDescMap.insert(std::make_pair(mInputs.size(), TensorDesc{}));
        if (!GetDMLTensorDesc(desc, mTensorDescMap[mInputs.size()], DML_TENSOR_FLAG_OWNED_BY_DML)) {
            return DAWN_INTERNAL_ERROR("Failed to get DML Buffer tensor.");
        }
        const void* ptr = &mTensorDescMap[mInputs.size()].bufferDesc;
        DML_TENSOR_DESC dmlTensorDesc = {DML_TENSOR_TYPE_BUFFER, ptr};
        InputInfo constantInfo = {mInputs.size(), constant->GetBuffer(), constant->GetByteLength(),
                                  true, 0};
        OperatorNode operatorNode = {NodeType::Input,
                                     mNodeIndex,
                                     0,
                                     dmlTensorDesc,
                                     "Input_Constant_" + std::to_string(mInputs.size()),
                                     constantInfo};
        mOperatorMap[constant->PrimaryOutput()] = operatorNode;
        mInputs.push_back(operatorNode);
        return {};
    }

    MaybeError Graph::AddInput(const op::Input* input) {
        const OperandDescriptor* desc = input->GetOperandDescriptor();
        mTensorDescMap.insert(std::make_pair(mInputs.size(), TensorDesc{}));
        if (!GetDMLTensorDesc(desc, mTensorDescMap[mInputs.size()])) {
            return DAWN_INTERNAL_ERROR("Failed to get DML Buffer tensor.");
        }
        const void* ptr = &mTensorDescMap[mInputs.size()].bufferDesc;
        DML_TENSOR_DESC dmlTensorDesc = {DML_TENSOR_TYPE_BUFFER, ptr};
        InputInfo constantInfo = {mInputs.size(), nullptr, 0, false, 0};
        OperatorNode operatorNode = {NodeType::Input, mNodeIndex,       0,
                                     dmlTensorDesc,   input->GetName(), constantInfo};
        mOperatorMap[input->PrimaryOutput()] = operatorNode;
        mInputs.push_back(operatorNode);
        return {};
    }

    MaybeError Graph::AddBinary(const op::Binary* binary) {
        switch (binary->GetType()) {
            case op::BinaryOpType::kAdd: {
                DAWN_ASSERT(mOperatorMap.find(binary->Inputs()[0].Get()) != mOperatorMap.end());
                DAWN_ASSERT(mOperatorMap.find(binary->Inputs()[1].Get()) != mOperatorMap.end());
                OperatorNode inputNodeA = mOperatorMap[binary->Inputs()[0].Get()];
                OperatorNode inputNodeB = mOperatorMap[binary->Inputs()[1].Get()];
                DML_ELEMENT_WISE_ADD_OPERATOR_DESC dmlAddOperatorDesc{};
                dmlAddOperatorDesc.ATensor = &inputNodeA.outputDESC;
                dmlAddOperatorDesc.BTensor = &inputNodeB.outputDESC;

                DML_TENSOR_DESC outputTensorDesc = inputNodeA.outputDESC;
                dmlAddOperatorDesc.OutputTensor = &outputTensorDesc;

                DML_OPERATOR_DESC dmlOperatorDesc = {};
                dmlOperatorDesc.Type = DML_OPERATOR_ELEMENT_WISE_ADD;
                dmlOperatorDesc.Desc = &dmlAddOperatorDesc;

                Microsoft::WRL::ComPtr<IDMLOperator> dmlOperator;
                CHECK_ERROR(mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));
                mCompiledOperatorMap[mNodes.size()] = dmlOperator;
                mNodes.push_back(
                    DML_OPERATOR_GRAPH_NODE_DESC{mCompiledOperatorMap[mNodes.size()].Get()});

                OperatorNode operatorNode = {NodeType::Operator, mNodeIndex, 0, outputTensorDesc};
                mOperatorMap[binary->PrimaryOutput()] = operatorNode;
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
                DAWN_ASSERT(mOperatorMap.find(unary->Inputs()[0].Get()) != mOperatorMap.end());
                OperatorNode inputNode = mOperatorMap[unary->Inputs()[0].Get()];
                DML_TENSOR_DESC inputTensorDesc = inputNode.outputDESC;
                DML_ACTIVATION_SIGMOID_OPERATOR_DESC dmlSigmoidOperatorDesc{};
                dmlSigmoidOperatorDesc.InputTensor = &inputTensorDesc;
                dmlSigmoidOperatorDesc.OutputTensor = &inputTensorDesc;

                DML_OPERATOR_DESC dmlOperatorDesc = {};
                dmlOperatorDesc.Type = DML_OPERATOR_ACTIVATION_SIGMOID;
                dmlOperatorDesc.Desc = &dmlSigmoidOperatorDesc;

                CHECK_ERROR(mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));
                mCompiledOperatorMap[mNodes.size()] = dmlOperator;
                mNodes.push_back(
                    DML_OPERATOR_GRAPH_NODE_DESC{mCompiledOperatorMap[mNodes.size()].Get()});

                OperatorNode operatorNode = {NodeType::Operator, mNodeIndex, 0, inputTensorDesc};
                mOperatorMap[unary->PrimaryOutput()] = operatorNode;
                return AddToGraph({inputNode});
                break;
            }
            default:
                return DAWN_UNIMPLEMENTED_ERROR(" Unary op is not implemented.");
        }
        return {};
    }

    MaybeError Graph::AddOutput(const std::string& name, const OperandBase* output) {
        auto operatorNode = mOperatorMap[output];
        if (operatorNode.nodeType == NodeType::Input) {
            return DAWN_INTERNAL_ERROR("Graph for input = output is invalid.");
        }
        operatorNode.name = name;
        DML_OUTPUT_GRAPH_EDGE_DESC outputEdge = {};
        outputEdge.FromNodeIndex = operatorNode.nodeIndex;
        outputEdge.FromNodeOutputIndex = operatorNode.outputNodeIndex;
        outputEdge.GraphOutputIndex = mOutputs.size();
        mOutputEdges.push_back(outputEdge);
        mOutputs.push_back(operatorNode);
        return {};
    }

    MaybeError Graph::AddBatchNorm(const op::BatchNorm* batchNorm) {
        return {};
    }

    MaybeError Graph::AddConv2d(const op::Conv2d* conv2d) {
        return {};
    }

    MaybeError Graph::AddConvTranspose2d(const op::ConvTranspose2d* convTranspose2d) {
        return DAWN_UNIMPLEMENTED_ERROR("ConvTranspose2D has not been supported on DirectML.");
    }

    MaybeError Graph::AddGru(const op::Gru* gru) {
        return DAWN_UNIMPLEMENTED_ERROR("Gru hasn't been supported on DirectML.");
    }

    MaybeError Graph::AddPad(const op::Pad* pad) {
        return {};
    }

    MaybeError Graph::AddPool2d(const op::Pool2d* pool2d) {
        return {};
    }

    MaybeError Graph::AddClamp(const op::Clamp* clamp) {
        return {};
    }

    MaybeError Graph::AddReduce(const op::Reduce* reduce) {
        return {};
    }

    MaybeError Graph::AddResample(const op::Resample* resample) {
        return {};
    }

    MaybeError Graph::AddReshape(const op::Reshape* reshape) {
        return {};
    }

    MaybeError Graph::AddSlice(const op::Slice* slice) {
        return {};
    }

    MaybeError Graph::AddSplit(const op::Split* split) {
        return {};
    }

    MaybeError Graph::AddSqueeze(const op::Squeeze* squeeze) {
        return {};
    }

    MaybeError Graph::AddTranspose(const op::Transpose* transpose) {
        return {};
    }

    MaybeError Graph::AddInstanceNorm(const op::InstanceNorm* instanceNorm) {
        return {};
    }

    MaybeError Graph::AddConcat(const op::Concat* concat) {
        return {};
    }

    MaybeError Graph::AddGemm(const op::Gemm* gemm) {
        return {};
    }

    MaybeError Graph::Finish() {
        if (mInputs.empty()) {
            return DAWN_VALIDATION_ERROR("Model inputs must be set.");
        }
        return {};
    }

    DML_GRAPH_DESC Graph::CreateGraphDesc(std::vector<DML_GRAPH_NODE_DESC>& nodes,
                                          std::vector<DML_GRAPH_EDGE_DESC>& inputEdges,
                                          std::vector<DML_GRAPH_EDGE_DESC>& outputEdges,
                                          std::vector<DML_GRAPH_EDGE_DESC>& intermediateEdges) {
        DML_GRAPH_DESC graphDesc = {};
        graphDesc.InputCount = static_cast<UINT>(mInputs.size());
        graphDesc.OutputCount = static_cast<UINT>(mOutputs.size());

        nodes.resize(mNodes.size());
        for (size_t i = 0; i < mNodes.size(); ++i) {
            nodes[i] = {DML_GRAPH_NODE_TYPE_OPERATOR, &mNodes[i]};
        }
        graphDesc.NodeCount = static_cast<UINT>(mNodes.size());
        graphDesc.Nodes = nodes.data();

        inputEdges.resize(mInputEdges.size());
        for (size_t i = 0; i < mInputEdges.size(); ++i) {
            inputEdges[i] = {DML_GRAPH_EDGE_TYPE_INPUT, &mInputEdges[i]};
        }
        graphDesc.InputEdgeCount = static_cast<UINT>(mInputEdges.size());
        graphDesc.InputEdges = inputEdges.data();

        outputEdges.resize(mOutputEdges.size());
        for (size_t i = 0; i < mOutputEdges.size(); ++i) {
            outputEdges[i] = {DML_GRAPH_EDGE_TYPE_OUTPUT, &mOutputEdges[i]};
        }
        graphDesc.OutputEdgeCount = static_cast<UINT>(mOutputEdges.size());
        graphDesc.OutputEdges = outputEdges.data();

        intermediateEdges.resize(mIntermediateEdges.size());
        for (size_t i = 0; i < mIntermediateEdges.size(); ++i) {
            intermediateEdges[i] = {DML_GRAPH_EDGE_TYPE_INTERMEDIATE, &mIntermediateEdges[i]};
        }
        graphDesc.IntermediateEdgeCount = static_cast<UINT>(mIntermediateEdges.size());
        graphDesc.IntermediateEdges = intermediateEdges.data();

        return graphDesc;
    }

    void Graph::CloseExecuteResetWait() {
        mCommandList->Close();

        ID3D12CommandList* commandLists[] = {mCommandList.Get()};
        mCommandQueue->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);

        ComPtr<ID3D12Device> device;
        mCommandQueue.Get()->GetDevice(IID_PPV_ARGS(device.GetAddressOf()));
        ComPtr<ID3D12Fence> fence;
        device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.GetAddressOf()));
        mCommandQueue.Get()->Signal(fence.Get(), 1);
        fence->SetEventOnCompletion(1, nullptr);
        mCommandAllocator->Reset();
        mCommandList->Reset(mCommandAllocator.Get(), nullptr);
        // CHECK_ERROR(m_residencySet.Reset());
    }

    MaybeError Graph::CompileImpl() {
        CHECK_ERROR(mDevice.Get()->QueryInterface(IID_PPV_ARGS(&mDevice1)));

        std::vector<DML_GRAPH_NODE_DESC> nodes;
        std::vector<DML_GRAPH_EDGE_DESC> inputEdges;
        std::vector<DML_GRAPH_EDGE_DESC> outputEdges;
        std::vector<DML_GRAPH_EDGE_DESC> intermediateEdges;

        DML_GRAPH_DESC graphDesc =
            CreateGraphDesc(nodes, inputEdges, outputEdges, intermediateEdges);

        CHECK_ERROR(mDevice1->CompileGraph(&graphDesc, DML_EXECUTION_FLAG_NONE,
                                           IID_PPV_ARGS(&mCompiledOperator)));

        IDMLCompiledOperator* dmlCompiledOperators[] = {mCompiledOperator.Get()};
        CHECK_ERROR(mDevice->CreateOperatorInitializer(ARRAYSIZE(dmlCompiledOperators),
                                                       dmlCompiledOperators,
                                                       IID_PPV_ARGS(&mOperatorInitializer)));

        DML_BINDING_PROPERTIES initializeBindingProperties =
            mOperatorInitializer->GetBindingProperties();
        DML_BINDING_PROPERTIES executeBindingProperties = mCompiledOperator->GetBindingProperties();
        UINT descriptorCount = std::max(initializeBindingProperties.RequiredDescriptorCount,
                                        executeBindingProperties.RequiredDescriptorCount);

        // Create descriptor heaps.
        D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc{};
        descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        descriptorHeapDesc.NumDescriptors = descriptorCount;
        descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        CHECK_ERROR(mD3D12Device->CreateDescriptorHeap(&descriptorHeapDesc,
                                                       IID_PPV_ARGS(&mDescriptorHeap)));

        // Set the descriptor heap(s).
        ID3D12DescriptorHeap* d3D12DescriptorHeaps[] = {mDescriptorHeap.Get()};
        mCommandList->SetDescriptorHeaps(ARRAYSIZE(d3D12DescriptorHeaps), d3D12DescriptorHeaps);

        // Create a binding table over the descriptor heap we just created.
        mBindingTableDesc.Dispatchable = mOperatorInitializer.Get();
        mBindingTableDesc.CPUDescriptorHandle =
            mDescriptorHeap->GetCPUDescriptorHandleForHeapStart();
        mBindingTableDesc.GPUDescriptorHandle =
            mDescriptorHeap->GetGPUDescriptorHandleForHeapStart();
        mBindingTableDesc.SizeInDescriptors = descriptorCount;

        CHECK_ERROR(mDevice->CreateBindingTable(&mBindingTableDesc, IID_PPV_ARGS(&mBindingTable)));

        UINT64 temporaryResourceSize = std::max(initializeBindingProperties.TemporaryResourceSize,
                                                executeBindingProperties.TemporaryResourceSize);
        UINT64 persistentResourceSize = executeBindingProperties.PersistentResourceSize;

        // Bind and initialize the operator on the GPU.
        Microsoft::WRL::ComPtr<ID3D12Resource> temporaryBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> persistentBuffer;

        if (temporaryResourceSize != 0) {
            mD3D12Device->CreateCommittedResource(
                &utils::CreateHeapProperties(), D3D12_HEAP_FLAG_NONE,
                &utils::CreateResourceDesc(temporaryResourceSize,
                                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&temporaryBuffer));

            if (initializeBindingProperties.TemporaryResourceSize != 0) {
                DML_BUFFER_BINDING bufferBinding{temporaryBuffer.Get(), 0, temporaryResourceSize};
                DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
                mBindingTable->BindTemporaryResource(&bindingDesc);
            }
        }

        if (persistentResourceSize != 0) {
            mD3D12Device->CreateCommittedResource(
                &utils::CreateHeapProperties(), D3D12_HEAP_FLAG_NONE,
                &utils::CreateResourceDesc(persistentResourceSize), D3D12_RESOURCE_STATE_COMMON,
                nullptr, IID_PPV_ARGS(&persistentBuffer));

            // The persistent resource should be bound as the output to the
            // IDMLOperatorInitializer.
            DML_BUFFER_BINDING bufferBinding{persistentBuffer.Get(), 0, persistentResourceSize};
            DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
            mBindingTable->BindOutputs(1, &bindingDesc);
        }

        // Initialize constant input
        uint64_t inputsResourceSize = 0;
        for (auto& input : mInputs) {
            if (input.inputInfo.isConstant) {
                uint32_t requiredAlignment = std::max(input.inputInfo.guaranteedBaseOffsetAlignment,
                                                      DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);
                uint64_t offset =
                    utils::RoundUpToMultiple(inputsResourceSize, (uint64_t)requiredAlignment);
                inputsResourceSize = offset + input.inputInfo.byteLength;
            }
        }

        if (inputsResourceSize) {
            std::vector<DML_BUFFER_BINDING> bufferBinding(mInputs.size());
            DML_BUFFER_ARRAY_BINDING dmlBufferArrayBinding = {};

            CHECK_ERROR(mD3D12Device->CreateCommittedResource(
                &utils::CreateHeapProperties(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE,
                &utils::CreateResourceDesc(inputsResourceSize), D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr, IID_PPV_ARGS(&mUploadBuffer)));

            CHECK_ERROR(mD3D12Device->CreateCommittedResource(
                &utils::CreateHeapProperties(), D3D12_HEAP_FLAG_NONE,
                &utils::CreateResourceDesc(inputsResourceSize,
                                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&mInputBuffer)));

            D3D12_RANGE constantBufferRange{0, inputsResourceSize};
            int8_t* constantBuffer;
            CHECK_ERROR(mUploadBuffer->Map(0, &constantBufferRange,
                                           reinterpret_cast<void**>(&constantBuffer)));
            uint64_t offset = 0;
            for (size_t i = 0; i < mInputs.size(); ++i) {
                OperatorNode input = mInputs[i];
                if (input.inputInfo.isConstant) {
                    uint32_t requiredAlignment =
                        std::max(input.inputInfo.guaranteedBaseOffsetAlignment,
                                 DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);
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
                                           inputsResourceSize);
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
        CHECK_ERROR(mDevice->CreateCommandRecorder(IID_PPV_ARGS(&mCommandRecorder)));
        mCommandRecorder->RecordDispatch(mCommandList.Get(), mOperatorInitializer.Get(),
                                         mBindingTable.Get());
        CloseExecuteResetWait();

        // Bind and execute the operator on the GPU.
        mCommandList->SetDescriptorHeaps(ARRAYSIZE(d3D12DescriptorHeaps), d3D12DescriptorHeaps);
        // Reset the binding table to bind for the operator we want to execute (it was
        // previously used to bind for the initializer).
        mBindingTableDesc.Dispatchable = mCompiledOperator.Get();
        mBindingTable->Reset(&mBindingTableDesc);

        if (temporaryResourceSize != 0) {
            DML_BUFFER_BINDING bufferBinding{temporaryBuffer.Get(), 0, temporaryResourceSize};
            DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
            mBindingTable->BindTemporaryResource(&bindingDesc);
        }

        if (persistentResourceSize != 0) {
            DML_BUFFER_BINDING bufferBinding{persistentBuffer.Get(), 0, persistentResourceSize};
            DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
            mBindingTable->BindPersistentResource(&bindingDesc);
        }
        return {};
    }

    WNNComputeGraphStatus Graph::ComputeImpl(NamedInputsBase* inputs, NamedOutputsBase* outputs) {
        auto namedInputs = inputs->GetRecords();

        // Initialize common inputs
        uint64_t inputsResourceSize = 0;
        for (auto& input : mInputs) {
            // All the inputs must be set.
            if (namedInputs.find(input.name) == namedInputs.end()) {
                dawn::ErrorLog() << "The input must be set.";
                return WNNComputeGraphStatus_Error;
            }

            if (!input.inputInfo.isConstant) {
                auto& resource = namedInputs[input.name]->resource;
                uint32_t requiredAlignment = std::max(input.inputInfo.guaranteedBaseOffsetAlignment,
                                                      DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);
                uint64_t offset =
                    utils::RoundUpToMultiple(inputsResourceSize, (uint64_t)requiredAlignment);
                inputsResourceSize = offset + resource.byteLength;
            }
        }

        if (inputsResourceSize) {
            CHECK_ERROR(mD3D12Device->CreateCommittedResource(
                &utils::CreateHeapProperties(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE,
                &utils::CreateResourceDesc(inputsResourceSize), D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr, IID_PPV_ARGS(&mUploadBuffer)));

            CHECK_ERROR(mD3D12Device->CreateCommittedResource(
                &utils::CreateHeapProperties(), D3D12_HEAP_FLAG_NONE,
                &utils::CreateResourceDesc(inputsResourceSize,
                                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&mInputBuffer)));

            std::vector<DML_BINDING_DESC> bindingDesc(mInputs.size());
            std::vector<DML_BUFFER_BINDING> bufferBinding(mInputs.size());

            D3D12_RANGE inputBufferRange{0, inputsResourceSize};
            int8_t* inputBuffer;
            CHECK_ERROR(
                mUploadBuffer->Map(0, &inputBufferRange, reinterpret_cast<void**>(&inputBuffer)));

            uint64_t offset = 0;
            for (size_t i = 0; i < mInputs.size(); ++i) {
                OperatorNode input = mInputs[i];
                if (!input.inputInfo.isConstant) {
                    uint32_t requiredAlignment =
                        std::max(input.inputInfo.guaranteedBaseOffsetAlignment,
                                 DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);
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

        // Prepare for outputs and read back buffer from Gpu
        uint64_t outputsResourceSize = 0;
        auto namedOutputs = outputs->GetRecords();
        for (auto namedOutput : outputs->GetRecords()) {
            const ArrayBufferView* output = namedOutput.second;
            DAWN_ASSERT(output->buffer != nullptr && output->byteLength != 0);
            outputsResourceSize += output->byteLength;
        }

        CHECK_ERROR(mD3D12Device->CreateCommittedResource(
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
        CHECK_ERROR(
            readbackBuffer->Map(0, &tensorBufferRange, reinterpret_cast<void**>(&outputBuffer)));

        std::vector<std::string> outputNames;
        for (auto& output : mOutputs) {
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
