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

namespace webnn_native { namespace dml {

    void CopyBufferRegion(ComPtr<ID3D12GraphicsCommandList> commandList,
                          ComPtr<ID3D12Resource> srcResource,
                          ComPtr<ID3D12Resource> destResource,
                          UINT64 resourceSize,
                          D3D12_RESOURCE_STATES state) {
        srcResource->Unmap(0, nullptr);
        D3D12_RESOURCE_BARRIER resourceBarrier;
        if (state == D3D12_RESOURCE_STATE_COPY_DEST) {
            resourceBarrier.Transition.pResource = destResource.Get();
        } else if (state == D3D12_RESOURCE_STATE_COPY_SOURCE) {
            resourceBarrier.Transition.pResource = srcResource.Get();
        } else {
            dawn::ErrorLog() << "Unsupported D3D12_RESOURCE_STATES.";
            DAWN_ASSERT(0);
        }
        resourceBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        resourceBarrier.Transition.StateAfter = state;
        resourceBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        resourceBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        resourceBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        commandList->ResourceBarrier(1, &resourceBarrier);
        commandList->CopyBufferRegion(destResource.Get(), 0, srcResource.Get(), 0, resourceSize);
        resourceBarrier.Transition.StateBefore = state;
        resourceBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        commandList->ResourceBarrier(1, &resourceBarrier);
    }

    // Strides are used to express broadcasting (by specifying a stride of 0) as well as
    // padding. If Strides is not specified, each dimension in the tensor is considered to
    // be contiguously packed, with no additional padding. The calculated strides refer to
    // https://docs.microsoft.com/en-us/windows/win32/direct3d12/dml-helper-functions#calculatestrides
    std::vector<UINT> CalculateBroadcastStrides(std::vector<UINT> dims,
                                                std::vector<bool> broadcast = {}) {
        size_t rank = dims.size();
        if (broadcast.empty()) {
            broadcast.resize(rank, false);
        }
        for (size_t i = 0; i < rank; ++i) {
            if (broadcast[i]) {
                dims[i] = 1;
            }
        }
        std::vector<UINT> strides(rank);
        strides[rank - 1] = broadcast[rank - 1] ? 0 : 1;
        size_t elements = 1;
        for (size_t i = 1; i < rank; i++) {
            size_t j = dims.size() - i - 1;
            elements *= dims[j + 1];
            strides[j] = broadcast[j] ? 0 : elements;
        }
        return strides;
    }

    std::vector<UINT> ConvertDimensions(const std::vector<int32_t>& dimensions) {
        std::vector<UINT> convertedDimensions;
        for (auto dim : dimensions) {
            DAWN_ASSERT(dim > 0);
            convertedDimensions.push_back(dim);
        }
        return convertedDimensions;
    }

    inline D3D12_HEAP_PROPERTIES CreateHeapProperties(
        D3D12_HEAP_TYPE type = D3D12_HEAP_TYPE_DEFAULT) {
        return {type, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
    };

    inline D3D12_RESOURCE_DESC CreateResourceDesc(
        UINT64 width,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
        return {D3D12_RESOURCE_DIMENSION_BUFFER, 0,    width, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                D3D12_TEXTURE_LAYOUT_ROW_MAJOR,  flags};
    };

    bool CreateDmlTensorDesc(const std::unique_ptr<DmlTensorDesc>& dmlTensorDesc,
                             const std::vector<UINT>& dimensions = {},
                             const std::vector<UINT>& strides = {},
                             DML_TENSOR_DATA_TYPE dataType = DML_TENSOR_DATA_TYPE_FLOAT32,
                             DML_TENSOR_FLAGS tensorFlag = DML_TENSOR_FLAG_NONE) {
        dmlTensorDesc->dimensions = dimensions;
        dmlTensorDesc->strides = strides;

        size_t typeLength = 4;
        switch (dataType) {
            case DML_TENSOR_DATA_TYPE_FLOAT32:
            case DML_TENSOR_DATA_TYPE_INT32:
            case DML_TENSOR_DATA_TYPE_UINT32:
                typeLength = 4;
                break;
            case DML_TENSOR_DATA_TYPE_FLOAT16:
                typeLength = 2;
                break;
            default:
                DAWN_ASSERT(0);
        }

        size_t bufferLength = typeLength;
        if (dmlTensorDesc->dimensions.size() > DML_TENSOR_DIMENSION_COUNT_MAX) {
            dawn::ErrorLog() << "Tensor dimension count " << dmlTensorDesc->dimensions.size()
                             << " is greater than DML_TENSOR_DIMENSION_COUNT_MAX "
                             << DML_TENSOR_DIMENSION_COUNT_MAX;
            return false;
        }
        if (dmlTensorDesc->dimensions.size() == 0) {
            dmlTensorDesc->dimensions.resize(1);
            dmlTensorDesc->dimensions[0] = 1;
        } else {
            for (uint32_t i = 0; i < dmlTensorDesc->dimensions.size(); ++i) {
                int32_t d = dmlTensorDesc->dimensions[i];
                if (d < 0) {
                    dawn::ErrorLog() << "DML doesn't support the negative dimension value";
                    return false;
                }
                dmlTensorDesc->dimensions[i] = d;
                bufferLength *= dmlTensorDesc->dimensions[i];
            }
        }

        dmlTensorDesc->bufferDesc.DimensionCount = dmlTensorDesc->dimensions.size();
        dmlTensorDesc->bufferDesc.Sizes = dmlTensorDesc->dimensions.data();
        dmlTensorDesc->bufferDesc.Strides = dmlTensorDesc->strides.data();
        dmlTensorDesc->bufferDesc.TotalTensorSizeInBytes = bufferLength;
        dmlTensorDesc->bufferDesc.GuaranteedBaseOffsetAlignment = 0;
        dmlTensorDesc->bufferDesc.DataType = dataType;
        dmlTensorDesc->bufferDesc.Flags = tensorFlag;
        return true;
    }

    bool CreateDmlTensorDesc(const std::unique_ptr<DmlTensorDesc>& dmlTensorDesc,
                             OperandDescriptor const* desc,
                             DML_TENSOR_FLAGS tensorFlag = DML_TENSOR_FLAGS::DML_TENSOR_FLAG_NONE) {
        DAWN_ASSERT(desc != nullptr);
        std::vector<UINT> dimensions;
        DML_TENSOR_DATA_TYPE dataType;
        dimensions.assign(desc->dimensions, desc->dimensions + desc->dimensionsCount);
        if (desc->type == wnn::OperandType::Float32) {
            dataType = DML_TENSOR_DATA_TYPE_FLOAT32;
        } else if (desc->type == wnn::OperandType::Float16) {
            dataType = DML_TENSOR_DATA_TYPE_FLOAT16;
        } else if (desc->type == wnn::OperandType::Int32) {
            dataType = DML_TENSOR_DATA_TYPE_INT32;
        } else if (desc->type == wnn::OperandType::Uint32) {
            dataType = DML_TENSOR_DATA_TYPE_UINT32;
        } else {
            return false;
        }

        return CreateDmlTensorDesc(dmlTensorDesc, dimensions, {}, dataType, tensorFlag);
    }

    bool CreateDmlTensorDesc(const std::unique_ptr<DmlTensorDesc>& dmlTensorDesc,
                             DML_TENSOR_DESC* tensorDESC,
                             const std::vector<UINT>& dimensions = {},
                             const std::vector<UINT>& strides = {}) {
        DAWN_ASSERT(tensorDESC != nullptr);
        const DML_BUFFER_TENSOR_DESC* bufferTensorDesc =
            reinterpret_cast<const DML_BUFFER_TENSOR_DESC*>(tensorDESC->Desc);
        return CreateDmlTensorDesc(dmlTensorDesc, dimensions, strides, bufferTensorDesc->DataType,
                                   bufferTensorDesc->Flags);
    }

    std::shared_ptr<EdgeInfoBase> CreateEdgeFromThisNode(const DML_TENSOR_DESC& outputTensorDesc,
                                                         const uint32_t nodeIndex,
                                                         const uint32_t outputNodeIndex = 0) {
        std::shared_ptr<EdgeInfo> edgeInfo(new EdgeInfo());
        edgeInfo->outputTensorDESC = outputTensorDesc;
        edgeInfo->nodeIndex = nodeIndex;
        edgeInfo->outputNodeIndex = outputNodeIndex;
        std::shared_ptr<EdgeInfoBase> edge(edgeInfo);
        return edge;
    }

    // Add an intermediate node to the graph, and add the related input edges(if exist) and
    // intermediate edges to this node by the way.
    MaybeError Graph::AddEdgesToThisNode(std::vector<std::shared_ptr<EdgeInfoBase>> edges) {
        std::unique_ptr<DML_OPERATOR_GRAPH_NODE_DESC> nodeDesc(new DML_OPERATOR_GRAPH_NODE_DESC);
        nodeDesc->Operator = mIntermediateNodesMap[mIntermediateNodes.size()].Get();

        for (size_t i = 0; i < edges.size(); ++i) {
            if (edges[i]->isInputEdge) {
                auto edge = reinterpret_cast<InputEdgeInfo*>(edges[i].get());
                std::unique_ptr<DML_INPUT_GRAPH_EDGE_DESC> inputEdgeDesc(
                    new DML_INPUT_GRAPH_EDGE_DESC);
                inputEdgeDesc->GraphInputIndex = edge->inputIndex;
                inputEdgeDesc->ToNodeIndex = mIntermediateNodes.size();
                inputEdgeDesc->ToNodeInputIndex = i;
                mInputEdges.push_back({DML_GRAPH_EDGE_TYPE_INPUT, inputEdgeDesc.get()});
                mInputEdgesDesc.push_back(std::move(inputEdgeDesc));
            } else {
                auto edge = reinterpret_cast<EdgeInfo*>(edges[i].get());
                std::unique_ptr<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdgeDesc(
                    new DML_INTERMEDIATE_GRAPH_EDGE_DESC);
                intermediateEdgeDesc->FromNodeIndex = edge->nodeIndex;
                intermediateEdgeDesc->FromNodeOutputIndex = edge->outputNodeIndex;
                intermediateEdgeDesc->ToNodeIndex = mIntermediateNodes.size();
                intermediateEdgeDesc->ToNodeInputIndex = i;
                mIntermediateEdges.push_back(
                    {DML_GRAPH_EDGE_TYPE_INTERMEDIATE, intermediateEdgeDesc.get()});
                mIntermediateEdgesDesc.push_back(std::move(intermediateEdgeDesc));
            }
        }
        mIntermediateNodes.push_back({DML_GRAPH_NODE_TYPE_OPERATOR, nodeDesc.get()});
        mIntermediateNodesDesc.push_back(std::move(nodeDesc));
        return {};
    }

    Graph::Graph(Context* context) : GraphBase(context) {
        wnn::DevicePreference devicePreference = GetContext()->GetContextOptions().devicePreference;
        bool useGpu = devicePreference == wnn::DevicePreference::Cpu ? false : true;

        wnn::PowerPreference powerPreference = GetContext()->GetContextOptions().powerPreference;
        DXGI_GPU_PREFERENCE gpuPreference;
        switch (powerPreference) {
            case wnn::PowerPreference::High_performance:
                gpuPreference = DXGI_GPU_PREFERENCE::DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE;
                break;
            case wnn::PowerPreference::Low_power:
                gpuPreference = DXGI_GPU_PREFERENCE::DXGI_GPU_PREFERENCE_MINIMUM_POWER;
                break;
            default:
                gpuPreference = DXGI_GPU_PREFERENCE::DXGI_GPU_PREFERENCE_UNSPECIFIED;
        }
        // Set up Direct3D 12.
        utils::InitD3D12(mCommandList, mCommandQueue, mCommandAllocator, mD3D12Device,
                         gpuPreference, useGpu);

        // Create the DirectML device.
        DML_CREATE_DEVICE_FLAGS dmlCreateDeviceFlags = DML_CREATE_DEVICE_FLAG_NONE;
#if defined(_DEBUG)
        dmlCreateDeviceFlags = DML_CREATE_DEVICE_FLAG_DEBUG;
#endif
        WEBNN_CHECK(
            DMLCreateDevice(mD3D12Device.Get(), dmlCreateDeviceFlags, IID_PPV_ARGS(&mDevice)));
    }

    MaybeError Graph::AddConstant(const op::Constant* constant) {
        const OperandDescriptor* desc = constant->GetOperandDescriptor();
        std::unique_ptr<DmlTensorDesc> dmlTensorDesc(new DmlTensorDesc);
        if (!CreateDmlTensorDesc(dmlTensorDesc, desc, DML_TENSOR_FLAG_OWNED_BY_DML)) {
            return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
        }
        DML_TENSOR_DESC outputTensorDESC = {DML_TENSOR_TYPE_BUFFER, &(dmlTensorDesc->bufferDesc)};

        std::shared_ptr<InputEdgeInfo> inputEdgeInfo(new InputEdgeInfo());
        inputEdgeInfo->outputTensorDESC = outputTensorDESC;
        inputEdgeInfo->name = "Input_Constant_" + std::to_string(mInputs.size());
        inputEdgeInfo->isInputEdge = true;
        inputEdgeInfo->inputIndex = mInputs.size();
        inputEdgeInfo->buffer = constant->GetBuffer();
        inputEdgeInfo->byteLength = constant->GetByteLength();
        inputEdgeInfo->isConstantInput = true;
        std::shared_ptr<EdgeInfoBase> edge(inputEdgeInfo);

        mGraphNodesMap[constant->PrimaryOutput()] = edge;
        mInputs.push_back(*inputEdgeInfo);
        mDmlTensorsDesc.push_back(std::move(dmlTensorDesc));
        return {};
    }

    MaybeError Graph::AddInput(const op::Input* input) {
        const OperandDescriptor* desc = input->GetOperandDescriptor();
        std::unique_ptr<DmlTensorDesc> dmlTensorDesc(new DmlTensorDesc);
        if (!CreateDmlTensorDesc(dmlTensorDesc, desc)) {
            return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
        }
        DML_TENSOR_DESC outputTensorDESC = {DML_TENSOR_TYPE_BUFFER, &(dmlTensorDesc->bufferDesc)};

        std::shared_ptr<InputEdgeInfo> inputEdgeInfo(new InputEdgeInfo());
        inputEdgeInfo->outputTensorDESC = outputTensorDESC;
        inputEdgeInfo->name = input->GetName();
        inputEdgeInfo->isInputEdge = true;
        inputEdgeInfo->inputIndex = mInputs.size();
        std::shared_ptr<EdgeInfoBase> edge(inputEdgeInfo);

        mGraphNodesMap[input->PrimaryOutput()] = edge;
        mInputs.push_back(*inputEdgeInfo);
        mDmlTensorsDesc.push_back(std::move(dmlTensorDesc));
        return {};
    }

    MaybeError Graph::AddBinary(const op::Binary* binary) {
        switch (binary->GetType()) {
            case op::BinaryOpType::kAdd: {
                DAWN_ASSERT(mGraphNodesMap.find(binary->Inputs()[0].Get()) != mGraphNodesMap.end());
                DAWN_ASSERT(mGraphNodesMap.find(binary->Inputs()[1].Get()) != mGraphNodesMap.end());
                auto aEdge = mGraphNodesMap[binary->Inputs()[0].Get()];
                auto bEdge = mGraphNodesMap[binary->Inputs()[1].Get()];

                auto outputDims = binary->Outputs()[0].Get()->Shape();

                // Broadcast inputA
                auto aDims = binary->Inputs()[0].Get()->Shape();
                std::vector<bool> aBroadcast(outputDims.size(), false);
                auto aCount = outputDims.size() - aDims.size();
                for (size_t i = 0; i < aCount; ++i) {
                    aBroadcast[i] = true;
                }
                for (size_t i = 0; i < aDims.size(); ++i) {
                    if (aDims[i] == 1 && outputDims[aCount + i] != 1) {
                        aBroadcast[aCount + i] = true;
                    }
                }
                auto aStrides =
                    CalculateBroadcastStrides(ConvertDimensions(outputDims), aBroadcast);

                std::unique_ptr<DmlTensorDesc> aDmlTensorDesc(new DmlTensorDesc);
                if (!CreateDmlTensorDesc(aDmlTensorDesc, &aEdge->outputTensorDESC,
                                         ConvertDimensions(outputDims), aStrides)) {
                    return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
                }
                DML_TENSOR_DESC aTensorDesc = {DML_TENSOR_TYPE_BUFFER, &aDmlTensorDesc->bufferDesc};

                // Broadcast inputB
                auto bDims = binary->Inputs()[1].Get()->Shape();
                std::vector<bool> bBroadcast(outputDims.size(), false);
                auto bCount = outputDims.size() - bDims.size();
                for (size_t i = 0; i < bCount; ++i) {
                    bBroadcast[i] = true;
                }
                for (size_t i = 0; i < bDims.size(); ++i) {
                    if (bDims[i] == 1 && outputDims[bCount + i] != 1) {
                        bBroadcast[bCount + i] = true;
                    }
                }
                auto bStrides =
                    CalculateBroadcastStrides(ConvertDimensions(outputDims), bBroadcast);

                std::unique_ptr<DmlTensorDesc> bDmlTensorDesc(new DmlTensorDesc);
                if (!CreateDmlTensorDesc(bDmlTensorDesc, &bEdge->outputTensorDESC,
                                         ConvertDimensions(outputDims), bStrides)) {
                    return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
                }
                DML_TENSOR_DESC bTensorDesc = {DML_TENSOR_TYPE_BUFFER, &bDmlTensorDesc->bufferDesc};

                std::unique_ptr<DmlTensorDesc> dmlTensorDesc(new DmlTensorDesc);
                if (!CreateDmlTensorDesc(dmlTensorDesc, ConvertDimensions(outputDims))) {
                    return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
                }
                DML_TENSOR_DESC outputTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                                    &dmlTensorDesc->bufferDesc};

                DML_ELEMENT_WISE_ADD_OPERATOR_DESC dmlAddOperatorDesc{};
                dmlAddOperatorDesc.ATensor = &aTensorDesc;
                dmlAddOperatorDesc.BTensor = &bTensorDesc;
                dmlAddOperatorDesc.OutputTensor = &outputTensorDesc;

                DML_OPERATOR_DESC dmlOperatorDesc = {};
                dmlOperatorDesc.Type = DML_OPERATOR_ELEMENT_WISE_ADD;
                dmlOperatorDesc.Desc = &dmlAddOperatorDesc;

                ComPtr<IDMLOperator> dmlOperator;
                WEBNN_CHECK(mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));
                mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;
                mDmlTensorsDesc.push_back(std::move(dmlTensorDesc));
                mDmlTensorsDesc.push_back(std::move(aDmlTensorDesc));
                mDmlTensorsDesc.push_back(std::move(bDmlTensorDesc));

                mGraphNodesMap[binary->PrimaryOutput()] =
                    CreateEdgeFromThisNode(outputTensorDesc, mIntermediateNodes.size());
                return AddEdgesToThisNode({aEdge, bEdge});
            }
            default:
                return DAWN_UNIMPLEMENTED_ERROR(" Binary op is not implemented.");
        }
    }

    MaybeError Graph::AddUnary(const op::Unary* unary) {
        switch (unary->GetType()) {
            case op::UnaryOpType::kSigmoid: {
                ComPtr<IDMLOperator> dmlOperator;
                DAWN_ASSERT(mGraphNodesMap.find(unary->Inputs()[0].Get()) != mGraphNodesMap.end());
                auto edge = mGraphNodesMap[unary->Inputs()[0].Get()];
                DML_TENSOR_DESC inputTensorDesc = edge->outputTensorDESC;
                DML_ACTIVATION_SIGMOID_OPERATOR_DESC dmlSigmoidOperatorDesc{};
                dmlSigmoidOperatorDesc.InputTensor = &inputTensorDesc;
                dmlSigmoidOperatorDesc.OutputTensor = &inputTensorDesc;

                DML_OPERATOR_DESC dmlOperatorDesc = {};
                dmlOperatorDesc.Type = DML_OPERATOR_ACTIVATION_SIGMOID;
                dmlOperatorDesc.Desc = &dmlSigmoidOperatorDesc;

                WEBNN_CHECK(mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));
                mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;

                mGraphNodesMap[unary->PrimaryOutput()] =
                    CreateEdgeFromThisNode(inputTensorDesc, mIntermediateNodes.size());
                return AddEdgesToThisNode({edge});
                break;
            }
            default:
                return DAWN_UNIMPLEMENTED_ERROR(" Unary op is not implemented.");
        }
        return {};
    }

    MaybeError Graph::AddSplit(const op::Split* split) {
        ComPtr<IDMLOperator> dmlOperator;
        DAWN_ASSERT(split->Inputs().size() == 1);
        OperandBase* inputOperand = split->Inputs()[0].Get();

        auto inputDims = inputOperand->Shape();
        int32_t axis = split->GetAxis();
        // This value must be in the range [0, InputTensor.DimensionCount - 1]. Negative values
        // address dimensions from the end.
        if (axis < 0) {
            axis = axis + inputDims.size();
        }

        size_t outputNum = split->Outputs().size();

        std::vector<DML_TENSOR_DESC> outputTensorsDesc;
        outputTensorsDesc.reserve(outputNum);
        for (size_t i = 0; i < outputNum; ++i) {
            std::unique_ptr<DmlTensorDesc> dmlTensorDesc(new DmlTensorDesc);
            if (!CreateDmlTensorDesc(dmlTensorDesc,
                                     ConvertDimensions(split->Outputs()[i].Get()->Shape()))) {
                return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
            }
            DML_TENSOR_DESC outputTensorDesc = {DML_TENSOR_TYPE_BUFFER, &dmlTensorDesc->bufferDesc};
            outputTensorsDesc.push_back(outputTensorDesc);
            mDmlTensorsDesc.push_back(std::move(dmlTensorDesc));
        }

        DAWN_ASSERT(mGraphNodesMap.find(inputOperand) != mGraphNodesMap.end());
        auto edge = mGraphNodesMap[inputOperand];
        DML_TENSOR_DESC inputTensorDesc = edge->outputTensorDESC;

        DML_SPLIT_OPERATOR_DESC dmlSplitOperatorDesc{};
        dmlSplitOperatorDesc.Axis = axis;
        dmlSplitOperatorDesc.InputTensor = &inputTensorDesc;
        dmlSplitOperatorDesc.OutputCount = outputTensorsDesc.size();
        dmlSplitOperatorDesc.OutputTensors = outputTensorsDesc.data();

        DML_OPERATOR_DESC dmlOperatorDesc = {};
        dmlOperatorDesc.Type = DML_OPERATOR_SPLIT;
        dmlOperatorDesc.Desc = &dmlSplitOperatorDesc;

        WEBNN_CHECK(mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));
        mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;

        for (size_t i = 0; i < outputNum; ++i) {
            mGraphNodesMap[split->Outputs()[i].Get()] =
                CreateEdgeFromThisNode(outputTensorsDesc[i], mIntermediateNodes.size(), i);
        }
        return AddEdgesToThisNode({edge});
    }

    MaybeError Graph::AddOutput(const std::string& name, const OperandBase* output) {
        auto edge = mGraphNodesMap[output];
        if (edge->isInputEdge) {
            return DAWN_INTERNAL_ERROR("Graph for input = output is invalid.");
        }
        edge->name = name;
        std::unique_ptr<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdgeDesc(new DML_OUTPUT_GRAPH_EDGE_DESC);
        auto outputEdgeInfo = reinterpret_cast<EdgeInfo*>(edge.get());
        outputEdgeDesc->FromNodeIndex = outputEdgeInfo->nodeIndex;
        outputEdgeDesc->FromNodeOutputIndex = outputEdgeInfo->outputNodeIndex;
        outputEdgeDesc->GraphOutputIndex = mOutputs.size();
        mOutputEdges.push_back({DML_GRAPH_EDGE_TYPE_OUTPUT, outputEdgeDesc.get()});
        mOutputEdgesDesc.push_back(std::move(outputEdgeDesc));

        mOutputs.push_back(*outputEdgeInfo);
        return {};
    }

    MaybeError Graph::AddBatchNorm(const op::BatchNorm* batchNorm) {
        return DAWN_UNIMPLEMENTED_ERROR("BatchNorm hasn't been supported on DirectML.");
    }

    MaybeError Graph::AddConv2d(const op::Conv2d* conv2d) {
        return DAWN_UNIMPLEMENTED_ERROR("Conv2d hasn't been supported on DirectML.");
    }

    MaybeError Graph::AddConvTranspose2d(const op::ConvTranspose2d* convTranspose2d) {
        return DAWN_UNIMPLEMENTED_ERROR("ConvTranspose2D has not been supported on DirectML.");
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
        if (mInputs.empty()) {
            return DAWN_VALIDATION_ERROR("Model inputs must be set.");
        }
        WEBNN_CHECK(mDevice.Get()->QueryInterface(IID_PPV_ARGS(&mDevice1)));

        // Compiles a graph of DirectML operators into an object that can be dispatched to the GPU.
        DML_GRAPH_DESC graphDesc = {};
        graphDesc.InputCount = static_cast<UINT>(mInputs.size());
        graphDesc.OutputCount = static_cast<UINT>(mOutputs.size());
        graphDesc.NodeCount = static_cast<UINT>(mIntermediateNodes.size());
        graphDesc.Nodes = mIntermediateNodes.data();
        graphDesc.InputEdgeCount = static_cast<UINT>(mInputEdges.size());
        graphDesc.InputEdges = mInputEdges.data();
        graphDesc.OutputEdgeCount = static_cast<UINT>(mOutputEdges.size());
        graphDesc.OutputEdges = mOutputEdges.data();
        graphDesc.IntermediateEdgeCount = static_cast<UINT>(mIntermediateEdges.size());
        graphDesc.IntermediateEdges = mIntermediateEdges.data();

        WEBNN_CHECK(mDevice1->CompileGraph(&graphDesc, DML_EXECUTION_FLAG_NONE,
                                           IID_PPV_ARGS(&mCompiledOperator)));
        return {};
    }

    MaybeError Graph::CompileImpl() {
        IDMLCompiledOperator* compiledOperators[] = {mCompiledOperator.Get()};
        ComPtr<IDMLOperatorInitializer> compiledOperatorInitializer;
        WEBNN_CHECK(mDevice->CreateOperatorInitializer(ARRAYSIZE(compiledOperators),
                                                       compiledOperators,
                                                       IID_PPV_ARGS(&compiledOperatorInitializer)));

        DML_BINDING_PROPERTIES initializeBindingProperties =
            compiledOperatorInitializer->GetBindingProperties();
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
        ID3D12DescriptorHeap* descriptorHeaps[] = {mDescriptorHeap.Get()};
        mCommandList->SetDescriptorHeaps(ARRAYSIZE(descriptorHeaps), descriptorHeaps);

        // Create a binding table over the descriptor heap we just created.
        DML_BINDING_TABLE_DESC bindingTableDesc{};
        bindingTableDesc.Dispatchable = compiledOperatorInitializer.Get();
        bindingTableDesc.CPUDescriptorHandle =
            mDescriptorHeap->GetCPUDescriptorHandleForHeapStart();
        bindingTableDesc.GPUDescriptorHandle =
            mDescriptorHeap->GetGPUDescriptorHandleForHeapStart();
        bindingTableDesc.SizeInDescriptors = descriptorCount;

        WEBNN_CHECK(mDevice->CreateBindingTable(&bindingTableDesc, IID_PPV_ARGS(&mBindingTable)));

        UINT64 temporaryResourceSize = std::max(initializeBindingProperties.TemporaryResourceSize,
                                                executeBindingProperties.TemporaryResourceSize);
        UINT64 persistentResourceSize = executeBindingProperties.PersistentResourceSize;

        // Bind and initialize the operator on the GPU.
        if (temporaryResourceSize != 0) {
            mD3D12Device->CreateCommittedResource(
                &CreateHeapProperties(), D3D12_HEAP_FLAG_NONE,
                &CreateResourceDesc(temporaryResourceSize,
                                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&mTemporaryResource));

            if (initializeBindingProperties.TemporaryResourceSize != 0) {
                DML_BUFFER_BINDING bufferBinding{mTemporaryResource.Get(), 0,
                                                 temporaryResourceSize};
                DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
                mBindingTable->BindTemporaryResource(&bindingDesc);
            }
        }

        if (persistentResourceSize != 0) {
            mD3D12Device->CreateCommittedResource(
                &CreateHeapProperties(), D3D12_HEAP_FLAG_NONE,
                &CreateResourceDesc(persistentResourceSize,
                                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&mPersistentResource));

            // The persistent resource should be bound as the output to the
            // IDMLOperatorInitializer.
            DML_BUFFER_BINDING bufferBinding{mPersistentResource.Get(), 0, persistentResourceSize};
            DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
            mBindingTable->BindOutputs(1, &bindingDesc);
        }

        // Initialize constant inputs.
        uint64_t constantInputsResourceSize = 0;
        for (auto& input : mInputs) {
            if (input.isConstantInput) {
                uint64_t offset = utils::RoundUpToMultiple(
                    constantInputsResourceSize, (uint64_t)DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);
                constantInputsResourceSize = offset + input.byteLength;
            }
        }

        if (constantInputsResourceSize) {
            WEBNN_CHECK(mD3D12Device->CreateCommittedResource(
                &CreateHeapProperties(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE,
                &CreateResourceDesc(constantInputsResourceSize), D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr, IID_PPV_ARGS(&mUploadResource)));

            WEBNN_CHECK(mD3D12Device->CreateCommittedResource(
                &CreateHeapProperties(), D3D12_HEAP_FLAG_NONE,
                &CreateResourceDesc(constantInputsResourceSize,
                                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&mInputResource)));

            D3D12_RANGE constantBufferRange{0, constantInputsResourceSize};
            int8_t* constantBuffer;
            WEBNN_CHECK(mUploadResource->Map(0, &constantBufferRange,
                                             reinterpret_cast<void**>(&constantBuffer)));

            std::vector<DML_BUFFER_BINDING> inputBufferBinding(mInputs.size());
            DML_BUFFER_ARRAY_BINDING inputBufferArrayBinding = {};
            uint64_t offset = 0;
            for (size_t i = 0; i < mInputs.size(); ++i) {
                auto input = mInputs[i];
                if (input.isConstantInput) {
                    uint32_t requiredAlignment = DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT;
                    offset = utils::RoundUpToMultiple(offset, (uint64_t)requiredAlignment);
                    inputBufferBinding[i].Buffer = mInputResource.Get();
                    inputBufferBinding[i].Offset = offset;
                    inputBufferBinding[i].SizeInBytes = input.byteLength;

                    void* dest = constantBuffer + offset;
                    const void* src = input.buffer;
                    memcpy(dest, src, static_cast<size_t>(input.byteLength));
                    offset = offset + input.byteLength;
                }
            }
            inputBufferArrayBinding.BindingCount = inputBufferBinding.size();
            inputBufferArrayBinding.Bindings = inputBufferBinding.data();
            // Copy buffer from mUploadResource to mInputResource.
            CopyBufferRegion(mCommandList, mUploadResource, mInputResource,
                             constantInputsResourceSize, D3D12_RESOURCE_STATE_COPY_DEST);
            DML_BINDING_DESC inputBindingDesc{DML_BINDING_TYPE_BUFFER_ARRAY,
                                              &inputBufferArrayBinding};
            mBindingTable->BindInputs(1, &inputBindingDesc);
        }

        // Record execution of the operator initializer.
        // The command recorder is a stateless object that records Dispatches into an existing
        // Direct3D 12 command list.
        WEBNN_CHECK(mDevice->CreateCommandRecorder(IID_PPV_ARGS(&mCommandRecorder)));
        mCommandRecorder->RecordDispatch(mCommandList.Get(), compiledOperatorInitializer.Get(),
                                         mBindingTable.Get());
        utils::CloseExecuteResetWait(mCommandList, mCommandQueue, mCommandAllocator, mD3D12Device);

        // Bind and execute the operator on the GPU.
        mCommandList->SetDescriptorHeaps(ARRAYSIZE(descriptorHeaps), descriptorHeaps);
        // Reset the binding table to bind for the operator we want to execute (it was
        // previously used to bind for the initializer).
        bindingTableDesc.Dispatchable = mCompiledOperator.Get();
        mBindingTable->Reset(&bindingTableDesc);

        if (temporaryResourceSize != 0) {
            DML_BUFFER_BINDING bufferBinding{mTemporaryResource.Get(), 0, temporaryResourceSize};
            DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
            mBindingTable->BindTemporaryResource(&bindingDesc);
        }

        if (persistentResourceSize != 0) {
            DML_BUFFER_BINDING bufferBinding{mPersistentResource.Get(), 0, persistentResourceSize};
            DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
            mBindingTable->BindPersistentResource(&bindingDesc);
        }
        return {};
    }

    WNNComputeGraphStatus Graph::ComputeImpl(NamedInputsBase* inputs, NamedOutputsBase* outputs) {
        auto namedInputs = inputs->GetRecords();

        // Initialize common inputs.
        uint64_t inputsResourceSize = 0;
        for (auto& input : mInputs) {
            // All the inputs must be set.
            if (!input.isConstantInput && namedInputs.find(input.name) == namedInputs.end()) {
                dawn::ErrorLog() << "The input must be set.";
                return WNNComputeGraphStatus_Error;
            }

            if (!input.isConstantInput) {
                auto& resource = namedInputs[input.name].resource;
                uint64_t offset = utils::RoundUpToMultiple(
                    inputsResourceSize, (uint64_t)DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);
                inputsResourceSize = offset + resource.byteLength;
            }
        }

        if (inputsResourceSize) {
            WEBNN_CHECK(mD3D12Device->CreateCommittedResource(
                &CreateHeapProperties(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE,
                &CreateResourceDesc(inputsResourceSize), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                IID_PPV_ARGS(&mUploadResource)));

            WEBNN_CHECK(mD3D12Device->CreateCommittedResource(
                &CreateHeapProperties(), D3D12_HEAP_FLAG_NONE,
                &CreateResourceDesc(inputsResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&mInputResource)));

            D3D12_RANGE inputBufferRange{0, inputsResourceSize};
            int8_t* inputBuffer;
            WEBNN_CHECK(
                mUploadResource->Map(0, &inputBufferRange, reinterpret_cast<void**>(&inputBuffer)));

            uint64_t offset = 0;
            std::vector<DML_BINDING_DESC> inputBindingDesc(mInputs.size());
            std::vector<DML_BUFFER_BINDING> inputBufferBinding(mInputs.size());
            for (size_t i = 0; i < mInputs.size(); ++i) {
                auto input = mInputs[i];
                if (!input.isConstantInput) {
                    uint32_t requiredAlignment = DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT;
                    offset = utils::RoundUpToMultiple(offset, (uint64_t)requiredAlignment);
                    auto& resource = namedInputs[input.name].resource;
                    inputBufferBinding[i].Buffer = mInputResource.Get();
                    inputBufferBinding[i].Offset = offset;
                    inputBufferBinding[i].SizeInBytes = resource.byteLength;
                    inputBindingDesc[i] = {DML_BINDING_TYPE_BUFFER, &inputBufferBinding[i]};
                    void* dest = inputBuffer + offset;
                    memcpy(dest, static_cast<int8_t*>(resource.buffer) + resource.byteOffset,
                           resource.byteLength);
                    offset = offset + resource.byteLength;
                }
            }
            // Copy buffer from mUploadResource to mInputResource.
            CopyBufferRegion(mCommandList, mUploadResource, mInputResource, inputsResourceSize,
                             D3D12_RESOURCE_STATE_COPY_DEST);
            mBindingTable->BindInputs(inputBindingDesc.size(), inputBindingDesc.data());
        }

        // Prepare for outputs and read back buffer from Gpu.
        auto namedOutputs = outputs->GetRecords();
        std::vector<ArrayBufferView> outputArrayBufferViews;
        uint64_t outputsResourceSize = 0;
        for (size_t i = 0; i < mOutputs.size(); ++i) {
            std::string name = mOutputs[i].name;
            auto namedOutputs = outputs->GetRecords();
            ArrayBufferView output;
            if (namedOutputs.find(name) != namedOutputs.end()) {
                output = namedOutputs[name];
                outputArrayBufferViews.push_back(output);
                DAWN_ASSERT(output.buffer != nullptr && output.byteLength != 0);

                uint64_t offset = utils::RoundUpToMultiple(
                    outputsResourceSize, (uint64_t)DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);
                outputsResourceSize = offset + output.byteLength;
            }
        }

        ComPtr<ID3D12Resource> outputResource;
        if (outputsResourceSize) {
            WEBNN_CHECK(mD3D12Device->CreateCommittedResource(
                &CreateHeapProperties(), D3D12_HEAP_FLAG_NONE,
                &CreateResourceDesc(outputsResourceSize,
                                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&outputResource)));

            std::vector<DML_BINDING_DESC> outputBindingDesc(mOutputs.size());
            std::vector<DML_BUFFER_BINDING> outputBufferBinding(mOutputs.size());

            uint64_t offset = 0;
            for (size_t i = 0; i < mOutputs.size(); ++i) {
                auto output = outputArrayBufferViews[i];
                uint32_t requiredAlignment = DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT;
                offset = utils::RoundUpToMultiple(offset, (uint64_t)requiredAlignment);
                outputBufferBinding[i].Buffer = outputResource.Get();
                outputBufferBinding[i].Offset = offset;
                outputBufferBinding[i].SizeInBytes = output.byteLength;
                outputBindingDesc[i] = {DML_BINDING_TYPE_BUFFER, &outputBufferBinding[i]};
                offset = offset + output.byteLength;
            }
            mBindingTable->BindOutputs(outputBindingDesc.size(), outputBindingDesc.data());
        }
        // Record execution of the compiled operator.
        mCommandRecorder->RecordDispatch(mCommandList.Get(), mCompiledOperator.Get(),
                                         mBindingTable.Get());
        utils::CloseExecuteResetWait(mCommandList, mCommandQueue, mCommandAllocator, mD3D12Device);

        ComPtr<ID3D12Resource> readbackBuffer;
        mD3D12Device->CreateCommittedResource(
            &CreateHeapProperties(D3D12_HEAP_TYPE_READBACK), D3D12_HEAP_FLAG_NONE,
            &CreateResourceDesc(outputsResourceSize), D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_PPV_ARGS(&readbackBuffer));
        CopyBufferRegion(mCommandList, outputResource, readbackBuffer, outputsResourceSize,
                         D3D12_RESOURCE_STATE_COPY_SOURCE);
        utils::CloseExecuteResetWait(mCommandList, mCommandQueue, mCommandAllocator, mD3D12Device);

        D3D12_RANGE tensorBufferRange{0, outputsResourceSize};
        int8_t* outputBuffer;
        WEBNN_CHECK(
            readbackBuffer->Map(0, &tensorBufferRange, reinterpret_cast<void**>(&outputBuffer)));

        uint64_t offset = 0;
        for (size_t i = 0; i < mOutputs.size(); ++i) {
            uint32_t requiredAlignment = DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT;
            offset = utils::RoundUpToMultiple(offset, (uint64_t)requiredAlignment);
            ArrayBufferView output = outputArrayBufferViews[i];
            memcpy(static_cast<int8_t*>(output.buffer) + output.byteOffset, outputBuffer + offset,
                   output.byteLength);
            offset += output.byteLength;
        }

        readbackBuffer->Unmap(0, nullptr);
        return WNNComputeGraphStatus_Success;
    }
}}  // namespace webnn_native::dml
