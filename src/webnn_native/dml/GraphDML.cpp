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

#define CREATE_OPERATOR(type, dmlSpecificOperatorDesc) \
    DML_OPERATOR_DESC dmlOperatorDesc = {};            \
    dmlOperatorDesc.Type = DML_OPERATOR_##type;        \
    dmlOperatorDesc.Desc = &dmlSpecificOperatorDesc;   \
    WEBNN_CHECK(mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));

#define CREATE_BINARY_OPERATOR(type, aTensorDesc, bTensorDesc, outputTensorDesc, dmlOperator) \
    DML_ELEMENT_WISE_##type##_OPERATOR_DESC dmlSpecificOperatorDesc{};                        \
    dmlSpecificOperatorDesc.ATensor = &aTensorDesc;                                           \
    dmlSpecificOperatorDesc.BTensor = &bTensorDesc;                                           \
    dmlSpecificOperatorDesc.OutputTensor = &outputTensorDesc;                                 \
    CREATE_OPERATOR(ELEMENT_WISE_##type, dmlSpecificOperatorDesc)

#define CREATE_UNARY_OPERATOR(type, inputTensorDesc, dmlOperator) \
    DML_##type##_OPERATOR_DESC dmlSpecificOperatorDesc{};         \
    dmlSpecificOperatorDesc.InputTensor = &inputTensorDesc;       \
    dmlSpecificOperatorDesc.OutputTensor = &inputTensorDesc;      \
    CREATE_OPERATOR(type, dmlSpecificOperatorDesc)

    void CopyBufferRegion(ComPtr<ID3D12GraphicsCommandList> commandList,
                          ComPtr<ID3D12Resource> srcResource,
                          ComPtr<ID3D12Resource> destResource,
                          UINT64 resourceSize,
                          D3D12_RESOURCE_STATES state,
                          bool needBarrierEnd = true) {
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
        if (needBarrierEnd) {
            resourceBarrier.Transition.StateBefore = state;
            resourceBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            commandList->ResourceBarrier(1, &resourceBarrier);
        }
    }

    std::vector<bool> GetBroadcastFlags(const std::vector<int32_t>& inputShape,
                                        const std::vector<int32_t>& outputShape,
                                        size_t skipAxes = 0) {
        if (inputShape.size() < skipAxes || inputShape.size() > outputShape.size()) {
            dawn::ErrorLog() << "Shapes are incompatible, broadcasting failed.";
            DAWN_ASSERT(0);
        }
        std::vector<bool> BroadcastFlags(outputShape.size(), false);
        auto aCount = outputShape.size() - inputShape.size();
        for (size_t i = 0; i < aCount; ++i) {
            BroadcastFlags[i] = true;
        }
        for (size_t i = 0; i < inputShape.size() - skipAxes; ++i) {
            if (inputShape[i] == 1 && outputShape[aCount + i] != 1) {
                BroadcastFlags[aCount + i] = true;
            }
        }
        return BroadcastFlags;
    }

    // Strides are used to express broadcasting (by specifying a stride of 0) as well as
    // padding. If Strides is not specified, each dimension in the tensor is considered to
    // be contiguously packed, with no additional padding. The calculated strides refer to
    // https://docs.microsoft.com/en-us/windows/win32/direct3d12/dml-helper-functions#calculatestrides
    std::vector<UINT> CalculateStridesForBroadcast(std::vector<UINT> dims,
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

    std::vector<UINT> CalculateStridesForReshape(const std::vector<UINT>& targetDims) {
        size_t rank = targetDims.size();
        std::vector<UINT> strides(rank);
        size_t elements = 1;
        for (size_t i = 1; i < rank; i++) {
            size_t j = targetDims.size() - i - 1;
            elements *= targetDims[j + 1];
            strides[j] = elements;
        }
        return strides;
    }

    uint32_t SizeOfShape(const std::vector<UINT>& dims) {
        uint32_t prod = 1;
        for (size_t i = 0; i < dims.size(); ++i)
            prod *= dims[i];
        return prod;
    }

    std::vector<UINT> ConvertDimensions(const std::vector<int32_t>& dimensions) {
        std::vector<UINT> convertedDimensions;
        for (auto dim : dimensions) {
            if (dim < 0) {
                dawn::ErrorLog() << "DML doesn't support the negative dimension value";
                DAWN_ASSERT(0);
            }
            convertedDimensions.push_back(dim);
        }
        return convertedDimensions;
    }

    std::vector<int32_t> ExpandDimensions(const std::vector<int32_t>& dims, size_t rank) {
        DAWN_ASSERT(rank >= dims.size());
        std::vector<int32_t> newDims(rank, 1);
        for (size_t i = 0; i < dims.size(); ++i) {
            newDims[newDims.size() - i - 1] = dims[dims.size() - i - 1];
        }
        return newDims;
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
                             const std::vector<UINT>& dimensions,
                             const std::vector<UINT>& strides = {},
                             DML_TENSOR_DATA_TYPE dataType = DML_TENSOR_DATA_TYPE_FLOAT32,
                             DML_TENSOR_FLAGS tensorFlag = DML_TENSOR_FLAG_NONE) {
        dmlTensorDesc->dimensions = dimensions;
        dmlTensorDesc->strides = strides;
        if (!strides.empty() && dimensions.size() != strides.size()) {
            dawn::ErrorLog() << "Dimension size should be equal to strides size.";
            return false;
        }

        size_t typeLength = 4;
        switch (dataType) {
            case DML_TENSOR_DATA_TYPE_FLOAT32:
            case DML_TENSOR_DATA_TYPE_INT32:
            case DML_TENSOR_DATA_TYPE_UINT32:
                break;
            case DML_TENSOR_DATA_TYPE_FLOAT16:
                typeLength = 2;
                break;
            default:
                dawn::ErrorLog() << "This data type is not supported";
                return false;
        }

        size_t elementsCount = 1;
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
                auto dim = dmlTensorDesc->dimensions[i];
                if (strides.empty()) {
                    elementsCount *= dim;
                } else {
                    // The specific dim from broadcasting shouldn't increase the count of elements.
                    if (strides[i] == 0) {
                        dim = 1;
                    }
                    elementsCount *= dim;
                }
            }
        }

        dmlTensorDesc->bufferDesc.DimensionCount = dmlTensorDesc->dimensions.size();
        dmlTensorDesc->bufferDesc.Sizes = dmlTensorDesc->dimensions.data();
        dmlTensorDesc->bufferDesc.Strides = dmlTensorDesc->strides.data();
        dmlTensorDesc->bufferDesc.TotalTensorSizeInBytes = elementsCount * typeLength;
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
        for (uint32_t i = 0; i < desc->dimensionsCount; ++i) {
            if (desc->dimensions[i] < 0) {
                dawn::ErrorLog() << "DML doesn't support the negative dimension value";
                return false;
            }
        }
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
            dawn::ErrorLog() << "This data type is not supported";
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
                                                         const uint32_t outputNodeIndex = 0,
                                                         bool isInputEdge = false) {
        std::shared_ptr<EdgeInfo> edgeInfo(new EdgeInfo());
        edgeInfo->outputTensorDESC = outputTensorDesc;
        edgeInfo->nodeIndex = nodeIndex;
        edgeInfo->outputNodeIndex = outputNodeIndex;
        edgeInfo->isInputEdge = isInputEdge;
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

        mGraphEdgesMap[constant->PrimaryOutput()] = edge;
        mInputs.push_back(inputEdgeInfo);
        mDmlTensorsDesc.push_back(std::move(dmlTensorDesc));
        return {};
    }

    MaybeError Graph::createConstantInput(DML_TENSOR_DESC& tensorDESC,
                                          void const* value,
                                          size_t size,
                                          const std::vector<UINT>& dmlTensorDims,
                                          const std::vector<UINT>& strides,
                                          DML_TENSOR_DATA_TYPE dataType,
                                          DML_TENSOR_FLAGS tensorFlag) {
        std::unique_ptr<char> buffer(new char[size]);
        memcpy(buffer.get(), value, size);

        std::unique_ptr<DmlTensorDesc> dmlTensorDesc(new DmlTensorDesc);
        if (!CreateDmlTensorDesc(dmlTensorDesc, dmlTensorDims, strides, dataType, tensorFlag)) {
            return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
        }
        tensorDESC = {DML_TENSOR_TYPE_BUFFER, &(dmlTensorDesc->bufferDesc)};

        std::shared_ptr<InputEdgeInfo> inputEdgeInfo(new InputEdgeInfo());
        inputEdgeInfo->outputTensorDESC = tensorDESC;
        inputEdgeInfo->name = "Input_Constant_" + std::to_string(mInputs.size());
        inputEdgeInfo->isInputEdge = true;
        inputEdgeInfo->inputIndex = mInputs.size();
        inputEdgeInfo->buffer = static_cast<void*>(buffer.get());
        inputEdgeInfo->byteLength = size;
        inputEdgeInfo->isConstantInput = true;

        mInputs.push_back(inputEdgeInfo);
        mDmlTensorsDesc.push_back(std::move(dmlTensorDesc));
        mConstantsBuffer.push_back(std::move(buffer));
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

        mGraphEdgesMap[input->PrimaryOutput()] = edge;
        mInputs.push_back(inputEdgeInfo);
        mDmlTensorsDesc.push_back(std::move(dmlTensorDesc));
        return {};
    }

    MaybeError Graph::AddBinary(const op::Binary* binary) {
        DAWN_ASSERT(binary->Inputs().size() == 2);
        DAWN_ASSERT(mGraphEdgesMap.find(binary->Inputs()[0].Get()) != mGraphEdgesMap.end());
        DAWN_ASSERT(mGraphEdgesMap.find(binary->Inputs()[1].Get()) != mGraphEdgesMap.end());

        auto aEdge = mGraphEdgesMap[binary->Inputs()[0].Get()];
        auto bEdge = mGraphEdgesMap[binary->Inputs()[1].Get()];
        auto aDims = binary->Inputs()[0].Get()->Shape();
        auto bDims = binary->Inputs()[1].Get()->Shape();
        auto outputDims = binary->Outputs()[0].Get()->Shape();
        size_t aRank = aDims.size(), bRank = bDims.size(), outputRank = outputDims.size();
        size_t broadcastSkipAxis = 0;
        std::vector<int32_t> aNewDims, bNewDims, outputNewDims = outputDims;

        if (binary->GetType() == op::BinaryOpType::kMatMul) {
            // DML GEMM requires 4D input tensors.
            if (aRank > 4 || bRank > 4) {
                return DAWN_INTERNAL_ERROR("The size of input dimensions is greater than 4.");
            }
            if (aRank < 4) {
                aDims = ExpandDimensions(aDims, 4);
            }

            if (bRank < 4) {
                if (bRank == 1) {
                    // If b is 1-D, it is converted to a 2-D tensor by by appending a 1 to
                    // its dimensions.
                    bDims.push_back(1);
                }
                bDims = ExpandDimensions(bDims, 4);
            }

            if (outputRank < 4) {
                outputNewDims = ExpandDimensions(outputDims, 4);
            }

            if (aRank > 2 || bRank > 2) {
                // If either a or b is N-D, N > 2, it is treated as a stack of matrices
                // with dimensions corresponding to the last two indices. The matrix
                // multiplication will be broadcasted accordingly by following
                // [numpy-broadcasting-rule].
                broadcastSkipAxis = 2;
            }
            aNewDims = bNewDims = outputNewDims;
            aNewDims[2] = aDims[2];
            aNewDims[3] = aDims[3];
            bNewDims[2] = bDims[2];
            bNewDims[3] = bDims[3];
        } else {
            aNewDims = bNewDims = outputNewDims;
        }

        auto aBroadcastFlags = GetBroadcastFlags(aDims, aNewDims, broadcastSkipAxis);
        auto aNewStrides =
            CalculateStridesForBroadcast(ConvertDimensions(aNewDims), aBroadcastFlags);
        std::unique_ptr<DmlTensorDesc> aDmlTensorDesc(new DmlTensorDesc);
        if (!CreateDmlTensorDesc(aDmlTensorDesc, &aEdge->outputTensorDESC,
                                 ConvertDimensions(aNewDims), aNewStrides)) {
            return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
        }
        DML_TENSOR_DESC aTensorDesc = {DML_TENSOR_TYPE_BUFFER, &aDmlTensorDesc->bufferDesc};

        auto bBroadcastFlags = GetBroadcastFlags(bDims, bNewDims, broadcastSkipAxis);
        auto bNewStrides =
            CalculateStridesForBroadcast(ConvertDimensions(bNewDims), bBroadcastFlags);
        std::unique_ptr<DmlTensorDesc> bDmlTensorDesc(new DmlTensorDesc);
        if (!CreateDmlTensorDesc(bDmlTensorDesc, &bEdge->outputTensorDESC,
                                 ConvertDimensions(bNewDims), bNewStrides)) {
            return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
        }
        DML_TENSOR_DESC bTensorDesc = {DML_TENSOR_TYPE_BUFFER, &bDmlTensorDesc->bufferDesc};

        std::unique_ptr<DmlTensorDesc> outputDmlTensorDesc(new DmlTensorDesc);
        if (!CreateDmlTensorDesc(outputDmlTensorDesc, ConvertDimensions(outputNewDims))) {
            return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
        }
        DML_TENSOR_DESC outputTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                            &outputDmlTensorDesc->bufferDesc};

        ComPtr<IDMLOperator> dmlOperator;
        switch (binary->GetType()) {
            case op::BinaryOpType::kAdd: {
                CREATE_BINARY_OPERATOR(ADD, aTensorDesc, bTensorDesc, outputTensorDesc,
                                       dmlOperator);
            } break;
            case op::BinaryOpType::kDiv: {
                CREATE_BINARY_OPERATOR(DIVIDE, aTensorDesc, bTensorDesc, outputTensorDesc,
                                       dmlOperator);
            } break;
            case op::BinaryOpType::kMul: {
                CREATE_BINARY_OPERATOR(MULTIPLY, aTensorDesc, bTensorDesc, outputTensorDesc,
                                       dmlOperator);
            } break;
            case op::BinaryOpType::kSub: {
                CREATE_BINARY_OPERATOR(SUBTRACT, aTensorDesc, bTensorDesc, outputTensorDesc,
                                       dmlOperator);
            } break;
            case op::BinaryOpType::kMax: {
                CREATE_BINARY_OPERATOR(MAX, aTensorDesc, bTensorDesc, outputTensorDesc,
                                       dmlOperator);
            } break;
            case op::BinaryOpType::kMin: {
                CREATE_BINARY_OPERATOR(MIN, aTensorDesc, bTensorDesc, outputTensorDesc,
                                       dmlOperator);
            } break;
            case op::BinaryOpType::kPower: {
                DML_ELEMENT_WISE_POW_OPERATOR_DESC dmlSpecificOperatorDesc{};
                dmlSpecificOperatorDesc.InputTensor = &aTensorDesc;
                dmlSpecificOperatorDesc.ExponentTensor = &bTensorDesc;
                dmlSpecificOperatorDesc.OutputTensor = &outputTensorDesc;
                DML_OPERATOR_DESC dmlOperatorDesc = {};
                dmlOperatorDesc.Type = DML_OPERATOR_ELEMENT_WISE_POW;
                dmlOperatorDesc.Desc = &dmlSpecificOperatorDesc;
                WEBNN_CHECK(mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));
            } break;
            case op::BinaryOpType::kMatMul: {
                DML_GEMM_OPERATOR_DESC dmlSpecificOperatorDesc{};
                dmlSpecificOperatorDesc.ATensor = &aTensorDesc;
                dmlSpecificOperatorDesc.BTensor = &bTensorDesc;
                dmlSpecificOperatorDesc.OutputTensor = &outputTensorDesc;
                dmlSpecificOperatorDesc.Alpha = 1.0;
                DML_OPERATOR_DESC dmlOperatorDesc = {};
                dmlOperatorDesc.Type = DML_OPERATOR_GEMM;
                dmlOperatorDesc.Desc = &dmlSpecificOperatorDesc;
                WEBNN_CHECK(mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));
            } break;
            default:
                return DAWN_UNIMPLEMENTED_ERROR(" Binary op is not implemented.");
        }
        if (outputDims != outputNewDims) {
            if (!CreateDmlTensorDesc(outputDmlTensorDesc, ConvertDimensions(outputDims))) {
                return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
            }
        }
        mDmlTensorsDesc.push_back(std::move(outputDmlTensorDesc));
        mDmlTensorsDesc.push_back(std::move(aDmlTensorDesc));
        mDmlTensorsDesc.push_back(std::move(bDmlTensorDesc));

        mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;
        mGraphEdgesMap[binary->PrimaryOutput()] =
            CreateEdgeFromThisNode(outputTensorDesc, mIntermediateNodes.size());
        return AddEdgesToThisNode({aEdge, bEdge});
    }

    MaybeError Graph::AddUnary(const op::Unary* unary) {
        DAWN_ASSERT(unary->Inputs().size() == 1);
        const OperandBase* inputOperand = unary->Inputs()[0].Get();
        DAWN_ASSERT(mGraphEdgesMap.find(inputOperand) != mGraphEdgesMap.end());

        auto inputEdge = mGraphEdgesMap[inputOperand];
        auto inputDims = inputOperand->Shape();
        std::vector<std::shared_ptr<EdgeInfoBase>> inputEdges = {inputEdge};
        DML_TENSOR_DESC inputTensorDesc = inputEdge->outputTensorDESC;
        ComPtr<IDMLOperator> dmlOperator;
        switch (unary->GetType()) {
            case op::UnaryOpType::kAbs: {
                CREATE_UNARY_OPERATOR(ELEMENT_WISE_ABS, inputTensorDesc, dmlOperator);
            } break;
            case op::UnaryOpType::kCeil: {
                CREATE_UNARY_OPERATOR(ELEMENT_WISE_CEIL, inputTensorDesc, dmlOperator);
            } break;
            case op::UnaryOpType::kCos: {
                CREATE_UNARY_OPERATOR(ELEMENT_WISE_COS, inputTensorDesc, dmlOperator);
            } break;
            case op::UnaryOpType::kExp: {
                CREATE_UNARY_OPERATOR(ELEMENT_WISE_EXP, inputTensorDesc, dmlOperator);
            } break;
            case op::UnaryOpType::kFloor: {
                CREATE_UNARY_OPERATOR(ELEMENT_WISE_FLOOR, inputTensorDesc, dmlOperator);
            } break;
            case op::UnaryOpType::kHardSwish: {
                dawn::WarningLog() << "The hardSwish is emulated from other operations, maybe the "
                                      "performance isn't best";
                std::shared_ptr<EdgeInfoBase> createdOutputEdge;
                std::shared_ptr<EdgeInfoBase> intermediateEdge;
                auto constantDims = ConvertDimensions(inputDims);
                uint32_t length = SizeOfShape(constantDims);
                DML_TENSOR_DESC constantInputTensorDesc, constantSixInputTensorDesc,
                    intermediateTensorDesc;
                std::vector<float> constant(length, 3);
                size_t initialInputIndex = mInputs.size() - 1;
                // x+3
                {
                    // Create the first constant input.
                    if (createConstantInput(constantInputTensorDesc, constant.data(),
                                            length * sizeof(float), constantDims, {},
                                            DML_TENSOR_DATA_TYPE_FLOAT32)
                            .IsError()) {
                        return DAWN_INTERNAL_ERROR("Failed to create a constant input tensor.");
                    };
                    CREATE_BINARY_OPERATOR(ADD, inputTensorDesc, constantInputTensorDesc,
                                           inputTensorDesc, dmlOperator);
                    mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;
                    createdOutputEdge =
                        CreateEdgeFromThisNode(inputTensorDesc, mIntermediateNodes.size());
                    if (AddEdgesToThisNode({inputEdge, mInputs.back()}).IsError()) {
                        return DAWN_INTERNAL_ERROR("Failed to create input edges for this node.");
                    };
                }

                // min(6, (x + 3))
                {
                    ComPtr<IDMLOperator> dmlOperator;
                    intermediateTensorDesc = createdOutputEdge->outputTensorDESC;
                    intermediateEdge = createdOutputEdge;
                    constant = std::vector<float>(length, 6);
                    // Create the second constant input.
                    if (createConstantInput(constantSixInputTensorDesc, constant.data(),
                                            length * sizeof(float), constantDims, {},
                                            DML_TENSOR_DATA_TYPE_FLOAT32)
                            .IsError()) {
                        return DAWN_INTERNAL_ERROR("Failed to create a constant input tensor.");
                    };
                    CREATE_BINARY_OPERATOR(MIN, intermediateTensorDesc, constantInputTensorDesc,
                                           intermediateTensorDesc, dmlOperator);
                    mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;
                    createdOutputEdge =
                        CreateEdgeFromThisNode(intermediateTensorDesc, mIntermediateNodes.size());
                    if (AddEdgesToThisNode({intermediateEdge, mInputs.back()}).IsError()) {
                        return DAWN_INTERNAL_ERROR("Failed to create input edges for this node.");
                    };
                }

                // max(0, min(6, (x + 3)))
                {
                    ComPtr<IDMLOperator> dmlOperator;
                    intermediateTensorDesc = createdOutputEdge->outputTensorDESC;
                    intermediateEdge = createdOutputEdge;
                    constant = std::vector<float>(length, 0);
                    // Create the third constant input.
                    if (createConstantInput(constantInputTensorDesc, constant.data(),
                                            length * sizeof(float), constantDims, {},
                                            DML_TENSOR_DATA_TYPE_FLOAT32)
                            .IsError()) {
                        return DAWN_INTERNAL_ERROR("Failed to create a constant input tensor.");
                    };
                    CREATE_BINARY_OPERATOR(MAX, intermediateTensorDesc, constantInputTensorDesc,
                                           intermediateTensorDesc, dmlOperator);
                    mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;
                    createdOutputEdge =
                        CreateEdgeFromThisNode(intermediateTensorDesc, mIntermediateNodes.size());
                    if (AddEdgesToThisNode({intermediateEdge, mInputs.back()}).IsError()) {
                        return DAWN_INTERNAL_ERROR("Failed to create input edges for this node.");
                    };
                }

                // x * max(0, min(6, (x + 3)))
                {
                    ComPtr<IDMLOperator> dmlOperator;
                    intermediateTensorDesc = createdOutputEdge->outputTensorDESC;
                    intermediateEdge = createdOutputEdge;
                    CREATE_BINARY_OPERATOR(MULTIPLY, inputTensorDesc, intermediateTensorDesc,
                                           inputTensorDesc, dmlOperator);
                    mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;
                    createdOutputEdge =
                        CreateEdgeFromThisNode(inputTensorDesc, mIntermediateNodes.size());
                    if (AddEdgesToThisNode({inputEdge, intermediateEdge}).IsError()) {
                        return DAWN_INTERNAL_ERROR("Failed to create input edges for this node.");
                    };
                }

                // x * max(0, min(6, (x + 3))) / 6
                {
                    ComPtr<IDMLOperator> dmlOperator;
                    intermediateTensorDesc = createdOutputEdge->outputTensorDESC;
                    intermediateEdge = createdOutputEdge;
                    CREATE_BINARY_OPERATOR(DIVIDE, intermediateTensorDesc,
                                           constantSixInputTensorDesc, intermediateTensorDesc,
                                           dmlOperator);
                    mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;
                    mGraphEdgesMap[unary->PrimaryOutput()] =
                        CreateEdgeFromThisNode(intermediateTensorDesc, mIntermediateNodes.size());
                    // Reuse the second constant input we created above.
                    return AddEdgesToThisNode({intermediateEdge, mInputs[initialInputIndex + 2]});
                }
            } break;
            case op::UnaryOpType::kLog: {
                CREATE_UNARY_OPERATOR(ELEMENT_WISE_LOG, inputTensorDesc, dmlOperator);
            } break;
            case op::UnaryOpType::kLeakyRelu: {
                DML_ACTIVATION_LEAKY_RELU_OPERATOR_DESC dmlSpecificOperatorDesc{};
                dmlSpecificOperatorDesc.InputTensor = &inputTensorDesc;
                dmlSpecificOperatorDesc.OutputTensor = &inputTensorDesc;
                dmlSpecificOperatorDesc.Alpha =
                    reinterpret_cast<const op::LeakyRelu*>(unary)->GetAlpha();
                CREATE_OPERATOR(ACTIVATION_LEAKY_RELU, dmlSpecificOperatorDesc)
            } break;
            // DML doesn't support element-wise negative, emulated it from multiplying input by -1.
            case op::UnaryOpType::kNeg: {
                auto constantDims = ConvertDimensions(inputDims);
                uint32_t length = SizeOfShape(constantDims);
                DML_TENSOR_DESC constantInputTensorDesc;
                if (inputOperand->Type() == wnn::OperandType::Float32) {
                    std::vector<float> constant(length, -1);
                    if (createConstantInput(constantInputTensorDesc, constant.data(),
                                            length * sizeof(float), constantDims, {},
                                            DML_TENSOR_DATA_TYPE_FLOAT32)
                            .IsError()) {
                        return DAWN_INTERNAL_ERROR("Failed to create a constant input tensor.");
                    };
                } else if (inputOperand->Type() == wnn::OperandType::Int32) {
                    std::vector<int32_t> constant(length, -1);
                    if (createConstantInput(constantInputTensorDesc, constant.data(),
                                            length * sizeof(int32_t), constantDims, {},
                                            DML_TENSOR_DATA_TYPE_INT32)
                            .IsError()) {
                        return DAWN_INTERNAL_ERROR("Failed to create a constant input tensor.");
                    };
                } else {
                    return DAWN_UNIMPLEMENTED_ERROR("This data type is not supported for neg.");
                }

                CREATE_BINARY_OPERATOR(MULTIPLY, inputTensorDesc, constantInputTensorDesc,
                                       inputTensorDesc, dmlOperator);
                inputEdges.push_back(mInputs.back());
            } break;
            case op::UnaryOpType::kRelu: {
                CREATE_UNARY_OPERATOR(ACTIVATION_RELU, inputTensorDesc, dmlOperator);
            } break;
            case op::UnaryOpType::kSigmoid: {
                CREATE_UNARY_OPERATOR(ACTIVATION_SIGMOID, inputTensorDesc, dmlOperator);
            } break;
            case op::UnaryOpType::kSin: {
                CREATE_UNARY_OPERATOR(ELEMENT_WISE_SIN, inputTensorDesc, dmlOperator);
            } break;
            case op::UnaryOpType::kSoftmax: {
                CREATE_UNARY_OPERATOR(ACTIVATION_SOFTMAX, inputTensorDesc, dmlOperator);
            } break;
            case op::UnaryOpType::kTan: {
                CREATE_UNARY_OPERATOR(ELEMENT_WISE_TAN, inputTensorDesc, dmlOperator);
            } break;
            case op::UnaryOpType::kTanh: {
                CREATE_UNARY_OPERATOR(ACTIVATION_TANH, inputTensorDesc, dmlOperator);
            } break;
            default:
                return DAWN_UNIMPLEMENTED_ERROR("This Unary op is not implemented.");
        }
        mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;
        mGraphEdgesMap[unary->PrimaryOutput()] =
            CreateEdgeFromThisNode(inputTensorDesc, mIntermediateNodes.size());
        return AddEdgesToThisNode(inputEdges);
    }

    MaybeError Graph::AddSplit(const op::Split* split) {
        DAWN_ASSERT(split->Inputs().size() == 1);
        auto inputOperand = split->Inputs()[0].Get();
        DAWN_ASSERT(mGraphEdgesMap.find(inputOperand) != mGraphEdgesMap.end());

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

        auto edge = mGraphEdgesMap[inputOperand];
        DML_TENSOR_DESC inputTensorDesc = edge->outputTensorDESC;

        DML_SPLIT_OPERATOR_DESC dmlSplitOperatorDesc{};
        dmlSplitOperatorDesc.Axis = axis;
        dmlSplitOperatorDesc.InputTensor = &inputTensorDesc;
        dmlSplitOperatorDesc.OutputCount = outputTensorsDesc.size();
        dmlSplitOperatorDesc.OutputTensors = outputTensorsDesc.data();

        DML_OPERATOR_DESC dmlOperatorDesc = {};
        dmlOperatorDesc.Type = DML_OPERATOR_SPLIT;
        dmlOperatorDesc.Desc = &dmlSplitOperatorDesc;

        ComPtr<IDMLOperator> dmlOperator;
        WEBNN_CHECK(mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));
        mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;

        for (size_t i = 0; i < outputNum; ++i) {
            mGraphEdgesMap[split->Outputs()[i].Get()] =
                CreateEdgeFromThisNode(outputTensorsDesc[i], mIntermediateNodes.size(), i);
        }
        return AddEdgesToThisNode({edge});
    }

    MaybeError Graph::AddReshape(const op::Reshape* reshape) {
        DAWN_ASSERT(reshape->Inputs().size() == 1);
        const OperandBase* inputOperand = reshape->Inputs()[0].Get();
        DAWN_ASSERT(mGraphEdgesMap.find(inputOperand) != mGraphEdgesMap.end());

        auto outputDims = reshape->Outputs()[0].Get()->Shape();
        if (outputDims.size() > DML_TENSOR_DIMENSION_COUNT_MAX) {
            return DAWN_INTERNAL_ERROR("The size of target new shape is not supported by DML.");
        }
        std::unique_ptr<DmlTensorDesc> outputDmlTensorDesc(new DmlTensorDesc);
        // Reshape needn't new strides, because the layout has not been changed.
        if (!CreateDmlTensorDesc(outputDmlTensorDesc, ConvertDimensions(outputDims))) {
            return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
        }
        DML_TENSOR_DESC outputTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                            &outputDmlTensorDesc->bufferDesc};
        // Reshape is not a real node in DML, just need to update the edge.
        mGraphEdgesMap[reshape->PrimaryOutput()] =
            CreateEdgeFromThisNode(outputTensorDesc, mIntermediateNodes.size(), 0,
                                   mGraphEdgesMap[inputOperand]->isInputEdge);
        return {};
    }

    MaybeError Graph::AddOutput(const std::string& name, const OperandBase* output) {
        auto outputEdge = mGraphEdgesMap[output];
        if (outputEdge->isInputEdge) {
            // Deal with a graph with single Reshape node.
            // https://github.com/microsoft/DirectML/issues/71
            ComPtr<IDMLOperator> dmlOperator;
            auto edge = outputEdge;
            auto inputTensorDesc = outputEdge->outputTensorDESC;
            CREATE_UNARY_OPERATOR(ACTIVATION_IDENTITY, inputTensorDesc, dmlOperator);
            mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;
            auto outputEdge = CreateEdgeFromThisNode(inputTensorDesc, mIntermediateNodes.size());
            if (AddEdgesToThisNode({edge}).IsError()) {
                return DAWN_INTERNAL_ERROR("Failed to create input edges for this node.");
            };
        }
        outputEdge->name = name;
        std::unique_ptr<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdgeDesc(new DML_OUTPUT_GRAPH_EDGE_DESC);
        auto outputEdgeInfo = reinterpret_cast<EdgeInfo*>(outputEdge.get());
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

    void Graph::FillUploadResourceAndInputBindings(
        uint64_t uploadResourceSize,
        std::vector<DML_BUFFER_BINDING>& inputBufferBinding,
        std::map<std::string, Input> namedInputs) {
        WEBNN_CHECK(mD3D12Device->CreateCommittedResource(
            &CreateHeapProperties(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE,
            &CreateResourceDesc(uploadResourceSize), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
            IID_PPV_ARGS(&mUploadResource)));

        WEBNN_CHECK(mD3D12Device->CreateCommittedResource(
            &CreateHeapProperties(), D3D12_HEAP_FLAG_NONE,
            &CreateResourceDesc(uploadResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&mInputResource)));

        D3D12_RANGE uploadBufferRange{0, uploadResourceSize};
        int8_t* uploadBuffer;
        WEBNN_CHECK(
            mUploadResource->Map(0, &uploadBufferRange, reinterpret_cast<void**>(&uploadBuffer)));
        uint64_t offset = 0;
        for (size_t i = 0; i < mInputs.size(); ++i) {
            auto input = mInputs[i];
            if (namedInputs.empty()) {
                if (input->isConstantInput) {
                    uint32_t requiredAlignment = DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT;
                    offset = utils::RoundUpToMultiple(offset, (uint64_t)requiredAlignment);
                    inputBufferBinding[i].Buffer = mInputResource.Get();
                    inputBufferBinding[i].Offset = offset;
                    inputBufferBinding[i].SizeInBytes = input->byteLength;
                    memcpy(uploadBuffer + offset, input->buffer,
                           static_cast<size_t>(input->byteLength));
                    offset = offset + input->byteLength;
                }
            } else {
                if (!input->isConstantInput) {
                    uint32_t requiredAlignment = DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT;
                    offset = utils::RoundUpToMultiple(offset, (uint64_t)requiredAlignment);
                    auto resource = namedInputs[input->name].resource;
                    inputBufferBinding[i].Buffer = mInputResource.Get();
                    inputBufferBinding[i].Offset = offset;
                    inputBufferBinding[i].SizeInBytes = resource.byteLength;
                    memcpy(uploadBuffer + offset,
                           static_cast<int8_t*>(resource.buffer) + resource.byteOffset,
                           resource.byteLength);
                    offset = offset + resource.byteLength;
                }
            }
        }
        mUploadResource->Unmap(0, nullptr);
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

        // Describe and create a constant buffer view (CBV), Shader resource view (SRV), and
        // unordered access view (UAV) descriptor heap.
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
        // The size of the binding table, in descriptors. This is the maximum number of descriptors
        // that DirectML is permitted to write, from the start of both the supplied CPU and GPU
        // descriptor handles.
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

        // Persistent resources must be supplied during initialization of a compiled operator (where
        // it is bound as an output of the operator initializer) as well as during execution.
        if (persistentResourceSize != 0) {
            mD3D12Device->CreateCommittedResource(
                &CreateHeapProperties(), D3D12_HEAP_FLAG_NONE,
                &CreateResourceDesc(persistentResourceSize,
                                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&mPersistentResource));

            DML_BUFFER_BINDING bufferBinding{mPersistentResource.Get(), 0, persistentResourceSize};
            DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
            mBindingTable->BindOutputs(1, &bindingDesc);
        }

        // Initialize constant inputs.
        uint64_t constantInputsResourceSize = 0;
        for (auto& input : mInputs) {
            if (input->isConstantInput) {
                uint64_t offset = utils::RoundUpToMultiple(
                    constantInputsResourceSize, (uint64_t)DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);
                constantInputsResourceSize = offset + input->byteLength;
            }
        }

        if (constantInputsResourceSize) {
            std::vector<DML_BUFFER_BINDING> inputBufferBinding(mInputs.size());
            FillUploadResourceAndInputBindings(constantInputsResourceSize, inputBufferBinding);
            // Copy buffer from mUploadResource to mInputResource.
            CopyBufferRegion(mCommandList, mUploadResource, mInputResource,
                             constantInputsResourceSize, D3D12_RESOURCE_STATE_COPY_DEST);

            DML_BUFFER_ARRAY_BINDING inputBufferArrayBinding = {};
            inputBufferArrayBinding.BindingCount = inputBufferBinding.size();
            inputBufferArrayBinding.Bindings = inputBufferBinding.data();
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
            if (!input->isConstantInput && namedInputs.find(input->name) == namedInputs.end()) {
                dawn::ErrorLog() << "The input must be set.";
                return WNNComputeGraphStatus_Error;
            }

            if (!input->isConstantInput) {
                auto& resource = namedInputs[input->name].resource;
                uint64_t offset = utils::RoundUpToMultiple(
                    inputsResourceSize, (uint64_t)DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);
                inputsResourceSize = offset + resource.byteLength;
            }
        }

        if (inputsResourceSize) {
            std::vector<DML_BUFFER_BINDING> inputBufferBinding(mInputs.size());
            FillUploadResourceAndInputBindings(inputsResourceSize, inputBufferBinding, namedInputs);
            // Copy buffer from mUploadResource to mInputResource.
            CopyBufferRegion(mCommandList, mUploadResource, mInputResource, inputsResourceSize,
                             D3D12_RESOURCE_STATE_COPY_DEST);

            std::vector<DML_BINDING_DESC> inputBindingDesc(mInputs.size());
            for (size_t i = 0; i < inputBufferBinding.size(); ++i) {
                if (inputBufferBinding[i].Buffer != nullptr) {
                    inputBindingDesc[i] = {DML_BINDING_TYPE_BUFFER, &inputBufferBinding[i]};
                }
            }
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

        ComPtr<ID3D12Resource> readBackResource;
        mD3D12Device->CreateCommittedResource(
            &CreateHeapProperties(D3D12_HEAP_TYPE_READBACK), D3D12_HEAP_FLAG_NONE,
            &CreateResourceDesc(outputsResourceSize), D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_PPV_ARGS(&readBackResource));
        // Copy buffer from outputResource to readBackResource.
        CopyBufferRegion(mCommandList, outputResource, readBackResource, outputsResourceSize,
                         D3D12_RESOURCE_STATE_COPY_SOURCE, false);
        utils::CloseExecuteResetWait(mCommandList, mCommandQueue, mCommandAllocator, mD3D12Device);

        D3D12_RANGE tensorBufferRange{0, outputsResourceSize};
        int8_t* readBackBuffer;
        WEBNN_CHECK(readBackResource->Map(0, &tensorBufferRange,
                                          reinterpret_cast<void**>(&readBackBuffer)));

        uint64_t offset = 0;
        for (size_t i = 0; i < mOutputs.size(); ++i) {
            uint32_t requiredAlignment = DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT;
            offset = utils::RoundUpToMultiple(offset, (uint64_t)requiredAlignment);
            ArrayBufferView output = outputArrayBufferViews[i];
            memcpy(static_cast<int8_t*>(output.buffer) + output.byteOffset, readBackBuffer + offset,
                   output.byteLength);
            offset += output.byteLength;
        }

        readBackResource->Unmap(0, nullptr);
        return WNNComputeGraphStatus_Success;
    }
}}  // namespace webnn_native::dml
