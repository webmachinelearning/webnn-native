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

    std::vector<bool> GetBroadcastFlags(const std::vector<UINT>& inputShape,
                                        const std::vector<UINT>& outputShape,
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

    std::vector<UINT> ExpandDimensions(const std::vector<UINT>& dims, size_t rank) {
        DAWN_ASSERT(rank >= dims.size());
        std::vector<UINT> newDims(rank, 1);
        for (size_t i = 0; i < dims.size(); ++i) {
            newDims[newDims.size() - i - 1] = dims[dims.size() - i - 1];
        }
        return newDims;
    }

    enum TransposeType { NhwcToNchw, NchwToNhwc };

    std::vector<UINT> transposeStrides(TransposeType transposeType,
                                       const std::vector<UINT>& inputDims) {
        UINT nStride = 0, cStride = 0, hStride = 0, wStride = 0;
        switch (transposeType) {
            case NhwcToNchw:
                nStride = inputDims[1] * inputDims[2] * inputDims[3];
                hStride = inputDims[2] * inputDims[3];
                wStride = inputDims[3];
                cStride = 1;
                return {nStride, cStride, hStride, wStride};
            case NchwToNhwc:
                nStride = inputDims[1] * inputDims[2] * inputDims[3];
                cStride = inputDims[2] * inputDims[3];
                hStride = inputDims[3];
                wStride = 1;
                return {nStride, hStride, wStride, cStride};
            default:
                DAWN_ASSERT(0);
                break;
        }
    }

    std::vector<UINT> transposeDimensions(TransposeType transposeType,
                                          const std::vector<UINT>& inputDims) {
        std::vector<UINT> newInputDims(4);
        switch (transposeType) {
            case NhwcToNchw:
                newInputDims[0] = inputDims[0];
                newInputDims[1] = inputDims[3];
                newInputDims[2] = inputDims[1];
                newInputDims[3] = inputDims[2];
                break;
            case NchwToNhwc:
                newInputDims[0] = inputDims[0];
                newInputDims[1] = inputDims[2];
                newInputDims[2] = inputDims[3];
                newInputDims[3] = inputDims[1];
                break;
            default:
                DAWN_ASSERT(0);
                break;
        }
        return newInputDims;
    }

    std::vector<UINT> transposeFilterDimensionsAsOihw(wnn::Conv2dFilterOperandLayout filterLayout,
                                                      const std::vector<UINT>& filterDims) {
        std::vector<UINT> newFilterDims(4);
        switch (filterLayout) {
            case wnn::Conv2dFilterOperandLayout::Ohwi:
                newFilterDims.resize(4);
                newFilterDims[0] = filterDims[0];
                newFilterDims[1] = filterDims[3];
                newFilterDims[2] = filterDims[1];
                newFilterDims[3] = filterDims[2];
                break;
            case wnn::Conv2dFilterOperandLayout::Hwio:
                newFilterDims[0] = filterDims[3];
                newFilterDims[1] = filterDims[2];
                newFilterDims[2] = filterDims[0];
                newFilterDims[3] = filterDims[1];
                break;
            case wnn::Conv2dFilterOperandLayout::Ihwo:
                newFilterDims[0] = filterDims[3];
                newFilterDims[1] = filterDims[0];
                newFilterDims[2] = filterDims[1];
                newFilterDims[3] = filterDims[2];
                break;
            default:
                DAWN_ASSERT(0);
                break;
        }
        return newFilterDims;
    }

    std::vector<UINT> transposeFilterStridesAsOihw(wnn::Conv2dFilterOperandLayout filterLayout,
                                                   const std::vector<UINT>& filterDims) {
        UINT hStride = 0, wStride = 0, iStride = 0, oStride = 0;
        switch (filterLayout) {
            case wnn::Conv2dFilterOperandLayout::Hwio:
                hStride = filterDims[1] * filterDims[2] * filterDims[3];
                wStride = filterDims[2] * filterDims[3];
                iStride = filterDims[3];
                oStride = 1;
                break;
            case wnn::Conv2dFilterOperandLayout::Ohwi:
                oStride = filterDims[1] * filterDims[2] * filterDims[3];
                hStride = filterDims[2] * filterDims[3];
                wStride = filterDims[3];
                iStride = 1;
                break;
            case wnn::Conv2dFilterOperandLayout::Ihwo:
                iStride = filterDims[1] * filterDims[2] * filterDims[3];
                hStride = filterDims[2] * filterDims[3];
                wStride = filterDims[3];
                oStride = 1;
                break;
            default:
                DAWN_ASSERT(0);
                break;
        }
        return {oStride, iStride, hStride, wStride};
    }

    template <typename T>
    std::vector<UINT> ImplicitPadding(const T* options,
                                      const std::vector<UINT>& inputDims,
                                      const std::vector<UINT>& filterDims) {
        std::vector<UINT> strides, dilations;
        strides.assign(options->strides, options->strides + options->stridesCount);
        dilations.assign(options->dilations, options->dilations + options->dilationsCount);

        // {paddingTop, paddingBottom, paddingLeft, paddingRight}
        int32_t paddingTop, paddingBottom, paddingLeft, paddingRight;
        webnn_native::utils::ComputeImplicitPaddingForAutoPad(
            options->autoPad, dilations[0], inputDims[2], filterDims[filterDims.size() - 2],
            strides[0], paddingTop, paddingBottom);
        webnn_native::utils::ComputeImplicitPaddingForAutoPad(
            options->autoPad, dilations[1], inputDims[3], filterDims[filterDims.size() - 1],
            strides[1], paddingLeft, paddingRight);
        return {static_cast<UINT>(paddingTop), static_cast<UINT>(paddingBottom),
                static_cast<UINT>(paddingLeft), static_cast<UINT>(paddingRight)};
    }

    template <typename T>
    std::vector<UINT> ExplicitPadding(const T* options) {
        UINT paddingTop = static_cast<UINT>(options->padding[0]);
        UINT paddingBottom = static_cast<UINT>(options->padding[1]);
        UINT paddingLeft = static_cast<UINT>(options->padding[2]);
        UINT paddingRight = static_cast<UINT>(options->padding[3]);

        return {paddingTop, paddingBottom, paddingLeft, paddingRight};
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

    bool CreateDmlTensorDesc(std::vector<std::shared_ptr<DmlTensorDesc>>& dmlTensorsDesc,
                             const std::shared_ptr<DmlTensorDesc>& dmlTensorDesc,
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
        auto TotalTensorSizeInBytes = elementsCount * typeLength;
        dmlTensorDesc->bufferDesc.DimensionCount = dmlTensorDesc->dimensions.size();
        dmlTensorDesc->bufferDesc.Sizes = dmlTensorDesc->dimensions.data();
        dmlTensorDesc->bufferDesc.Strides = dmlTensorDesc->strides.data();
        dmlTensorDesc->bufferDesc.TotalTensorSizeInBytes = TotalTensorSizeInBytes;
        dmlTensorDesc->bufferDesc.GuaranteedBaseOffsetAlignment = 0;
        dmlTensorDesc->bufferDesc.DataType = dataType;
        dmlTensorDesc->bufferDesc.Flags = tensorFlag;

        dmlTensorsDesc.push_back(dmlTensorDesc);
        return true;
    }

    bool CreateDmlTensorDesc(std::vector<std::shared_ptr<DmlTensorDesc>>& dmlTensorsDesc,
                             const std::shared_ptr<DmlTensorDesc>& dmlTensorDesc,
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

        return CreateDmlTensorDesc(dmlTensorsDesc, dmlTensorDesc, dimensions, {}, dataType,
                                   tensorFlag);
    }

    bool CreateDmlTensorDesc(std::vector<std::shared_ptr<DmlTensorDesc>>& dmlTensorsDesc,
                             const std::shared_ptr<DmlTensorDesc>& dmlTensorDesc,
                             DML_TENSOR_DESC* tensorDESC,
                             const std::vector<UINT>& dimensions = {},
                             const std::vector<UINT>& strides = {}) {
        DAWN_ASSERT(tensorDESC != nullptr);
        const DML_BUFFER_TENSOR_DESC* bufferTensorDesc =
            reinterpret_cast<const DML_BUFFER_TENSOR_DESC*>(tensorDESC->Desc);
        return CreateDmlTensorDesc(dmlTensorsDesc, dmlTensorDesc, dimensions, strides,
                                   bufferTensorDesc->DataType, bufferTensorDesc->Flags);
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

    std::shared_ptr<EdgeInfoBase> updateEdge(std::shared_ptr<EdgeInfoBase> edge,
                                             const DML_TENSOR_DESC& tensorDesc) {
        if (edge->isInputEdge) {
            std::shared_ptr<InputEdgeInfo> newEdgeInfo(new InputEdgeInfo());
            memcpy(static_cast<void*>(newEdgeInfo.get()), static_cast<void*>(edge.get()),
                   sizeof(InputEdgeInfo));
            newEdgeInfo->outputTensorDESC = tensorDesc;
            std::shared_ptr<EdgeInfoBase> newEdge(newEdgeInfo);
            return newEdge;
        } else {
            std::shared_ptr<EdgeInfo> newEdgeInfo(new EdgeInfo());
            memcpy(static_cast<void*>(newEdgeInfo.get()), static_cast<void*>(edge.get()),
                   sizeof(EdgeInfo));
            newEdgeInfo->outputTensorDESC = tensorDesc;
            std::shared_ptr<EdgeInfoBase> newEdge(newEdgeInfo);
            return newEdge;
        }
    }

    // Add an intermediate node to the graph, and add the related input edges(if exist) and
    // intermediate edges to this node by the way.
    void Graph::AddEdgesToThisNode(std::vector<std::shared_ptr<EdgeInfoBase>> edges) {
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
        std::shared_ptr<DmlTensorDesc> dmlTensorDesc(new DmlTensorDesc);
        if (!CreateDmlTensorDesc(mDmlTensorsDesc, dmlTensorDesc, desc,
                                 DML_TENSOR_FLAG_OWNED_BY_DML)) {
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
        mConstantSet.insert(constant->PrimaryOutput());
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

        std::shared_ptr<DmlTensorDesc> dmlTensorDesc(new DmlTensorDesc);
        if (!CreateDmlTensorDesc(mDmlTensorsDesc, dmlTensorDesc, dmlTensorDims, strides, dataType,
                                 tensorFlag)) {
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
        mConstantsBuffer.push_back(std::move(buffer));
        return {};
    }

    MaybeError Graph::AddInput(const op::Input* input) {
        const OperandDescriptor* desc = input->GetOperandDescriptor();
        std::shared_ptr<DmlTensorDesc> dmlTensorDesc(new DmlTensorDesc);
        if (!CreateDmlTensorDesc(mDmlTensorsDesc, dmlTensorDesc, desc)) {
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
        return {};
    }

    MaybeError Graph::AddBinary(const op::Binary* binary) {
        DAWN_ASSERT(binary->Inputs().size() == 2);
        DAWN_ASSERT(mGraphEdgesMap.find(binary->Inputs()[0].Get()) != mGraphEdgesMap.end());
        DAWN_ASSERT(mGraphEdgesMap.find(binary->Inputs()[1].Get()) != mGraphEdgesMap.end());

        auto aEdge = mGraphEdgesMap[binary->Inputs()[0].Get()];
        auto bEdge = mGraphEdgesMap[binary->Inputs()[1].Get()];
        auto aDims = ConvertDimensions(binary->Inputs()[0].Get()->Shape());
        auto bDims = ConvertDimensions(binary->Inputs()[1].Get()->Shape());
        auto outputDims = ConvertDimensions(binary->Outputs()[0].Get()->Shape());
        size_t aRank = aDims.size(), bRank = bDims.size(), outputRank = outputDims.size();
        size_t broadcastSkipAxis = 0;
        std::vector<UINT> aNewDims, bNewDims, outputNewDims = outputDims;

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
        auto aNewStrides = CalculateStridesForBroadcast(aNewDims, aBroadcastFlags);
        std::shared_ptr<DmlTensorDesc> aDmlTensorDesc(new DmlTensorDesc);
        if (!CreateDmlTensorDesc(mDmlTensorsDesc, aDmlTensorDesc, &aEdge->outputTensorDESC,
                                 aNewDims, aNewStrides)) {
            return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
        }
        DML_TENSOR_DESC aTensorDesc = {DML_TENSOR_TYPE_BUFFER, &aDmlTensorDesc->bufferDesc};

        auto bBroadcastFlags = GetBroadcastFlags(bDims, bNewDims, broadcastSkipAxis);
        auto bNewStrides = CalculateStridesForBroadcast(bNewDims, bBroadcastFlags);
        std::shared_ptr<DmlTensorDesc> bDmlTensorDesc(new DmlTensorDesc);
        if (!CreateDmlTensorDesc(mDmlTensorsDesc, bDmlTensorDesc, &bEdge->outputTensorDESC,
                                 bNewDims, bNewStrides)) {
            return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
        }
        DML_TENSOR_DESC bTensorDesc = {DML_TENSOR_TYPE_BUFFER, &bDmlTensorDesc->bufferDesc};

        std::shared_ptr<DmlTensorDesc> outputDmlTensorDesc(new DmlTensorDesc);
        if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc, outputNewDims)) {
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
            if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc, outputDims)) {
                return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
            }
        }

        mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;
        mGraphEdgesMap[binary->PrimaryOutput()] =
            CreateEdgeFromThisNode(outputTensorDesc, mIntermediateNodes.size());
        AddEdgesToThisNode({aEdge, bEdge});
        return {};
    }

    MaybeError Graph::HardSwish(std::shared_ptr<EdgeInfoBase>& inputEdge,
                                const std::vector<UINT>& inputDims) {
        dawn::WarningLog() << "The hardSwish is emulated from other operations, maybe the "
                              "performance isn't best";
        std::shared_ptr<EdgeInfoBase> intermediateEdge, outputEdge;
        uint32_t length = SizeOfShape(inputDims);
        DML_TENSOR_DESC constantInputTensorDesc, constantSixInputTensorDesc, intermediateTensorDesc,
            inputTensorDesc = inputEdge->outputTensorDESC;
        std::vector<float> constant(length, 3);
        size_t initialInputIndex = mInputs.size() - 1;
        ComPtr<IDMLOperator> dmlOperator;
        // x+3
        {
            // Create the first constant input.
            if (createConstantInput(constantInputTensorDesc, constant.data(),
                                    length * sizeof(float), inputDims, {},
                                    DML_TENSOR_DATA_TYPE_FLOAT32)
                    .IsError()) {
                return DAWN_INTERNAL_ERROR("Failed to create a constant input tensor.");
            };
            // ComPtr<IDMLOperator> dmlOperator;
            CREATE_BINARY_OPERATOR(ADD, inputTensorDesc, constantInputTensorDesc, inputTensorDesc,
                                   dmlOperator);
            mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;
            outputEdge = CreateEdgeFromThisNode(inputTensorDesc, mIntermediateNodes.size());
            AddEdgesToThisNode({inputEdge, mInputs.back()});
        }

        // min(6, (x + 3))
        {
            intermediateTensorDesc = outputEdge->outputTensorDESC;
            intermediateEdge = outputEdge;
            constant = std::vector<float>(length, 6);
            if (createConstantInput(constantSixInputTensorDesc, constant.data(),
                                    length * sizeof(float), inputDims, {},
                                    DML_TENSOR_DATA_TYPE_FLOAT32)
                    .IsError()) {
                return DAWN_INTERNAL_ERROR("Failed to create a constant input tensor.");
            };
            CREATE_BINARY_OPERATOR(MIN, intermediateTensorDesc, constantInputTensorDesc,
                                   intermediateTensorDesc, dmlOperator);
            mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;
            outputEdge = CreateEdgeFromThisNode(intermediateTensorDesc, mIntermediateNodes.size());
            AddEdgesToThisNode({intermediateEdge, mInputs.back()});
        }

        // max(0, min(6, (x + 3)))
        {
            intermediateTensorDesc = outputEdge->outputTensorDESC;
            intermediateEdge = outputEdge;
            constant = std::vector<float>(length, 0);
            // Create the third constant input.
            if (createConstantInput(constantInputTensorDesc, constant.data(),
                                    length * sizeof(float), inputDims, {},
                                    DML_TENSOR_DATA_TYPE_FLOAT32)
                    .IsError()) {
                return DAWN_INTERNAL_ERROR("Failed to create a constant input tensor.");
            };
            CREATE_BINARY_OPERATOR(MAX, intermediateTensorDesc, constantInputTensorDesc,
                                   intermediateTensorDesc, dmlOperator);
            mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;
            outputEdge = CreateEdgeFromThisNode(intermediateTensorDesc, mIntermediateNodes.size());
            AddEdgesToThisNode({intermediateEdge, mInputs.back()});
        }

        // x * max(0, min(6, (x + 3)))
        {
            intermediateTensorDesc = outputEdge->outputTensorDESC;
            intermediateEdge = outputEdge;
            CREATE_BINARY_OPERATOR(MULTIPLY, inputTensorDesc, intermediateTensorDesc,
                                   inputTensorDesc, dmlOperator);
            mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;
            outputEdge = CreateEdgeFromThisNode(inputTensorDesc, mIntermediateNodes.size());
            AddEdgesToThisNode({inputEdge, intermediateEdge});
        }

        // x * max(0, min(6, (x + 3))) / 6
        {
            intermediateTensorDesc = outputEdge->outputTensorDESC;
            intermediateEdge = outputEdge;
            CREATE_BINARY_OPERATOR(DIVIDE, intermediateTensorDesc, constantSixInputTensorDesc,
                                   intermediateTensorDesc, dmlOperator);
            mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;
            inputEdge = CreateEdgeFromThisNode(intermediateTensorDesc, mIntermediateNodes.size());
            // Reuse the second constant input we created above.
            AddEdgesToThisNode({intermediateEdge, mInputs[initialInputIndex + 2]});
            return {};
        }
    }

    MaybeError Graph::AddUnary(const op::Unary* unary) {
        DAWN_ASSERT(unary->Inputs().size() == 1);
        const OperandBase* inputOperand = unary->Inputs()[0].Get();
        DAWN_ASSERT(mGraphEdgesMap.find(inputOperand) != mGraphEdgesMap.end());

        auto inputEdge = mGraphEdgesMap[inputOperand];
        auto inputDims = ConvertDimensions(inputOperand->Shape());
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
                if (HardSwish(inputEdge, inputDims).IsError()) {
                    return DAWN_INTERNAL_ERROR("Failed to create the HardSwish.");
                };
                mGraphEdgesMap[unary->PrimaryOutput()] = inputEdge;
                return {};
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
                uint32_t length = SizeOfShape(inputDims);
                DML_TENSOR_DESC constantInputTensorDesc;
                if (inputOperand->Type() == wnn::OperandType::Float32) {
                    std::vector<float> constant(length, -1);
                    if (createConstantInput(constantInputTensorDesc, constant.data(),
                                            length * sizeof(float), inputDims, {},
                                            DML_TENSOR_DATA_TYPE_FLOAT32)
                            .IsError()) {
                        return DAWN_INTERNAL_ERROR("Failed to create a constant input tensor.");
                    };
                } else if (inputOperand->Type() == wnn::OperandType::Int32) {
                    std::vector<int32_t> constant(length, -1);
                    if (createConstantInput(constantInputTensorDesc, constant.data(),
                                            length * sizeof(int32_t), inputDims, {},
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
        AddEdgesToThisNode(inputEdges);
        return {};
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

        auto edge = mGraphEdgesMap[inputOperand];
        DML_TENSOR_DESC inputTensorDesc = edge->outputTensorDESC;
        std::vector<DML_TENSOR_DESC> outputTensorsDesc;
        outputTensorsDesc.reserve(outputNum);
        for (size_t i = 0; i < outputNum; ++i) {
            std::shared_ptr<DmlTensorDesc> dmlTensorDesc(new DmlTensorDesc);
            if (!CreateDmlTensorDesc(mDmlTensorsDesc, dmlTensorDesc,
                                     ConvertDimensions(split->Outputs()[i].Get()->Shape()))) {
                return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
            }
            DML_TENSOR_DESC outputTensorDesc = {DML_TENSOR_TYPE_BUFFER, &dmlTensorDesc->bufferDesc};
            outputTensorsDesc.push_back(outputTensorDesc);
        }

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
        AddEdgesToThisNode({edge});
        return {};
    }

    MaybeError Graph::AddReshape(const op::Reshape* reshape) {
        DAWN_ASSERT(reshape->Inputs().size() == 1);
        const OperandBase* inputOperand = reshape->Inputs()[0].Get();
        DAWN_ASSERT(mGraphEdgesMap.find(inputOperand) != mGraphEdgesMap.end());

        auto inputEdge = mGraphEdgesMap[inputOperand];
        auto outputDims = ConvertDimensions(reshape->Outputs()[0].Get()->Shape());
        std::shared_ptr<DmlTensorDesc> outputDmlTensorDesc(new DmlTensorDesc);
        // Reshape needn't new strides, because the layout has not been changed.
        if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc, &inputEdge->outputTensorDESC,
                                 outputDims)) {
            return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
        }
        DML_TENSOR_DESC outputTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                            &outputDmlTensorDesc->bufferDesc};
        // Reshape is not a real node in DML, just need to update the edge created from it.
        mGraphEdgesMap[reshape->PrimaryOutput()] = updateEdge(inputEdge, outputTensorDesc);
        return {};
    }

// Use IDENTITY to implement transpose by setting different strides for the input tensor and output
// tensor.
#define TRANSPOSE(inputTensorDesc, outputTensorDesc, dmlOperator)                       \
    DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC dmlSpecificOperatorDesc{};                  \
    dmlSpecificOperatorDesc.InputTensor = &inputTensorDesc;                             \
    dmlSpecificOperatorDesc.OutputTensor = &outputTensorDesc;                           \
    DML_OPERATOR_DESC dmlOperatorDesc = {};                                             \
    dmlOperatorDesc.Type = DML_OPERATOR_ELEMENT_WISE_IDENTITY;                          \
    dmlOperatorDesc.Desc = &dmlSpecificOperatorDesc;                                    \
    WEBNN_CHECK(mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator))); \
    mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;

        MaybeError Graph::AddTranspose(const op::Transpose* transpose) {
            DAWN_ASSERT(transpose->Inputs().size() == 1);
            const OperandBase* inputOperand = transpose->Inputs()[0].Get();
            DAWN_ASSERT(mGraphEdgesMap.find(inputOperand) != mGraphEdgesMap.end());

            auto inputDims = ConvertDimensions(transpose->Inputs()[0].Get()->Shape());
            auto outputDims = ConvertDimensions(transpose->Outputs()[0].Get()->Shape());
            std::vector<int32_t> permutation = transpose->GetPermutation();

            // Transpose need new strides, because the layout has been changed.
            std::vector<UINT> strides(outputDims.size()), transposedStrides;
            uint32_t stride = 1;
            for (size_t i = strides.size(); i-- > 0;) {
                strides[i] = stride;
                stride *= inputDims[i];
            }
            // Permute the strides.
            for (auto dimPermuted : permutation) {
                transposedStrides.push_back(strides[dimPermuted]);
            }

            auto edge = mGraphEdgesMap[inputOperand];
            // Transpose is not a real node in DML, just need to update the edge.
            std::shared_ptr<DmlTensorDesc> inputDmlTensorDesc(new DmlTensorDesc);
            if (!CreateDmlTensorDesc(mDmlTensorsDesc, inputDmlTensorDesc, &edge->outputTensorDESC,
                                     outputDims, transposedStrides)) {
                return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
            }

            std::shared_ptr<DmlTensorDesc> outputDmlTensorDesc(new DmlTensorDesc);
            if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc, outputDims)) {
                return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
            }

            DML_TENSOR_DESC inputTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                               &inputDmlTensorDesc->bufferDesc};

            DML_TENSOR_DESC outputTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                                &outputDmlTensorDesc->bufferDesc};

            ComPtr<IDMLOperator> dmlOperator;
            TRANSPOSE(inputTensorDesc, outputTensorDesc, dmlOperator);

            mGraphEdgesMap[transpose->PrimaryOutput()] =
                CreateEdgeFromThisNode(outputTensorDesc, mIntermediateNodes.size());
            AddEdgesToThisNode({edge});
            return {};
        }

        DML_OPERATOR_DESC* CreateFusedOperator(
            FusionOperatorBase* activation,
            DML_ACTIVATION_LINEAR_OPERATOR_DESC& dmlActicationOperatorDesc,
            DML_OPERATOR_DESC& dmlFusedOperatorDesc) {
            if (activation == nullptr) {
                return nullptr;
            }

            dmlActicationOperatorDesc.InputTensor = nullptr;
            dmlActicationOperatorDesc.OutputTensor = nullptr;
            dmlActicationOperatorDesc.Alpha = 0.0;
            dmlActicationOperatorDesc.Beta = 0.0;
            switch (activation->GetFusionType()) {
                case FusionType::Relu: {
                    dmlFusedOperatorDesc.Type = DML_OPERATOR_ACTIVATION_RELU;
                } break;
                case FusionType::Sigmoid: {
                    dmlFusedOperatorDesc.Type = DML_OPERATOR_ACTIVATION_SIGMOID;
                } break;
                case FusionType::Tanh: {
                    dmlFusedOperatorDesc.Type = DML_OPERATOR_ACTIVATION_TANH;
                } break;
                case FusionType::LeakyRelu: {
                    dmlActicationOperatorDesc.Alpha =
                        reinterpret_cast<op::FusionLeakyRelu*>(activation)->GetAlpha();
                    dmlFusedOperatorDesc.Type = DML_OPERATOR_ACTIVATION_LEAKY_RELU;
                } break;
                case FusionType::Clamp:
                case FusionType::HardSwish:
                    return nullptr;
                default:
                    dawn::ErrorLog() << "This fusion type is not supported.";
                    DAWN_ASSERT(0);
            }
            dmlFusedOperatorDesc.Desc = &dmlActicationOperatorDesc;
            return &dmlFusedOperatorDesc;
        }

        MaybeError Graph::EmulateFusedOperator(FusionOperatorBase* activation,
                                               std::shared_ptr<EdgeInfoBase>& inputEdge,
                                               const std::vector<UINT>& inputDims) {
            // HardSwish and Clamp are not supported for fusion, so we add them directly to
            // emulate. Currently we implement Relu6 operator by Clamp.
            if (activation == nullptr) {
                return {};
            }

            auto fusionType = activation->GetFusionType();
            if (fusionType == FusionType::Clamp) {
                auto clamp = reinterpret_cast<const op::FusionClamp*>(activation);
                inputEdge = Clamp(clamp, inputEdge);
            } else if (fusionType == FusionType::HardSwish) {
                if (HardSwish(inputEdge, inputDims).IsError()) {
                    return DAWN_INTERNAL_ERROR("Failed to create the HardSwish.");
                };
            }
            return {};
        }

        std::shared_ptr<EdgeInfoBase> Graph::Clamp(const op::ClampBase* clamp,
                                                   std::shared_ptr<EdgeInfoBase> inputEdge) {
            DML_TENSOR_DESC inputTensorDesc = inputEdge->outputTensorDESC;

            DML_ELEMENT_WISE_CLIP_OPERATOR_DESC desc = {};
            desc.InputTensor = &inputTensorDesc;
            desc.OutputTensor = &inputTensorDesc;
            desc.ScaleBias = nullptr;
            desc.Min = clamp->GetMinValue();
            desc.Max = clamp->GetMaxValue();
            DML_OPERATOR_DESC dmlOperatorDesc = {};
            dmlOperatorDesc.Type = DML_OPERATOR_ELEMENT_WISE_CLIP;
            dmlOperatorDesc.Desc = &desc;

            ComPtr<IDMLOperator> dmlOperator;
            WEBNN_CHECK(mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));
            mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;

            std::shared_ptr<EdgeInfoBase> outputEdge =
                CreateEdgeFromThisNode(inputTensorDesc, mIntermediateNodes.size());
            AddEdgesToThisNode({inputEdge});
            return outputEdge;
        }

        MaybeError Graph::AddClamp(const op::Clamp* clamp) {
            auto inputsOperand = clamp->Inputs();
            DAWN_ASSERT(inputsOperand.size() == 1);
            auto inputEdge = mGraphEdgesMap[inputsOperand[0].Get()];
            auto inputDims = ConvertDimensions(inputsOperand[0].Get()->Shape());
            mGraphEdgesMap[clamp->PrimaryOutput()] = Clamp(clamp, inputEdge);
            return {};
        }

        MaybeError Graph::AddConv2d(const op::Conv2d* conv2d) {
            auto inputsOperand = conv2d->Inputs();
            DAWN_ASSERT(inputsOperand.size() == 2 || inputsOperand.size() == 3);
            DAWN_ASSERT(mGraphEdgesMap.find(inputsOperand[0].Get()) != mGraphEdgesMap.end());
            DAWN_ASSERT(mGraphEdgesMap.find(inputsOperand[1].Get()) != mGraphEdgesMap.end());

            auto inputEdge = mGraphEdgesMap[inputsOperand[0].Get()];
            auto filterEdge = mGraphEdgesMap[inputsOperand[1].Get()];

            auto inputDims = ConvertDimensions(inputsOperand[0].Get()->Shape());
            auto filterDims = ConvertDimensions(inputsOperand[1].Get()->Shape());
            auto outputDims = ConvertDimensions(conv2d->Outputs()[0].Get()->Shape());
            std::vector<UINT> newInputDims = inputDims, newFilterDims = filterDims,
                              newOutputDims = outputDims, newInputStrides, newFilterStrides,
                              newOutputStrides;

            const Conv2dOptions* options = conv2d->GetOptions();

            DML_TENSOR_DESC inputTensorDesc = inputEdge->outputTensorDESC;
            if (options->inputLayout == wnn::InputOperandLayout::Nhwc) {
                newInputDims = transposeDimensions(NhwcToNchw, inputDims);
                newOutputDims = transposeDimensions(NhwcToNchw, outputDims);
                newInputStrides = transposeStrides(NhwcToNchw, inputDims);

                std::shared_ptr<DmlTensorDesc> inputDmlTensorDesc(new DmlTensorDesc);
                if (!CreateDmlTensorDesc(mDmlTensorsDesc, inputDmlTensorDesc,
                                         &inputEdge->outputTensorDESC, newInputDims,
                                         newInputStrides)) {
                    return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
                }
                inputTensorDesc = {DML_TENSOR_TYPE_BUFFER, &inputDmlTensorDesc->bufferDesc};
            }

            DML_TENSOR_DESC filterTensorDesc = filterEdge->outputTensorDESC;
            if (options->filterLayout != wnn::Conv2dFilterOperandLayout::Oihw) {
                newFilterDims = transposeFilterDimensionsAsOihw(options->filterLayout, filterDims);
                newFilterStrides = transposeFilterStridesAsOihw(options->filterLayout, filterDims);

                std::shared_ptr<DmlTensorDesc> filterDmlTensorDesc(new DmlTensorDesc);
                if (!CreateDmlTensorDesc(mDmlTensorsDesc, filterDmlTensorDesc,
                                         &filterEdge->outputTensorDESC, newFilterDims,
                                         newFilterStrides)) {
                    return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
                }
                filterTensorDesc = {DML_TENSOR_TYPE_BUFFER, &filterDmlTensorDesc->bufferDesc};
            }

            std::vector<std::shared_ptr<EdgeInfoBase>> inputEdges = {inputEdge, filterEdge};

            const DML_TENSOR_DESC* biasTensorDescPtr = nullptr;
            DML_TENSOR_DESC newBiasTensorDesc = {};
            if (options->bias != nullptr) {
                DAWN_ASSERT(mGraphEdgesMap.find(inputsOperand[2].Get()) != mGraphEdgesMap.end());
                auto biasEdge = mGraphEdgesMap[inputsOperand[2].Get()];
                auto biasDims = ConvertDimensions(conv2d->Inputs()[2].Get()->Shape());
                if (biasDims[0] != newFilterDims[0] || biasDims.size() != 1) {
                    return DAWN_INTERNAL_ERROR(
                        "The bias should be 1-D tensor with the shape of [output_channels].");
                }

                // Reshape bias from 1-D to 4-D for NCHW layout.
                std::vector<UINT> newBiasDims = {1, biasDims[0], 1, 1};
                std::shared_ptr<DmlTensorDesc> biasDmlTensorDesc(new DmlTensorDesc);
                if (!CreateDmlTensorDesc(mDmlTensorsDesc, biasDmlTensorDesc,
                                         &biasEdge->outputTensorDESC, newBiasDims)) {
                    return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
                }
                newBiasTensorDesc = {DML_TENSOR_TYPE_BUFFER, &biasDmlTensorDesc->bufferDesc};
                biasTensorDescPtr = &newBiasTensorDesc;
                inputEdges.push_back(biasEdge);
            }

            std::shared_ptr<DmlTensorDesc> outputDmlTensorDesc(new DmlTensorDesc);
            if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc, newOutputDims)) {
                return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
            }
            DML_TENSOR_DESC outputTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                                &outputDmlTensorDesc->bufferDesc};

            // FIXME(nhu): strides, dilations, padding should be uint32_t
            // need to fix the spec.
            std::vector<UINT> strides, dilations;
            strides.assign(options->strides, options->strides + options->stridesCount);
            dilations.assign(options->dilations, options->dilations + options->dilationsCount);

            std::vector<UINT> padding =
                options->autoPad == wnn::AutoPad::Explicit
                    ? ExplicitPadding<Conv2dOptions>(options)
                    : ImplicitPadding<Conv2dOptions>(options, newInputDims, newFilterDims);
            std::vector<UINT> startPadding = {padding[0], padding[2]};
            std::vector<UINT> endPadding = {padding[1], padding[3]};
            std::vector<UINT> defaultOutPadding = {0, 0};

            DML_ACTIVATION_LINEAR_OPERATOR_DESC dmlActicationOperatorDesc{};
            DML_OPERATOR_DESC dmlFusedOperatorDesc = {};
            DML_OPERATOR_DESC* fusedActivation = CreateFusedOperator(
                options->activation, dmlActicationOperatorDesc, dmlFusedOperatorDesc);

            ComPtr<IDMLOperator> dmlOperator;
            DML_CONVOLUTION_OPERATOR_DESC dmlSpecificOperatorDesc{};
            dmlSpecificOperatorDesc.InputTensor = &inputTensorDesc;
            dmlSpecificOperatorDesc.FilterTensor = &filterTensorDesc;
            dmlSpecificOperatorDesc.BiasTensor = biasTensorDescPtr;
            dmlSpecificOperatorDesc.OutputTensor = &outputTensorDesc;

            dmlSpecificOperatorDesc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
            dmlSpecificOperatorDesc.Direction = DML_CONVOLUTION_DIRECTION_FORWARD;
            dmlSpecificOperatorDesc.DimensionCount = inputDims.size() - 2;
            dmlSpecificOperatorDesc.Strides = strides.data();
            dmlSpecificOperatorDesc.Dilations = dilations.data();
            dmlSpecificOperatorDesc.StartPadding = startPadding.data();
            dmlSpecificOperatorDesc.EndPadding = endPadding.data();
            dmlSpecificOperatorDesc.OutputPadding = defaultOutPadding.data();
            dmlSpecificOperatorDesc.GroupCount = static_cast<UINT>(options->groups);
            dmlSpecificOperatorDesc.FusedActivation = fusedActivation;

            DML_OPERATOR_DESC dmlOperatorDesc = {};
            dmlOperatorDesc.Type = DML_OPERATOR_CONVOLUTION;
            dmlOperatorDesc.Desc = &dmlSpecificOperatorDesc;
            WEBNN_CHECK(mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));
            mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;

            // Transpose output from nchw->nhwc.
            std::shared_ptr<EdgeInfoBase> outputEdge;
            if (options->inputLayout == wnn::InputOperandLayout::Nhwc) {
                newOutputStrides = transposeStrides(NchwToNhwc, newOutputDims);
                std::shared_ptr<DmlTensorDesc> nhwcOutputDmlTensorDesc(new DmlTensorDesc);
                if (!CreateDmlTensorDesc(mDmlTensorsDesc, nhwcOutputDmlTensorDesc,
                                         &outputTensorDesc, outputDims, newOutputStrides)) {
                    return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
                }
                DML_TENSOR_DESC nhwcOutputTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                                        &nhwcOutputDmlTensorDesc->bufferDesc};
                outputEdge =
                    CreateEdgeFromThisNode(nhwcOutputTensorDesc, mIntermediateNodes.size());
            } else {
                outputEdge = CreateEdgeFromThisNode(outputTensorDesc, mIntermediateNodes.size());
            }

            AddEdgesToThisNode(inputEdges);
            if (EmulateFusedOperator(options->activation, outputEdge, inputDims).IsError()) {
                return DAWN_INTERNAL_ERROR("Failed to emulate fused operator.");
            };
            mGraphEdgesMap[conv2d->PrimaryOutput()] = outputEdge;
            return {};
        }

        MaybeError Graph::AddPool2d(const op::Pool2d* pool2d) {
            DAWN_ASSERT(pool2d->Inputs().size() == 1);
            const OperandBase* inputOperand = pool2d->Inputs()[0].Get();
            DAWN_ASSERT(mGraphEdgesMap.find(inputOperand) != mGraphEdgesMap.end());

            auto inputEdge = mGraphEdgesMap[inputOperand];
            auto inputDims = ConvertDimensions(inputOperand->Shape());
            auto outputDims = ConvertDimensions(pool2d->Outputs()[0].Get()->Shape());
            std::vector<UINT> newInputDims = inputDims, newOutputDims = outputDims, newInputStrides,
                              newOutputStrides;
            const Pool2dOptions* options = pool2d->GetOptions();

            DML_TENSOR_DESC inputTensorDesc = inputEdge->outputTensorDESC;
            if (options->layout == wnn::InputOperandLayout::Nhwc) {
                newInputDims = transposeDimensions(NhwcToNchw, inputDims);
                newOutputDims = transposeDimensions(NhwcToNchw, outputDims);
                newInputStrides = transposeStrides(NhwcToNchw, inputDims);

                std::shared_ptr<DmlTensorDesc> inputDmlTensorDesc(new DmlTensorDesc);
                if (!CreateDmlTensorDesc(mDmlTensorsDesc, inputDmlTensorDesc,
                                         &inputEdge->outputTensorDESC, newInputDims,
                                         newInputStrides)) {
                    return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
                }
                inputTensorDesc = {DML_TENSOR_TYPE_BUFFER, &inputDmlTensorDesc->bufferDesc};
            }

            std::shared_ptr<DmlTensorDesc> outputDmlTensorDesc(new DmlTensorDesc);
            if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc, newOutputDims)) {
                return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
            }
            DML_TENSOR_DESC outputTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                                &outputDmlTensorDesc->bufferDesc};

            std::vector<UINT> strides, dilations;
            strides.assign(reinterpret_cast<const UINT*>(options->strides),
                           reinterpret_cast<const UINT*>(options->strides) + options->stridesCount);
            dilations.assign(
                reinterpret_cast<const UINT*>(options->dilations),
                reinterpret_cast<const UINT*>(options->dilations) + options->stridesCount);

            std::vector<UINT> windowSizes;
            if (options->windowDimensions != nullptr) {
                const UINT* windowDimensions =
                    reinterpret_cast<const UINT*>(options->windowDimensions);
                windowSizes.assign(windowDimensions,
                                   windowDimensions + options->windowDimensionsCount);
            } else {
                windowSizes = {newInputDims[2], newInputDims[3]};
            }

            auto padding = options->autoPad == wnn::AutoPad::Explicit
                               ? ExplicitPadding<Pool2dOptions>(options)
                               : ImplicitPadding<Pool2dOptions>(options, newInputDims, windowSizes);
            std::vector<UINT> startPadding = {padding[0], padding[2]};
            std::vector<UINT> endPadding = {padding[1], padding[3]};

            ComPtr<IDMLOperator> dmlOperator;
            if (pool2d->GetType() == op::Pool2dType::kAveragePool2d) {
                if (dilations[0] != 1 || dilations[1] != 1) {
                    return DAWN_INTERNAL_ERROR(
                        "The dilations of average pool2d are not supported.");
                }
                DML_AVERAGE_POOLING_OPERATOR_DESC desc = {};
                desc.InputTensor = &inputTensorDesc;
                desc.OutputTensor = &outputTensorDesc;
                desc.DimensionCount = static_cast<UINT>(windowSizes.size());
                desc.Strides = strides.data();
                desc.WindowSize = windowSizes.data();
                desc.StartPadding = startPadding.data();
                desc.EndPadding = endPadding.data();
                desc.IncludePadding = false;
                DML_OPERATOR_DESC dmlOperatorDesc = {};
                dmlOperatorDesc.Type = DML_OPERATOR_AVERAGE_POOLING;
                dmlOperatorDesc.Desc = &desc;
                WEBNN_CHECK(mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));
            }
            // L2Pool2d is not supported, emulate it by referring to
            // https://github.com/tensorflow/tfjs/issues/5539.
            else if (pool2d->GetType() == op::Pool2dType::kL2Pool2d) {
                if (dilations[0] != 1 || dilations[1] != 1) {
                    return DAWN_INTERNAL_ERROR("The dilations of L2 pool2d are not supported.");
                }
                DML_LP_POOLING_OPERATOR_DESC desc = {};
                desc.InputTensor = &inputTensorDesc;
                desc.OutputTensor = &outputTensorDesc;
                desc.DimensionCount = static_cast<UINT>(windowSizes.size());
                desc.Strides = strides.data();
                desc.WindowSize = windowSizes.data();
                desc.StartPadding = startPadding.data();
                desc.EndPadding = endPadding.data();
                desc.P = 2;
                DML_OPERATOR_DESC dmlOperatorDesc = {};
                dmlOperatorDesc.Type = DML_OPERATOR_LP_POOLING;
                dmlOperatorDesc.Desc = &desc;
                WEBNN_CHECK(mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));
            } else if (pool2d->GetType() == op::Pool2dType::kMaxPool2d) {
                if (dilations[0] != 1 || dilations[1] != 1) {
                    for (size_t i = 0; i < windowSizes.size(); ++i) {
                        uint32_t paddedInputSize =
                            newInputDims[2 + i] + startPadding[i] + endPadding[i];
                        uint32_t dilatedWindowSize = 1 + (windowSizes[i] - 1) * dilations[i];
                        newOutputDims[2 + i] =
                            (dilatedWindowSize >= paddedInputSize)
                                ? 1
                                : (paddedInputSize - dilatedWindowSize) / strides[i] + 1;
                    }
                    // Update output tensor.
                    if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc, newOutputDims)) {
                        return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
                    }
                }

                DML_MAX_POOLING2_OPERATOR_DESC desc = {};
                desc.InputTensor = &inputTensorDesc;
                desc.OutputTensor = &outputTensorDesc;
                desc.OutputIndicesTensor = nullptr;
                desc.DimensionCount = static_cast<UINT>(windowSizes.size());
                desc.Strides = strides.data();
                desc.WindowSize = windowSizes.data();
                desc.StartPadding = startPadding.data();
                desc.EndPadding = endPadding.data();
                desc.Dilations = dilations.data();
                DML_OPERATOR_DESC dmlOperatorDesc = {};
                dmlOperatorDesc.Type = DML_OPERATOR_MAX_POOLING2;
                dmlOperatorDesc.Desc = &desc;
                WEBNN_CHECK(mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));
            } else {
                return DAWN_INTERNAL_ERROR("This pool2d type is not supported.");
            }
            mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;

            // Transpose output from nchw->nhwc.
            std::shared_ptr<EdgeInfoBase> outputEdge;
            if (options->layout == wnn::InputOperandLayout::Nhwc) {
                newOutputStrides = transposeStrides(NchwToNhwc, newOutputDims);
                std::shared_ptr<DmlTensorDesc> nhwcOutputDmlTensorDesc(new DmlTensorDesc);
                if (!CreateDmlTensorDesc(mDmlTensorsDesc, nhwcOutputDmlTensorDesc,
                                         &outputTensorDesc, outputDims, newOutputStrides)) {
                    return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
                }
                DML_TENSOR_DESC nhwcOutputTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                                        &nhwcOutputDmlTensorDesc->bufferDesc};
                outputEdge =
                    CreateEdgeFromThisNode(nhwcOutputTensorDesc, mIntermediateNodes.size());
            } else {
                outputEdge = CreateEdgeFromThisNode(outputTensorDesc, mIntermediateNodes.size());
            }

            AddEdgesToThisNode({inputEdge});
            mGraphEdgesMap[pool2d->PrimaryOutput()] = outputEdge;
            return {};
        }

        MaybeError Graph::AddPad(const op::Pad* pad) {
            auto inputsOperand = pad->Inputs();
            DAWN_ASSERT(inputsOperand.size() == 2);
            DAWN_ASSERT(mGraphEdgesMap.find(inputsOperand[0].Get()) != mGraphEdgesMap.end());
            DAWN_ASSERT(mGraphEdgesMap.find(inputsOperand[1].Get()) != mGraphEdgesMap.end());

            auto inputEdge = mGraphEdgesMap[inputsOperand[0].Get()];
            auto paddingEdge = mGraphEdgesMap[inputsOperand[1].Get()];
            auto inputDims = ConvertDimensions(inputsOperand[0].Get()->Shape());
            auto paddingDims = ConvertDimensions(inputsOperand[1].Get()->Shape());
            auto outputDims = ConvertDimensions(pad->Outputs()[0].Get()->Shape());
            size_t inputRank = inputDims.size();

            // Workaround(mingming): If padding was added in mGraph, it must be used.
            // Use "Pad_"+std::to_string(mGraphEdgesMap.size()) to generate a unique name for the
            // output node. This may be a dml issue:
            // https://github.com/microsoft/DirectML/issues/133.
            std::string name = "Pad_" + std::to_string(mGraphEdgesMap.size());
            auto paddingTensorDesc = paddingEdge->outputTensorDESC;
            // Ensure that the DML_TENSOR_FLAGS of output tensor is DML_TENSOR_FLAG_NONE.
            std::shared_ptr<DmlTensorDesc> outputPaddingTensorDesc(new DmlTensorDesc);
            if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputPaddingTensorDesc, &paddingTensorDesc,
                                     paddingDims)) {
                return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
            }
            outputPaddingTensorDesc->bufferDesc.Flags = DML_TENSOR_FLAG_NONE;
            DML_TENSOR_DESC outputTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                                &outputPaddingTensorDesc->bufferDesc};
            ComPtr<IDMLOperator> dmlOperator;
            {
                DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC dmlSpecificOperatorDesc{};
                dmlSpecificOperatorDesc.InputTensor = &paddingTensorDesc;
                dmlSpecificOperatorDesc.OutputTensor = &outputTensorDesc;
                dmlSpecificOperatorDesc.ScaleBias = nullptr;
                DML_OPERATOR_DESC dmlOperatorDesc = {};
                dmlOperatorDesc.Type = DML_OPERATOR_ELEMENT_WISE_IDENTITY;
                dmlOperatorDesc.Desc = &dmlSpecificOperatorDesc;
                WEBNN_CHECK(mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));
            }
            mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;
            auto outputEdge = CreateEdgeFromThisNode(paddingTensorDesc, mIntermediateNodes.size());
            AddEdgesToThisNode({paddingEdge});

            outputEdge->name = name;
            std::unique_ptr<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdgeDesc(
                new DML_OUTPUT_GRAPH_EDGE_DESC);
            auto outputEdgeInfo = reinterpret_cast<EdgeInfo*>(outputEdge.get());
            outputEdgeDesc->FromNodeIndex = outputEdgeInfo->nodeIndex;
            outputEdgeDesc->FromNodeOutputIndex = outputEdgeInfo->outputNodeIndex;
            outputEdgeDesc->GraphOutputIndex = mOutputs.size();
            mOutputEdges.push_back({DML_GRAPH_EDGE_TYPE_OUTPUT, outputEdgeDesc.get()});
            mOutputEdgesDesc.push_back(std::move(outputEdgeDesc));
            mOutputs.push_back(*outputEdgeInfo);

            if (mConstantSet.find(inputsOperand[1].Get()) == mConstantSet.end()) {
                return DAWN_INTERNAL_ERROR("The padding constant is not found.");
            }

            const op::Constant* paddingConstant =
                reinterpret_cast<const op::Constant*>(inputsOperand[1]->Operator());
            const uint32_t* paddingData =
                static_cast<const uint32_t*>(paddingConstant->GetBuffer());
            std::vector<uint32_t> startPadding, endPadding;
            for (size_t i = 0; i < inputRank; ++i) {
                startPadding.push_back(paddingData[2 * i]);
                endPadding.push_back(paddingData[2 * i + 1]);
            }
            const PadOptions* options = pad->GetOptions();
            DML_PADDING_MODE paddingMode;
            switch (options->mode) {
                case wnn::PaddingMode::Edge:
                    paddingMode = DML_PADDING_MODE_EDGE;
                    break;
                case wnn::PaddingMode::Reflection:
                    paddingMode = DML_PADDING_MODE_REFLECTION;
                    break;
                case wnn::PaddingMode::Symmetric:
                    paddingMode = DML_PADDING_MODE_SYMMETRIC;
                    break;
                case wnn::PaddingMode::Constant:
                    paddingMode = DML_PADDING_MODE_CONSTANT;
                    break;
                default:
                    DAWN_ASSERT(0);
            }
            auto inputTensorDesc = inputEdge->outputTensorDESC;
            std::shared_ptr<DmlTensorDesc> outputDmlTensorDesc(new DmlTensorDesc);
            if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc, outputDims)) {
                return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
            }
            outputTensorDesc = {DML_TENSOR_TYPE_BUFFER, &outputDmlTensorDesc->bufferDesc};

            DML_PADDING_OPERATOR_DESC desc = {};
            desc.InputTensor = &inputTensorDesc;
            desc.OutputTensor = &outputTensorDesc;
            desc.PaddingMode = paddingMode;
            desc.PaddingValue = options->value;
            desc.DimensionCount = static_cast<UINT>(startPadding.size());
            desc.StartPadding = startPadding.data();
            desc.EndPadding = endPadding.data();
            DML_OPERATOR_DESC dmlOperatorDesc = {};
            dmlOperatorDesc.Type = DML_OPERATOR_PADDING;
            dmlOperatorDesc.Desc = &desc;

            WEBNN_CHECK(mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));
            mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;

            mGraphEdgesMap[pad->PrimaryOutput()] =
                CreateEdgeFromThisNode(outputTensorDesc, mIntermediateNodes.size());
            AddEdgesToThisNode({inputEdge});
            return {};
        }

        MaybeError Graph::AddBatchNorm(const op::BatchNorm* batchNorm) {
            return DAWN_UNIMPLEMENTED_ERROR("BatchNorm hasn't been supported on DirectML.");
        }

        MaybeError Graph::AddConvTranspose2d(const op::ConvTranspose2d* convTranspose2d) {
            return DAWN_UNIMPLEMENTED_ERROR("ConvTranspose2D has not been supported on DirectML.");
        }

        MaybeError Graph::AddGru(const op::Gru* gru) {
            return DAWN_UNIMPLEMENTED_ERROR("Gru hasn't been supported on DirectML.");
        }

#define CREATE_REDUCE_OPERATOR(type, inputTensorDesc, outputTensorDesc, axes, dmlOperator) \
    DML_REDUCE_OPERATOR_DESC desc = {};                                                    \
    desc.Function = DML_REDUCE_FUNCTION_##type;                                            \
    desc.InputTensor = &inputTensorDesc;                                                   \
    desc.OutputTensor = &outputTensorDesc;                                                 \
    desc.AxisCount = static_cast<UINT>(axes.size());                                       \
    desc.Axes = axes.data();                                                               \
    DML_OPERATOR_DESC dmlOperatorDesc = {};                                                \
    dmlOperatorDesc.Type = DML_OPERATOR_REDUCE;                                            \
    dmlOperatorDesc.Desc = &desc;                                                          \
    WEBNN_CHECK(mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));

        MaybeError Graph::AddReduce(const op::Reduce* reduce) {
            DAWN_ASSERT(reduce->Inputs().size() == 1);
            const OperandBase* inputOperand = reduce->Inputs()[0].Get();
            DAWN_ASSERT(mGraphEdgesMap.find(inputOperand) != mGraphEdgesMap.end());

            auto inputEdge = mGraphEdgesMap[inputOperand];
            const ReduceOptions* options = reduce->GetOptions();
            std::vector<std::uint32_t> axes;
            auto inputDims = ConvertDimensions(inputOperand->Shape());
            auto outputDims = ConvertDimensions(reduce->Outputs()[0].Get()->Shape());

            auto inputTensorDesc = inputEdge->outputTensorDESC;
            auto reducedDims = inputDims;
            for (size_t i = 0; i < options->axesCount; ++i) {
                // Axes values must be in the range [0, InputTensor.DimensionCount - 1].
                // The dimensions to reduce where -1 means the last dimension.
                uint32_t axis = options->axes[i] == -1 ? inputDims.size() - 1 : options->axes[i];
                axes.push_back(axis);
                reducedDims[axis] = 1;
            }
            std::shared_ptr<DmlTensorDesc> outputDmlTensorDesc(new DmlTensorDesc);
            if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc, &inputTensorDesc,
                                     reducedDims)) {
                return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
            }
            DML_TENSOR_DESC outputTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                                &outputDmlTensorDesc->bufferDesc};

            ComPtr<IDMLOperator> dmlOperator;
            switch (reduce->GetType()) {
                case op::ReduceType::kReduceL1: {
                    CREATE_REDUCE_OPERATOR(L1, inputTensorDesc, outputTensorDesc, axes, dmlOperator)
                } break;
                case op::ReduceType::kReduceL2: {
                    CREATE_REDUCE_OPERATOR(L2, inputTensorDesc, outputTensorDesc, axes, dmlOperator)
                } break;
                case op::ReduceType::kReduceMax: {
                    CREATE_REDUCE_OPERATOR(MAX, inputTensorDesc, outputTensorDesc, axes,
                                           dmlOperator)
                } break;
                case op::ReduceType::kReduceMean: {
                    CREATE_REDUCE_OPERATOR(AVERAGE, inputTensorDesc, outputTensorDesc, axes,
                                           dmlOperator)
                } break;
                case op::ReduceType::kReduceMin: {
                    CREATE_REDUCE_OPERATOR(MIN, inputTensorDesc, outputTensorDesc, axes,
                                           dmlOperator)
                } break;
                case op::ReduceType::kReduceProduct: {
                    CREATE_REDUCE_OPERATOR(MULTIPLY, inputTensorDesc, outputTensorDesc, axes,
                                           dmlOperator)
                } break;
                case op::ReduceType::kReduceSum: {
                    CREATE_REDUCE_OPERATOR(SUM, inputTensorDesc, outputTensorDesc, axes,
                                           dmlOperator)
                } break;
                default:
                    return DAWN_INTERNAL_ERROR("The reduce op type isn't supported.");
            }
            mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;
            auto outputEdge = CreateEdgeFromThisNode(outputTensorDesc, mIntermediateNodes.size());
            AddEdgesToThisNode({inputEdge});

            // Reshape if dimensions needn't be kept.
            if (!options->keepDimensions) {
                if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc,
                                         &outputEdge->outputTensorDESC, outputDims)) {
                    return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
                }
                DML_TENSOR_DESC outputTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                                    &outputDmlTensorDesc->bufferDesc};
                // Reshape is not a real node in DML, just need to update the edge created from it.
                outputEdge = updateEdge(outputEdge, outputTensorDesc);
            }

            mGraphEdgesMap[reduce->PrimaryOutput()] = outputEdge;
            return {};
        }

        MaybeError Graph::AddResample2d(const op::Resample2d* resample2d) {
            DAWN_ASSERT(resample2d->Inputs().size() == 1);
            const OperandBase* inputOperand = resample2d->Inputs()[0].Get();
            DAWN_ASSERT(mGraphEdgesMap.find(inputOperand) != mGraphEdgesMap.end());

            auto inputEdge = mGraphEdgesMap[inputOperand];
            auto inputDims = ConvertDimensions(inputOperand->Shape());
            auto outputDims = ConvertDimensions(resample2d->Outputs()[0].Get()->Shape());

            auto inputTensorDesc = inputEdge->outputTensorDESC;
            std::shared_ptr<DmlTensorDesc> outputDmlTensorDesc(new DmlTensorDesc);
            if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc, &inputTensorDesc,
                                     outputDims)) {
                return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
            }
            DML_TENSOR_DESC outputTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                                &outputDmlTensorDesc->bufferDesc};

            const Resample2dOptions* options = resample2d->GetOptions();
            DML_INTERPOLATION_MODE mode;
            switch (options->mode) {
                case wnn::InterpolationMode::NearestNeighbor:
                    mode = DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR;
                    break;
                case wnn::InterpolationMode::Linear:
                    mode = DML_INTERPOLATION_MODE_LINEAR;
                    break;
                default:
                    DAWN_ASSERT(0);
                    break;
            }

            // Scales is computed by dividing the output sizes by the input sizes.
            // InputPixelOffsets = 0.5f for each dimension.
            // OutputPixelOffsets = -0.5f for each dimension.
            std::vector<float> scales;
            for (size_t i = 0; i < inputDims.size(); ++i) {
                scales.push_back(outputDims[i] / inputDims[i]);
            }
            std::vector<float> inputPixelOffsets(4, 0.5), outputPixelOffsets(4, -0.5);

            DML_RESAMPLE1_OPERATOR_DESC desc = {};
            desc.InputTensor = &inputTensorDesc;
            desc.OutputTensor = &outputTensorDesc;
            desc.InterpolationMode = mode;
            desc.DimensionCount = 4;
            desc.Scales = scales.data();
            desc.InputPixelOffsets = inputPixelOffsets.data();
            desc.OutputPixelOffsets = outputPixelOffsets.data();
            DML_OPERATOR_DESC dmlOperatorDesc = {};
            dmlOperatorDesc.Type = DML_OPERATOR_RESAMPLE1;
            dmlOperatorDesc.Desc = &desc;

            ComPtr<IDMLOperator> dmlOperator;
            WEBNN_CHECK(mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));
            mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;

            mGraphEdgesMap[resample2d->PrimaryOutput()] =
                CreateEdgeFromThisNode(outputTensorDesc, mIntermediateNodes.size());
            AddEdgesToThisNode({inputEdge});
            return {};
        }

#define SLICE_ONE_AXIS(axis, index)                                                       \
    inputWindowOffsets[axis] =                                                            \
        starts[index] < 0 ? (starts[index] + inputDims[axis]) : starts[index];            \
    inputWindowSizes[axis] =                                                              \
        sizes[index] == -1 ? (inputDims[axis] - inputWindowOffsets[axis]) : sizes[index]; \
    do {                                                                                  \
    } while (0)

        MaybeError Graph::AddSlice(const op::Slice* slice) {
            DAWN_ASSERT(slice->Inputs().size() == 1);
            const OperandBase* inputOperand = slice->Inputs()[0].Get();
            DAWN_ASSERT(mGraphEdgesMap.find(inputOperand) != mGraphEdgesMap.end());

            auto inputEdge = mGraphEdgesMap[inputOperand];
            auto inputDims = ConvertDimensions(inputOperand->Shape());
            auto outputDims = ConvertDimensions(slice->Outputs()[0].Get()->Shape());

            auto inputTensorDesc = inputEdge->outputTensorDESC;
            std::shared_ptr<DmlTensorDesc> outputDmlTensorDesc(new DmlTensorDesc);
            if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc, &inputTensorDesc,
                                     outputDims)) {
                return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
            }
            DML_TENSOR_DESC outputTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                                &outputDmlTensorDesc->bufferDesc};

            std::vector<uint32_t> inputWindowOffsets(inputDims.size(), 0);
            std::vector<uint32_t> inputWindowSizes(inputDims);
            auto starts = slice->GetStarts();
            auto axes = slice->GetAxes();
            auto sizes = slice->GetSizes();
            if (axes.empty()) {
                for (size_t i = 0; i < inputDims.size(); ++i) {
                    SLICE_ONE_AXIS(i, i);
                }
            } else {
                for (size_t i = 0; i < axes.size(); ++i) {
                    if (axes[i] < 0) {
                        axes[i] = inputDims.size() + axes[i];
                    }
                    SLICE_ONE_AXIS(axes[i], i);
                }
            }
            std::vector<int32_t> inputWindowStrides(inputDims.size(), 1);

            DML_SLICE1_OPERATOR_DESC desc = {};
            desc.InputTensor = &inputTensorDesc;
            desc.OutputTensor = &outputTensorDesc;
            desc.DimensionCount = static_cast<uint32_t>(inputDims.size());
            desc.InputWindowOffsets = inputWindowOffsets.data();
            desc.InputWindowSizes = inputWindowSizes.data();
            desc.InputWindowStrides = inputWindowStrides.data();
            DML_OPERATOR_DESC dmlOperatorDesc = {};
            dmlOperatorDesc.Type = DML_OPERATOR_SLICE1;
            dmlOperatorDesc.Desc = &desc;

            ComPtr<IDMLOperator> dmlOperator;
            WEBNN_CHECK(mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));
            mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;

            mGraphEdgesMap[slice->PrimaryOutput()] =
                CreateEdgeFromThisNode(outputTensorDesc, mIntermediateNodes.size());
            AddEdgesToThisNode({inputEdge});
            return {};
        }

        MaybeError Graph::AddSqueeze(const op::Squeeze* squeeze) {
            DAWN_ASSERT(squeeze->Inputs().size() == 1);
            const OperandBase* inputOperand = squeeze->Inputs()[0].Get();
            DAWN_ASSERT(mGraphEdgesMap.find(inputOperand) != mGraphEdgesMap.end());

            auto inputEdge = mGraphEdgesMap[inputOperand];
            auto outputDims = ConvertDimensions(squeeze->Outputs()[0].Get()->Shape());
            std::shared_ptr<DmlTensorDesc> outputDmlTensorDesc(new DmlTensorDesc);
            // Squeeze perform like reshape which needn't new strides, because the layout has not
            // been changed.
            if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc,
                                     &inputEdge->outputTensorDESC, outputDims)) {
                return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
            }
            DML_TENSOR_DESC outputTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                                &outputDmlTensorDesc->bufferDesc};
            // Squeeze is not a real node in DML, just need to update the edge created from it.
            mGraphEdgesMap[squeeze->PrimaryOutput()] = updateEdge(inputEdge, outputTensorDesc);
            return {};
        }

        MaybeError Graph::AddInstanceNorm(const op::InstanceNorm* instanceNorm) {
            return DAWN_UNIMPLEMENTED_ERROR("InstanceNorm hasn't been supported on DirectML.");
        }

        MaybeError Graph::AddConcat(const op::Concat* concat) {
            DAWN_ASSERT(concat->Inputs().size() >= 1);
            auto inputsOperand = concat->Inputs();
            std::vector<std::shared_ptr<EdgeInfoBase>> inputEdges;
            std::shared_ptr<EdgeInfoBase> primaryEdge = mGraphEdgesMap[inputsOperand[0].Get()];
            auto primaryDims = ConvertDimensions(inputsOperand[0].Get()->Shape());

            std::vector<DML_TENSOR_DESC> inputTensorsDesc;
            for (auto& inputOperand : inputsOperand) {
                DAWN_ASSERT(mGraphEdgesMap.find(inputOperand.Get()) != mGraphEdgesMap.end());
                auto inputEdge = mGraphEdgesMap[inputOperand.Get()];
                auto inputDims = ConvertDimensions(inputOperand.Get()->Shape());
                inputEdges.push_back(inputEdge);

                // Expand dimensions to DML_TENSOR_DIMENSION_COUNT_MAX if needed.
                if (inputDims.size() < DML_TENSOR_DIMENSION_COUNT_MAX) {
                    auto newInputDims = ExpandDimensions(inputDims, DML_TENSOR_DIMENSION_COUNT_MAX);
                    std::shared_ptr<DmlTensorDesc> inputDmlTensorDesc(new DmlTensorDesc);
                    if (!CreateDmlTensorDesc(mDmlTensorsDesc, inputDmlTensorDesc,
                                             &inputEdge->outputTensorDESC, newInputDims)) {
                        return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
                    }
                    inputTensorsDesc.push_back(
                        {DML_TENSOR_TYPE_BUFFER, &inputDmlTensorDesc->bufferDesc});
                } else if (inputDims.size() == DML_TENSOR_DIMENSION_COUNT_MAX) {
                    inputTensorsDesc.push_back(inputEdge->outputTensorDESC);
                } else {
                    return DAWN_INTERNAL_ERROR("The size of input dimensions is greater than max");
                }
            }

            auto outputDims = ConvertDimensions(concat->Outputs()[0].Get()->Shape());
            auto newOutputDims = outputDims;
            if (outputDims.size() < DML_TENSOR_DIMENSION_COUNT_MAX) {
                newOutputDims = ExpandDimensions(outputDims, DML_TENSOR_DIMENSION_COUNT_MAX);
            }

            std::shared_ptr<DmlTensorDesc> outputDmlTensorDesc(new DmlTensorDesc);
            if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc,
                                     &primaryEdge->outputTensorDESC, newOutputDims)) {
                return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
            }
            DML_TENSOR_DESC outputTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                                &outputDmlTensorDesc->bufferDesc};

            // Update the axis to align with the DML_TENSOR_DIMENSION_COUNT_MAX.
            uint32_t axis = concat->GetAxis();
            axis += DML_TENSOR_DIMENSION_COUNT_MAX - primaryDims.size();

            DML_JOIN_OPERATOR_DESC desc = {};
            desc.Axis = axis;
            desc.InputCount = static_cast<uint32_t>(inputTensorsDesc.size());
            desc.InputTensors = inputTensorsDesc.data();
            desc.OutputTensor = &outputTensorDesc;
            DML_OPERATOR_DESC dmlOperatorDesc = {};
            dmlOperatorDesc.Type = DML_OPERATOR_JOIN;
            dmlOperatorDesc.Desc = &desc;

            ComPtr<IDMLOperator> dmlOperator;
            WEBNN_CHECK(mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));
            mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;

            mGraphEdgesMap[concat->PrimaryOutput()] =
                CreateEdgeFromThisNode(outputTensorDesc, mIntermediateNodes.size());
            AddEdgesToThisNode({inputEdges});

            // Reshape back according to output rank if needed to update the output edge.
            if (outputDims.size() < newOutputDims.size()) {
                if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc,
                                         &primaryEdge->outputTensorDESC, outputDims)) {
                    return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
                }
            }
            return {};
        }

        MaybeError Graph::AddGemm(const op::Gemm* gemm) {
            auto inputsOperand = gemm->Inputs();
            DAWN_ASSERT(inputsOperand.size() == 2 || inputsOperand.size() == 3);
            DAWN_ASSERT(mGraphEdgesMap.find(inputsOperand[0].Get()) != mGraphEdgesMap.end());
            auto aEdge = mGraphEdgesMap[inputsOperand[0].Get()];
            auto aDims = ConvertDimensions(inputsOperand[0].Get()->Shape());
            DAWN_ASSERT(mGraphEdgesMap.find(inputsOperand[1].Get()) != mGraphEdgesMap.end());
            auto bEdge = mGraphEdgesMap[inputsOperand[1].Get()];
            auto bDims = ConvertDimensions(inputsOperand[1].Get()->Shape());
            auto outputDims = ConvertDimensions(gemm->Outputs()[0].Get()->Shape());
            std::vector<std::shared_ptr<EdgeInfoBase>> inputEdges = {aEdge, bEdge};

            // The shape of a tensor is 2D definited in WebNN Spec, but DML only support 4D,
            // so expand dimensions to 4D.
            DAWN_ASSERT(aDims.size() == 2);
            aDims = ExpandDimensions(aDims, 4);
            std::shared_ptr<DmlTensorDesc> aDmlTensorDesc(new DmlTensorDesc);
            if (!CreateDmlTensorDesc(mDmlTensorsDesc, aDmlTensorDesc, &aEdge->outputTensorDESC,
                                     aDims)) {
                return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
            }
            DML_TENSOR_DESC aTensorDesc = {DML_TENSOR_TYPE_BUFFER, &aDmlTensorDesc->bufferDesc};

            DAWN_ASSERT(bDims.size() == 2);
            bDims = ExpandDimensions(bDims, 4);
            std::shared_ptr<DmlTensorDesc> bDmlTensorDesc(new DmlTensorDesc);
            if (!CreateDmlTensorDesc(mDmlTensorsDesc, bDmlTensorDesc, &bEdge->outputTensorDESC,
                                     bDims)) {
                return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
            }
            DML_TENSOR_DESC bTensorDesc = {DML_TENSOR_TYPE_BUFFER, &bDmlTensorDesc->bufferDesc};

            DAWN_ASSERT(outputDims.size() == 2);
            auto expandedOutputDims = ExpandDimensions(outputDims, 4);
            std::shared_ptr<DmlTensorDesc> outputDmlTensorDesc(new DmlTensorDesc);
            if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc, &aEdge->outputTensorDESC,
                                     expandedOutputDims)) {
                return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
            }
            DML_TENSOR_DESC outputTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                                &outputDmlTensorDesc->bufferDesc};

            // The operand c is optional.
            DML_TENSOR_DESC* cTensorDescPtr = nullptr;
            DML_TENSOR_DESC cTensorDesc;
            if (inputsOperand.size() == 3) {
                DAWN_ASSERT(mGraphEdgesMap.find(inputsOperand[2].Get()) != mGraphEdgesMap.end());
                auto cEdge = mGraphEdgesMap[inputsOperand[2].Get()];
                auto cDims = ConvertDimensions(inputsOperand[2].Get()->Shape());
                // It is either a scalar, or of the shape that is unidirectionally broadcastable to
                // the shape [M, N] definited in WebNN Spec, DML only support 4D, so expand
                // dimensions to 4D.
                cDims = ExpandDimensions(cDims, 4);
                // BroadCast the Shape of optional C to {1, 1, M, N } supported in DML.
                auto cBroadcastFlags = GetBroadcastFlags(cDims, expandedOutputDims);
                auto cNewStrides = CalculateStridesForBroadcast(cDims, cBroadcastFlags);
                cDims = expandedOutputDims;
                std::shared_ptr<DmlTensorDesc> cDmlTensorDesc(new DmlTensorDesc);
                if (!CreateDmlTensorDesc(mDmlTensorsDesc, cDmlTensorDesc, &cEdge->outputTensorDESC,
                                         cDims, cNewStrides)) {
                    return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
                }
                cTensorDesc = {DML_TENSOR_TYPE_BUFFER, &cDmlTensorDesc->bufferDesc};
                cTensorDescPtr = &cTensorDesc;
                inputEdges.push_back(cEdge);
            }

            const GemmOptions* options = gemm->GetOptions();
            DML_MATRIX_TRANSFORM aTranspose = gemm->GetOptions()->aTranspose
                                                  ? DML_MATRIX_TRANSFORM_TRANSPOSE
                                                  : DML_MATRIX_TRANSFORM_NONE;
            DML_MATRIX_TRANSFORM bTranspose = gemm->GetOptions()->bTranspose
                                                  ? DML_MATRIX_TRANSFORM_TRANSPOSE
                                                  : DML_MATRIX_TRANSFORM_NONE;
            DML_GEMM_OPERATOR_DESC desc{};
            desc.ATensor = &aTensorDesc;
            desc.BTensor = &bTensorDesc;
            desc.CTensor = cTensorDescPtr;
            desc.OutputTensor = &outputTensorDesc;
            desc.TransA = aTranspose;
            desc.TransB = bTranspose;
            desc.Alpha = options->alpha;
            desc.Beta = options->beta;
            DML_OPERATOR_DESC dmlOperatorDesc = {};
            dmlOperatorDesc.Type = DML_OPERATOR_GEMM;
            dmlOperatorDesc.Desc = &desc;

            ComPtr<IDMLOperator> dmlOperator;
            WEBNN_CHECK(mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));
            mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;

            mGraphEdgesMap[gemm->PrimaryOutput()] =
                CreateEdgeFromThisNode(outputTensorDesc, mIntermediateNodes.size());
            AddEdgesToThisNode({inputEdges});

            // Reshape back according to output rank if needed to update the output edge.
            if (outputDims.size() < expandedOutputDims.size()) {
                if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc,
                                         &aEdge->outputTensorDESC, outputDims)) {
                    return DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
                }
            }
            return {};
        }

        MaybeError Graph::AddOutput(const std::string& name, const OperandBase* output) {
            DAWN_ASSERT(mGraphEdgesMap.find(output) != mGraphEdgesMap.end());
            auto outputEdge = mGraphEdgesMap[output];
            DAWN_ASSERT(outputEdge != nullptr);
            if (outputEdge->isInputEdge) {
                // Deal with a graph with a single reshape node.
                // https://github.com/microsoft/DirectML/issues/71
                ComPtr<IDMLOperator> dmlOperator;
                auto edge = outputEdge;
                auto inputTensorDesc = outputEdge->outputTensorDESC;
                CREATE_UNARY_OPERATOR(ACTIVATION_IDENTITY, inputTensorDesc, dmlOperator);
                mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;

                auto outputEdge =
                    CreateEdgeFromThisNode(inputTensorDesc, mIntermediateNodes.size());
                AddEdgesToThisNode({edge});
            }
            outputEdge->name = name;
            std::unique_ptr<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdgeDesc(
                new DML_OUTPUT_GRAPH_EDGE_DESC);
            auto outputEdgeInfo = reinterpret_cast<EdgeInfo*>(outputEdge.get());
            outputEdgeDesc->FromNodeIndex = outputEdgeInfo->nodeIndex;
            outputEdgeDesc->FromNodeOutputIndex = outputEdgeInfo->outputNodeIndex;
            outputEdgeDesc->GraphOutputIndex = mOutputs.size();
            mOutputEdges.push_back({DML_GRAPH_EDGE_TYPE_OUTPUT, outputEdgeDesc.get()});
            mOutputEdgesDesc.push_back(std::move(outputEdgeDesc));

            mOutputs.push_back(*outputEdgeInfo);
            return {};
        }

        MaybeError Graph::Finish() {
            if (mInputs.empty()) {
                return DAWN_VALIDATION_ERROR("Model inputs must be set.");
            }
            WEBNN_CHECK(mDevice.Get()->QueryInterface(IID_PPV_ARGS(&mDevice1)));

            // Compiles a graph of DirectML operators into an object that can be dispatched to the
            // GPU.
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
            WEBNN_CHECK(mUploadResource->Map(0, &uploadBufferRange,
                                             reinterpret_cast<void**>(&uploadBuffer)));
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
            WEBNN_CHECK(
                mDevice->CreateOperatorInitializer(ARRAYSIZE(compiledOperators), compiledOperators,
                                                   IID_PPV_ARGS(&compiledOperatorInitializer)));

            DML_BINDING_PROPERTIES initializeBindingProperties =
                compiledOperatorInitializer->GetBindingProperties();
            DML_BINDING_PROPERTIES executeBindingProperties =
                mCompiledOperator->GetBindingProperties();
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
            // The size of the binding table, in descriptors. This is the maximum number of
            // descriptors that DirectML is permitted to write, from the start of both the supplied
            // CPU and GPU descriptor handles.
            bindingTableDesc.SizeInDescriptors = descriptorCount;

            WEBNN_CHECK(
                mDevice->CreateBindingTable(&bindingTableDesc, IID_PPV_ARGS(&mBindingTable)));

            UINT64 temporaryResourceSize =
                std::max(initializeBindingProperties.TemporaryResourceSize,
                         executeBindingProperties.TemporaryResourceSize);
            UINT64 persistentResourceSize = executeBindingProperties.PersistentResourceSize;

            // Bind and initialize the operator on the GPU.
            if (temporaryResourceSize != 0) {
                mD3D12Device->CreateCommittedResource(
                    &CreateHeapProperties(), D3D12_HEAP_FLAG_NONE,
                    &CreateResourceDesc(temporaryResourceSize,
                                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                    IID_PPV_ARGS(&mTemporaryResource));

                if (initializeBindingProperties.TemporaryResourceSize != 0) {
                    DML_BUFFER_BINDING bufferBinding{mTemporaryResource.Get(), 0,
                                                     temporaryResourceSize};
                    DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
                    mBindingTable->BindTemporaryResource(&bindingDesc);
                }
            }

            // Persistent resources must be supplied during initialization of a compiled operator
            // (where it is bound as an output of the operator initializer) as well as during
            // execution.
            if (persistentResourceSize != 0) {
                mD3D12Device->CreateCommittedResource(
                    &CreateHeapProperties(), D3D12_HEAP_FLAG_NONE,
                    &CreateResourceDesc(persistentResourceSize,
                                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                    IID_PPV_ARGS(&mPersistentResource));

                DML_BUFFER_BINDING bufferBinding{mPersistentResource.Get(), 0,
                                                 persistentResourceSize};
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
            utils::CloseExecuteResetWait(mCommandList, mCommandQueue, mCommandAllocator,
                                         mD3D12Device);

            // Bind and execute the operator on the GPU.
            mCommandList->SetDescriptorHeaps(ARRAYSIZE(descriptorHeaps), descriptorHeaps);
            // Reset the binding table to bind for the operator we want to execute (it was
            // previously used to bind for the initializer).
            bindingTableDesc.Dispatchable = mCompiledOperator.Get();
            mBindingTable->Reset(&bindingTableDesc);

            if (temporaryResourceSize != 0) {
                DML_BUFFER_BINDING bufferBinding{mTemporaryResource.Get(), 0,
                                                 temporaryResourceSize};
                DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
                mBindingTable->BindTemporaryResource(&bindingDesc);
            }

            if (persistentResourceSize != 0) {
                DML_BUFFER_BINDING bufferBinding{mPersistentResource.Get(), 0,
                                                 persistentResourceSize};
                DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
                mBindingTable->BindPersistentResource(&bindingDesc);
            }
            return {};
        }

        WNNComputeGraphStatus Graph::ComputeImpl(NamedInputsBase* inputs,
                                                 NamedOutputsBase* outputs) {
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
                FillUploadResourceAndInputBindings(inputsResourceSize, inputBufferBinding,
                                                   namedInputs);
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
            ArrayBufferView output;
            for (size_t i = 0; i < mOutputs.size(); ++i) {
                std::string name = mOutputs[i].name;
                auto namedOutputs = outputs->GetRecords();
                if (namedOutputs.find(name) != namedOutputs.end()) {
                    output = namedOutputs[name];
                    outputArrayBufferViews.push_back(output);
                    DAWN_ASSERT(output.buffer != nullptr && output.byteLength != 0);
                } else {
                    size_t byteLength = reinterpret_cast<const DML_BUFFER_TENSOR_DESC*>(
                                            mOutputs[i].outputTensorDESC.Desc)
                                            ->TotalTensorSizeInBytes;
                    // It is an unuseful output of dml graph. We need not read back and copy buffer
                    // to it.
                    output = {nullptr, byteLength, 0};
                    outputArrayBufferViews.push_back(output);
                }
                uint64_t offset = utils::RoundUpToMultiple(
                    outputsResourceSize, (uint64_t)DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);
                outputsResourceSize = offset + output.byteLength;
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
            utils::CloseExecuteResetWait(mCommandList, mCommandQueue, mCommandAllocator,
                                         mD3D12Device);

            ComPtr<ID3D12Resource> readBackResource;
            mD3D12Device->CreateCommittedResource(
                &CreateHeapProperties(D3D12_HEAP_TYPE_READBACK), D3D12_HEAP_FLAG_NONE,
                &CreateResourceDesc(outputsResourceSize), D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                IID_PPV_ARGS(&readBackResource));
            // Copy buffer from outputResource to readBackResource.
            CopyBufferRegion(mCommandList, outputResource, readBackResource, outputsResourceSize,
                             D3D12_RESOURCE_STATE_COPY_SOURCE, false);
            utils::CloseExecuteResetWait(mCommandList, mCommandQueue, mCommandAllocator,
                                         mD3D12Device);

            D3D12_RANGE tensorBufferRange{0, outputsResourceSize};
            int8_t* readBackBuffer;
            WEBNN_CHECK(readBackResource->Map(0, &tensorBufferRange,
                                              reinterpret_cast<void**>(&readBackBuffer)));

            uint64_t offset = 0;
            for (size_t i = 0; i < mOutputs.size(); ++i) {
                uint32_t requiredAlignment = DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT;
                offset = utils::RoundUpToMultiple(offset, (uint64_t)requiredAlignment);
                ArrayBufferView output = outputArrayBufferViews[i];
                if (output.buffer) {
                    memcpy(static_cast<int8_t*>(output.buffer) + output.byteOffset,
                           readBackBuffer + offset, output.byteLength);
                }
                offset += output.byteLength;
            }

            readBackResource->Unmap(0, nullptr);
            return WNNComputeGraphStatus_Success;
        }
}}  // namespace webnn_native::dml
