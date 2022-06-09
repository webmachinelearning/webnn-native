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

#include "webnn/native/dml/GraphDML.h"

#include <algorithm>

#include "webnn/native/NamedInputs.h"
#include "webnn/native/Utils.h"

namespace webnn::native ::dml {

#define CREATE_BINARY_OPERATOR(graphBuilder, type, aTensorDesc, bTensorDesc, outputTensorDesc) \
    do {                                                                                       \
        DML_ELEMENT_WISE_##type##_OPERATOR_DESC operatorDesc{};                                \
        operatorDesc.ATensor = &aTensorDesc;                                                   \
        operatorDesc.BTensor = &bTensorDesc;                                                   \
        operatorDesc.OutputTensor = &outputTensorDesc;                                         \
        graphBuilder->CreateOperator(DML_OPERATOR_ELEMENT_WISE_##type, &operatorDesc);         \
    } while (0)

#define CREATE_UNARY_OPERATOR(graphBuilder, type, inputTensorDesc)        \
    do {                                                                  \
        DML_##type##_OPERATOR_DESC operatorDesc{};                        \
        operatorDesc.InputTensor = &inputTensorDesc;                      \
        operatorDesc.OutputTensor = &inputTensorDesc;                     \
        graphBuilder->CreateOperator(DML_OPERATOR_##type, &operatorDesc); \
    } while (0)

#define CREATE_REDUCE_OPERATOR(graphBuilder, type, inputTensorDesc, outputTensorDesc, axes) \
    do {                                                                                    \
        DML_REDUCE_OPERATOR_DESC desc = {};                                                 \
        desc.Function = DML_REDUCE_FUNCTION_##type;                                         \
        desc.InputTensor = &inputTensorDesc;                                                \
        desc.OutputTensor = &outputTensorDesc;                                              \
        desc.AxisCount = static_cast<UINT>(axes.size());                                    \
        desc.Axes = axes.data();                                                            \
        graphBuilder->CreateOperator(DML_OPERATOR_REDUCE, &desc);                           \
    } while (0)

#define SLICE_ONE_AXIS(axis, index)                                                           \
    do {                                                                                      \
        inputWindowOffsets[axis] =                                                            \
            starts[index] < 0 ? (starts[index] + inputDims[axis]) : starts[index];            \
        inputWindowSizes[axis] =                                                              \
            sizes[index] == -1 ? (inputDims[axis] - inputWindowOffsets[axis]) : sizes[index]; \
    } while (0)

    // Append IDENTITY to remove the strides of input tensor. Use this to implement Reshape,
    // Squeeze, Transpose and avoid creating an invaild graph with input = output.
    MaybeError Graph::AppendIdentity(DML_TENSOR_DESC& outputTensorDesc,
                                     const DML_TENSOR_DESC& inputTensorDesc) {
        DAWN_INVALID_IF(
            CreateDmlTensorDesc(outputTensorDesc, &inputTensorDesc, {}, {}, true).IsError(),
            "Failed to create DML_TENSOR_DESC.");
        DML_ACTIVATION_IDENTITY_OPERATOR_DESC operatorDesc{};
        operatorDesc.InputTensor = &inputTensorDesc;
        operatorDesc.OutputTensor = &outputTensorDesc;
        mGraphBuilder->CreateOperator(DML_OPERATOR_ACTIVATION_IDENTITY, &operatorDesc);
        return {};
    }

    // Strides are used to express broadcasting (by specifying a stride of 0) as well as
    // padding. If Strides is not specified, each dimension in the tensor is considered to
    // be contiguously packed, with no additional padding. The calculated strides refer to
    // https://docs.microsoft.com/en-us/windows/win32/direct3d12/dml-helper-functions#calculatestrides
    std::vector<UINT> CalculateStridesForBroadcast(std::vector<UINT> originDims,
                                                   std::vector<UINT> broadcastedDims,
                                                   const DML_TENSOR_DESC& inputTensorDesc,
                                                   size_t skipAxes = 0) {
        auto originRank = originDims.size(), broadcastedRank = broadcastedDims.size();
        if (originRank < skipAxes || originRank > broadcastedRank) {
            dawn::ErrorLog() << "Shapes are incompatible, broadcasting failed.";
            DAWN_ASSERT(0);
        }
        std::vector<bool> broadcastFlags(broadcastedRank, false);
        auto rankGap = broadcastedRank - originRank;
        for (size_t i = 0; i < rankGap; ++i) {
            broadcastFlags[i] = true;
        }
        for (size_t i = 0; i < originRank - skipAxes; ++i) {
            if (originDims[i] == 1 && broadcastedDims[rankGap + i] != 1) {
                broadcastFlags[rankGap + i] = true;
            }
        }

        for (size_t i = 0; i < broadcastedRank; ++i) {
            if (broadcastFlags[i]) {
                broadcastedDims[i] = 1;
            }
        }
        std::vector<UINT> strides(broadcastedRank);
        const DML_BUFFER_TENSOR_DESC* bufferDesc =
            reinterpret_cast<const DML_BUFFER_TENSOR_DESC*>(inputTensorDesc.Desc);
        DAWN_ASSERT(bufferDesc != nullptr && broadcastedRank >= bufferDesc->DimensionCount);
        auto existedStrides = bufferDesc->Strides;
        if (existedStrides != nullptr) {
            auto indexBegin = broadcastedRank - bufferDesc->DimensionCount;
            for (size_t i = 0, j = 0; i < broadcastedRank; ++i) {
                if (i < indexBegin) {
                    strides[i] = 0;
                } else {
                    strides[i] = broadcastFlags[i] ? 0 : existedStrides[j];
                    ++j;
                }
            }
        } else {
            strides[broadcastedRank - 1] = broadcastFlags[broadcastedRank - 1] ? 0 : 1;
            size_t elements = 1;
            for (size_t i = 1; i < broadcastedRank; i++) {
                size_t j = broadcastedRank - i - 1;
                elements *= broadcastedDims[j + 1];
                strides[j] = broadcastFlags[j] ? 0 : elements;
            }
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

    std::vector<UINT> transposeFilterDimensionsAsIohw(
        wnn::ConvTranspose2dFilterOperandLayout filterLayout,
        const std::vector<UINT>& filterDims) {
        std::vector<UINT> newFilterDims(4);
        switch (filterLayout) {
            case wnn::ConvTranspose2dFilterOperandLayout::Hwoi:
                newFilterDims[0] = filterDims[3];
                newFilterDims[1] = filterDims[2];
                newFilterDims[2] = filterDims[0];
                newFilterDims[3] = filterDims[1];
                break;
            case wnn::ConvTranspose2dFilterOperandLayout::Ohwi:
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

    std::vector<UINT> transposeFilterStridesAsIohw(
        wnn::ConvTranspose2dFilterOperandLayout filterLayout,
        const std::vector<UINT>& filterDims) {
        UINT hStride = 0, wStride = 0, iStride = 0, oStride = 0;
        switch (filterLayout) {
            case wnn::ConvTranspose2dFilterOperandLayout::Hwoi:
                hStride = filterDims[1] * filterDims[2] * filterDims[3];
                wStride = filterDims[2] * filterDims[3];
                oStride = filterDims[3];
                iStride = 1;
                break;
            case wnn::ConvTranspose2dFilterOperandLayout::Ohwi:
                oStride = filterDims[1] * filterDims[2] * filterDims[3];
                hStride = filterDims[2] * filterDims[3];
                wStride = filterDims[3];
                iStride = 1;
                break;
            default:
                DAWN_ASSERT(0);
                break;
        }
        return {iStride, oStride, hStride, wStride};
    }

    template <typename T>
    std::vector<UINT> ImplicitPadding(const T* options,
                                      const std::vector<UINT>& inputDims,
                                      const std::vector<UINT>& filterDims) {
        return webnn::native::utils::ComputeImplicitPaddingForAutoPad<T, UINT>(
            options, {inputDims[2], inputDims[3]},
            {filterDims[filterDims.size() - 2], filterDims[filterDims.size() - 1]});
    }

    template <typename T>
    std::vector<UINT> ExplicitPadding(const T* options) {
        UINT paddingTop = static_cast<UINT>(options->padding[0]);
        UINT paddingBottom = static_cast<UINT>(options->padding[1]);
        UINT paddingLeft = static_cast<UINT>(options->padding[2]);
        UINT paddingRight = static_cast<UINT>(options->padding[3]);

        return {paddingTop, paddingBottom, paddingLeft, paddingRight};
    }

    DML_RECURRENT_NETWORK_DIRECTION getRecurrentSequenceDirection(
        wnn::RecurrentNetworkDirection direction) {
        switch (direction) {
            case wnn::RecurrentNetworkDirection::Forward:
                return DML_RECURRENT_NETWORK_DIRECTION_FORWARD;
            case wnn::RecurrentNetworkDirection::Backward:
                return DML_RECURRENT_NETWORK_DIRECTION_BACKWARD;
            case wnn::RecurrentNetworkDirection::Both:
                return DML_RECURRENT_NETWORK_DIRECTION_BIDIRECTIONAL;
            default:
                dawn::ErrorLog() << "This direction type is not supported";
                DAWN_ASSERT(0);
        }
    }

    MaybeError Graph::CreateDmlTensorDesc(DML_TENSOR_DESC& createdTensorDesc,
                                          const std::vector<UINT>& dimensions,
                                          const std::vector<UINT>& strides,
                                          DML_TENSOR_DATA_TYPE dataType,
                                          DML_TENSOR_FLAGS tensorFlag) {
        std::shared_ptr<TensorDesc> tensorDesc(new TensorDesc);
        tensorDesc->dimensions = dimensions;
        tensorDesc->strides = strides;
        DAWN_INVALID_IF(!strides.empty() && dimensions.size() != strides.size(),
                        "Dimension size should be equal to strides size.");
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
                return DAWN_INTERNAL_ERROR("This data type is not supported");
        }

        size_t elementsCount = 1;
        DAWN_INVALID_IF(tensorDesc->dimensions.size() > DML_TENSOR_DIMENSION_COUNT_MAX1,
                        "Tensor dimension count is greater than DML_TENSOR_DIMENSION_COUNT_MAX1.");
        if (tensorDesc->dimensions.size() == 0) {
            tensorDesc->dimensions.resize(1);
            tensorDesc->dimensions[0] = 1;
        } else {
            for (uint32_t i = 0; i < tensorDesc->dimensions.size(); ++i) {
                auto dim = tensorDesc->dimensions[i];
                if (strides.empty()) {
                    elementsCount *= dim;
                } else {
                    // The specific dim from broadcasting shouldn't increase the count of
                    // elements.
                    if (strides[i] == 0) {
                        dim = 1;
                    }
                    elementsCount *= dim;
                }
            }
        }
        auto TotalTensorSizeInBytes = elementsCount * typeLength;
        tensorDesc->bufferDesc.DimensionCount = tensorDesc->dimensions.size();
        tensorDesc->bufferDesc.Sizes = tensorDesc->dimensions.data();
        tensorDesc->bufferDesc.Strides = tensorDesc->strides.data();
        tensorDesc->bufferDesc.TotalTensorSizeInBytes = TotalTensorSizeInBytes;
        tensorDesc->bufferDesc.GuaranteedBaseOffsetAlignment = 0;
        tensorDesc->bufferDesc.DataType = dataType;
        tensorDesc->bufferDesc.Flags = tensorFlag;

        mTensorsDesc.push_back(tensorDesc);
        createdTensorDesc = {DML_TENSOR_TYPE_BUFFER, &tensorDesc->bufferDesc};
        return {};
    }

    MaybeError Graph::CreateDmlTensorDesc(DML_TENSOR_DESC& createdTensorDesc,
                                          OperandDescriptor const* desc,
                                          DML_TENSOR_FLAGS tensorFlag) {
        DAWN_ASSERT(desc != nullptr);
        std::vector<UINT> dimensions;
        DML_TENSOR_DATA_TYPE dataType = DML_TENSOR_DATA_TYPE_FLOAT32;
        for (uint32_t i = 0; i < desc->dimensionsCount; ++i) {
            DAWN_INVALID_IF(desc->dimensions[i] < 0,
                            "DML doesn't support the negative dimension value.");
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
            return DAWN_INTERNAL_ERROR("This data type is not supported.");
        }

        DAWN_INVALID_IF(
            CreateDmlTensorDesc(createdTensorDesc, dimensions, {}, dataType, tensorFlag).IsError(),
            "Failed to create DML_TENSOR_DESC.");
        return {};
    }

    MaybeError Graph::CreateDmlTensorDesc(DML_TENSOR_DESC& createdTensorDesc,
                                          DML_TENSOR_DESC const* tensorDesc,
                                          std::vector<UINT> dimensions,
                                          std::vector<UINT> strides,
                                          bool useDefaultFlags) {
        DAWN_ASSERT(tensorDesc != nullptr);
        const DML_BUFFER_TENSOR_DESC* desc =
            reinterpret_cast<const DML_BUFFER_TENSOR_DESC*>(tensorDesc->Desc);

        if (dimensions.empty()) {
            dimensions.assign(desc->Sizes, desc->Sizes + desc->DimensionCount);
        }
        DML_TENSOR_FLAGS tensorFlags = useDefaultFlags ? DML_TENSOR_FLAG_NONE : desc->Flags;
        DAWN_INVALID_IF(
            CreateDmlTensorDesc(createdTensorDesc, dimensions, strides, desc->DataType, tensorFlags)
                .IsError(),
            "Failed to create DML_TENSOR_DESC.");
        return {};
    }

    std::shared_ptr<NodeBase> updateNode(std::shared_ptr<NodeBase>& node,
                                         const DML_TENSOR_DESC& outputTensorDesc) {
        DAWN_ASSERT(node != nullptr);
        switch (node->type) {
            case NodeType::ConstantInput:
            case NodeType::NonConstantInput: {
                std::shared_ptr<InputNode> newNode(new InputNode());
                memcpy(static_cast<void*>(newNode.get()), static_cast<void*>(node.get()),
                       sizeof(InputNode));
                newNode->outputTensorDesc = outputTensorDesc;
                std::shared_ptr<NodeBase> nodeBase(newNode);
                return nodeBase;
            }
            case NodeType::Intermediate: {
                std::shared_ptr<Node> newNode(new Node());
                memcpy(static_cast<void*>(newNode.get()), static_cast<void*>(node.get()),
                       sizeof(Node));
                newNode->outputTensorDesc = outputTensorDesc;
                std::shared_ptr<NodeBase> nodeBase(newNode);
                return nodeBase;
            }
            default:
                dawn::ErrorLog() << "Invalid node type";
                DAWN_ASSERT(0);
                return nullptr;
        }
    }

    MaybeError Graph::TransposeOutputToNhwc(std::shared_ptr<NodeBase>& inputNode,
                                            const std::vector<UINT>& nchwOutputDims) {
        DAWN_ASSERT(inputNode != nullptr);
        auto nhwcOutputStrides = transposeStrides(NchwToNhwc, nchwOutputDims);
        auto nhwcOutputDims = transposeDimensions(NchwToNhwc, nchwOutputDims);
        DML_TENSOR_DESC updatedTensorDesc;
        DAWN_INVALID_IF(CreateDmlTensorDesc(updatedTensorDesc, &inputNode->outputTensorDesc,
                                            nhwcOutputDims, nhwcOutputStrides, true)
                            .IsError(),
                        "Failed to create DML_TENSOR_DESC.");
        inputNode = updateNode(inputNode, updatedTensorDesc);
        return {};
    }

    Graph::Graph(Context* context) : GraphBase(context) {
        DeviceDescriptor desc;

        wnn::DevicePreference devicePreference = GetContext()->GetContextOptions().devicePreference;
        desc.useGpu = devicePreference == wnn::DevicePreference::Cpu ? false : true;

        wnn::PowerPreference powerPreference = GetContext()->GetContextOptions().powerPreference;
        switch (powerPreference) {
            case wnn::PowerPreference::High_performance:
                desc.gpuPreference = DXGI_GPU_PREFERENCE::DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE;
                break;
            case wnn::PowerPreference::Low_power:
                desc.gpuPreference = DXGI_GPU_PREFERENCE::DXGI_GPU_PREFERENCE_MINIMUM_POWER;
                break;
            default:
                desc.gpuPreference = DXGI_GPU_PREFERENCE::DXGI_GPU_PREFERENCE_UNSPECIFIED;
        }

#ifdef _DEBUG
        desc.useDebugLayer = true;
#endif

        mDevice = Device::Create(desc);
        DAWN_ASSERT(mDevice != nullptr);
        mGraphBuilder.reset(new GraphBuilder(mDevice->GetIDMLDevice()));
    }

    MaybeError Graph::AddConstant(const op::Constant* constant) {
        const OperandDescriptor* desc = constant->GetOperandDescriptor();
        std::shared_ptr<InputNode> inputNode(new InputNode());
        DML_TENSOR_DESC outputTensorDesc;
        DAWN_INVALID_IF(
            CreateDmlTensorDesc(outputTensorDesc, desc, DML_TENSOR_FLAG_OWNED_BY_DML).IsError(),
            "Failed to create DML_TENSOR_DESC.");
        inputNode->outputTensorDesc = outputTensorDesc;
        inputNode->name = "Input_Constant_" + std::to_string(mInputs.size());
        inputNode->type = NodeType::ConstantInput;
        inputNode->inputIndex = mInputs.size();
        inputNode->buffer = constant->GetBuffer();
        inputNode->byteLength = constant->GetByteLength();
        std::shared_ptr<NodeBase> nodeBase(inputNode);

        mGraphNodesMap[constant->PrimaryOutput()] = nodeBase;
        mInputs.push_back(inputNode);
        mConstantSet.insert(constant->PrimaryOutput());
        return {};
    }

    MaybeError Graph::CreateConstantInput(std::shared_ptr<InputNode>& inputNode,
                                          void const* value,
                                          size_t size,
                                          const std::vector<UINT>& dimensions,
                                          const std::vector<UINT>& strides,
                                          DML_TENSOR_DATA_TYPE dataType,
                                          DML_TENSOR_FLAGS tensorFlag) {
        std::unique_ptr<char> buffer(new char[size]);
        memcpy(buffer.get(), value, size);
        DML_TENSOR_DESC inputTensorDesc;
        DAWN_INVALID_IF(
            CreateDmlTensorDesc(inputTensorDesc, dimensions, strides, dataType, tensorFlag)
                .IsError(),
            "Failed to create DML_TENSOR_DESC.");

        inputNode.reset(new InputNode());
        inputNode->outputTensorDesc = inputTensorDesc;
        inputNode->name = "Input_Constant_" + std::to_string(mInputs.size());
        inputNode->type = NodeType::ConstantInput;
        inputNode->inputIndex = mInputs.size();
        inputNode->buffer = static_cast<void*>(buffer.get());
        inputNode->byteLength = size;

        mInputs.push_back(inputNode);
        mConstantsBuffer.push_back(std::move(buffer));
        return {};
    }

    MaybeError Graph::AddInput(const op::Input* input) {
        const OperandDescriptor* desc = input->GetOperandDescriptor();
        DML_TENSOR_DESC outputTensorDesc;
        DAWN_INVALID_IF(CreateDmlTensorDesc(outputTensorDesc, desc).IsError(),
                        "Failed to create DML_TENSOR_DESC.");
        std::shared_ptr<InputNode> inputNode(new InputNode());
        inputNode->outputTensorDesc = outputTensorDesc;
        inputNode->name = input->GetName();
        inputNode->type = NodeType::NonConstantInput;
        inputNode->inputIndex = mInputs.size();
        inputNode->byteLength = mTensorsDesc.back()->bufferDesc.TotalTensorSizeInBytes;
        std::shared_ptr<NodeBase> nodeBase(inputNode);

        mGraphNodesMap[input->PrimaryOutput()] = nodeBase;
        mInputs.push_back(inputNode);
        return {};
    }

    MaybeError Graph::AddBinary(const op::Binary* binary) {
        DAWN_ASSERT(binary->Inputs().size() == 2);
        DAWN_ASSERT(mGraphNodesMap.find(binary->Inputs()[0].Get()) != mGraphNodesMap.end());
        DAWN_ASSERT(mGraphNodesMap.find(binary->Inputs()[1].Get()) != mGraphNodesMap.end());

        auto aNode = mGraphNodesMap[binary->Inputs()[0].Get()];
        auto bNode = mGraphNodesMap[binary->Inputs()[1].Get()];
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

        DML_TENSOR_DESC aTensorDesc, bTensorDesc, outputTensorDesc;
        auto aNewStrides = CalculateStridesForBroadcast(aDims, aNewDims, aNode->outputTensorDesc,
                                                        broadcastSkipAxis);
        auto bNewStrides = CalculateStridesForBroadcast(bDims, bNewDims, bNode->outputTensorDesc,
                                                        broadcastSkipAxis);
        DAWN_INVALID_IF(
            CreateDmlTensorDesc(aTensorDesc, &aNode->outputTensorDesc, aNewDims, aNewStrides)
                .IsError(),
            "Failed to create DML_TENSOR_DESC.");
        DAWN_INVALID_IF(
            CreateDmlTensorDesc(bTensorDesc, &bNode->outputTensorDesc, bNewDims, bNewStrides)
                .IsError(),
            "Failed to create DML_TENSOR_DESC.");
        DAWN_INVALID_IF(
            CreateDmlTensorDesc(outputTensorDesc, &aNode->outputTensorDesc, outputNewDims, {}, true)
                .IsError(),
            "Failed to create DML_TENSOR_DESC.");

        switch (binary->GetType()) {
            case op::BinaryOpType::kAdd: {
                CREATE_BINARY_OPERATOR(mGraphBuilder, ADD, aTensorDesc, bTensorDesc,
                                       outputTensorDesc);
            } break;
            case op::BinaryOpType::kDiv: {
                CREATE_BINARY_OPERATOR(mGraphBuilder, DIVIDE, aTensorDesc, bTensorDesc,
                                       outputTensorDesc);
            } break;
            case op::BinaryOpType::kMul: {
                CREATE_BINARY_OPERATOR(mGraphBuilder, MULTIPLY, aTensorDesc, bTensorDesc,
                                       outputTensorDesc);
            } break;
            case op::BinaryOpType::kSub: {
                CREATE_BINARY_OPERATOR(mGraphBuilder, SUBTRACT, aTensorDesc, bTensorDesc,
                                       outputTensorDesc);
            } break;
            case op::BinaryOpType::kMax: {
                CREATE_BINARY_OPERATOR(mGraphBuilder, MAX, aTensorDesc, bTensorDesc,
                                       outputTensorDesc);
            } break;
            case op::BinaryOpType::kMin: {
                CREATE_BINARY_OPERATOR(mGraphBuilder, MIN, aTensorDesc, bTensorDesc,
                                       outputTensorDesc);
            } break;
            case op::BinaryOpType::kPower: {
                DML_ELEMENT_WISE_POW_OPERATOR_DESC operatorDesc{};
                operatorDesc.InputTensor = &aTensorDesc;
                operatorDesc.ExponentTensor = &bTensorDesc;
                operatorDesc.OutputTensor = &outputTensorDesc;
                mGraphBuilder->CreateOperator(DML_OPERATOR_ELEMENT_WISE_POW, &operatorDesc);
            } break;
            case op::BinaryOpType::kMatMul: {
                DML_GEMM_OPERATOR_DESC operatorDesc{};
                operatorDesc.ATensor = &aTensorDesc;
                operatorDesc.BTensor = &bTensorDesc;
                operatorDesc.OutputTensor = &outputTensorDesc;
                operatorDesc.Alpha = 1.0;
                mGraphBuilder->CreateOperator(DML_OPERATOR_GEMM, &operatorDesc);
            } break;
            default:
                return DAWN_UNIMPLEMENTED_ERROR(" Binary op is not implemented.");
        }
        if (outputDims != outputNewDims) {
            DAWN_INVALID_IF(CreateDmlTensorDesc(outputTensorDesc, &aNode->outputTensorDesc,
                                                outputDims, {}, true)
                                .IsError(),
                            "Failed to create DML_TENSOR_DESC.");
        }

        mGraphBuilder->AddNodes({aNode, bNode});
        mGraphNodesMap[binary->PrimaryOutput()] = mGraphBuilder->CreateNode(outputTensorDesc);
        return {};
    }

    MaybeError Graph::HardSwish(std::shared_ptr<NodeBase>& inputNode,
                                const std::vector<UINT>& inputDims) {
        dawn::WarningLog() << "The hardSwish is emulated from other operations, maybe the "
                              "performance isn't best";
        DML_TENSOR_DESC intermediateTensorDesc, inputTensorDesc = inputNode->outputTensorDesc;
        std::shared_ptr<InputNode> secondConstantInputNode;
        std::shared_ptr<NodeBase> intermediateNode, outputNode;
        uint32_t length = SizeOfShape(inputDims);
        DML_TENSOR_DESC outputTensorDesc;
        DAWN_INVALID_IF(
            CreateDmlTensorDesc(outputTensorDesc, &inputNode->outputTensorDesc, inputDims, {}, true)
                .IsError(),
            "Failed to create DML_TENSOR_DESC.");
        std::vector<float> constant(length, 3);
        // x+3
        {
            // Create the first constant input.
            std::shared_ptr<InputNode> firstConstantInputNode;
            DAWN_INVALID_IF(
                CreateConstantInput(firstConstantInputNode, constant.data(), length * sizeof(float),
                                    inputDims, {}, DML_TENSOR_DATA_TYPE_FLOAT32)
                    .IsError(),
                "Failed to create constant input.");
            CREATE_BINARY_OPERATOR(mGraphBuilder, ADD, inputTensorDesc,
                                   firstConstantInputNode->outputTensorDesc, outputTensorDesc);

            mGraphBuilder->AddNodes({inputNode, firstConstantInputNode});
            outputNode = mGraphBuilder->CreateNode(outputTensorDesc);
        }

        // min(6, (x + 3))
        {
            intermediateTensorDesc = outputNode->outputTensorDesc;
            intermediateNode = outputNode;
            constant = std::vector<float>(length, 6);
            DAWN_INVALID_IF(CreateConstantInput(secondConstantInputNode, constant.data(),
                                                length * sizeof(float), inputDims, {},
                                                DML_TENSOR_DATA_TYPE_FLOAT32)
                                .IsError(),
                            "Failed to create DML_TENSOR_DESC.");
            CREATE_BINARY_OPERATOR(mGraphBuilder, MIN, intermediateTensorDesc,
                                   secondConstantInputNode->outputTensorDesc, outputTensorDesc);

            mGraphBuilder->AddNodes({intermediateNode, secondConstantInputNode});
            outputNode = mGraphBuilder->CreateNode(outputTensorDesc);
        }

        // max(0, min(6, (x + 3)))
        {
            intermediateTensorDesc = outputNode->outputTensorDesc;
            intermediateNode = outputNode;
            constant = std::vector<float>(length, 0);
            // Create the third constant input.
            std::shared_ptr<InputNode> thirdConstantInputNode;
            DAWN_INVALID_IF(
                CreateConstantInput(thirdConstantInputNode, constant.data(), length * sizeof(float),
                                    inputDims, {}, DML_TENSOR_DATA_TYPE_FLOAT32)
                    .IsError(),
                "Failed to create constant input.");
            CREATE_BINARY_OPERATOR(mGraphBuilder, MAX, intermediateTensorDesc,
                                   thirdConstantInputNode->outputTensorDesc, outputTensorDesc);

            mGraphBuilder->AddNodes({intermediateNode, thirdConstantInputNode});
            outputNode = mGraphBuilder->CreateNode(outputTensorDesc);
        }

        // x * max(0, min(6, (x + 3)))
        {
            intermediateTensorDesc = outputNode->outputTensorDesc;
            intermediateNode = outputNode;
            CREATE_BINARY_OPERATOR(mGraphBuilder, MULTIPLY, inputTensorDesc, intermediateTensorDesc,
                                   outputTensorDesc);

            mGraphBuilder->AddNodes({inputNode, intermediateNode});
            outputNode = mGraphBuilder->CreateNode(outputTensorDesc);
        }

        // x * max(0, min(6, (x + 3))) / 6
        {
            intermediateTensorDesc = outputNode->outputTensorDesc;
            intermediateNode = outputNode;
            CREATE_BINARY_OPERATOR(mGraphBuilder, DIVIDE, intermediateTensorDesc,
                                   secondConstantInputNode->outputTensorDesc, outputTensorDesc);

            mGraphBuilder->AddNodes({intermediateNode, secondConstantInputNode});
            inputNode = mGraphBuilder->CreateNode(outputTensorDesc);
            return {};
        }
    }

    MaybeError Graph::AddUnary(const op::Unary* unary) {
        DAWN_ASSERT(unary->Inputs().size() == 1);
        const OperandBase* inputOperand = unary->Inputs()[0].Get();
        DAWN_ASSERT(mGraphNodesMap.find(inputOperand) != mGraphNodesMap.end());

        auto inputNode = mGraphNodesMap[inputOperand];
        auto inputDims = ConvertDimensions(inputOperand->Shape());
        std::vector<std::shared_ptr<NodeBase>> inputNodes = {inputNode};
        DML_TENSOR_DESC inputTensorDesc = inputNode->outputTensorDesc;
        switch (unary->GetType()) {
            case op::UnaryOpType::kAbs: {
                CREATE_UNARY_OPERATOR(mGraphBuilder, ELEMENT_WISE_ABS, inputTensorDesc);
            } break;
            case op::UnaryOpType::kCeil: {
                CREATE_UNARY_OPERATOR(mGraphBuilder, ELEMENT_WISE_CEIL, inputTensorDesc);
            } break;
            case op::UnaryOpType::kCos: {
                CREATE_UNARY_OPERATOR(mGraphBuilder, ELEMENT_WISE_COS, inputTensorDesc);
            } break;
            case op::UnaryOpType::kExp: {
                CREATE_UNARY_OPERATOR(mGraphBuilder, ELEMENT_WISE_EXP, inputTensorDesc);
            } break;
            case op::UnaryOpType::kFloor: {
                CREATE_UNARY_OPERATOR(mGraphBuilder, ELEMENT_WISE_FLOOR, inputTensorDesc);
            } break;
            case op::UnaryOpType::kHardSwish: {
                if (HardSwish(inputNode, inputDims).IsError()) {
                    return DAWN_INTERNAL_ERROR("Failed to create the HardSwish.");
                };
                mGraphNodesMap[unary->PrimaryOutput()] = inputNode;
                return {};
            }
            case op::UnaryOpType::kLog: {
                CREATE_UNARY_OPERATOR(mGraphBuilder, ELEMENT_WISE_LOG, inputTensorDesc);
            } break;
            case op::UnaryOpType::kLeakyRelu: {
                DML_ACTIVATION_LEAKY_RELU_OPERATOR_DESC operatorDesc{};
                operatorDesc.InputTensor = &inputTensorDesc;
                operatorDesc.OutputTensor = &inputTensorDesc;
                operatorDesc.Alpha = reinterpret_cast<const op::LeakyRelu*>(unary)->GetAlpha();
                mGraphBuilder->CreateOperator(DML_OPERATOR_ACTIVATION_LEAKY_RELU, &operatorDesc);
            } break;
            // DML doesn't support element-wise negative, emulated it from multiplying input by
            // -1.
            case op::UnaryOpType::kNeg: {
                uint32_t length = SizeOfShape(inputDims);
                std::shared_ptr<InputNode> constantInputNode;
                if (inputOperand->Type() == wnn::OperandType::Float32) {
                    std::vector<float> constant(length, -1);
                    DAWN_INVALID_IF(CreateConstantInput(constantInputNode, constant.data(),
                                                        length * sizeof(float), inputDims, {},
                                                        DML_TENSOR_DATA_TYPE_FLOAT32)
                                        .IsError(),
                                    "Failed to create constant input.");
                } else if (inputOperand->Type() == wnn::OperandType::Int32) {
                    std::vector<int32_t> constant(length, -1);
                    DAWN_INVALID_IF(CreateConstantInput(constantInputNode, constant.data(),
                                                        length * sizeof(int32_t), inputDims, {},
                                                        DML_TENSOR_DATA_TYPE_INT32)
                                        .IsError(),
                                    "Failed to create constant input.");
                } else {
                    return DAWN_UNIMPLEMENTED_ERROR("This data type is not supported for neg.");
                }

                CREATE_BINARY_OPERATOR(mGraphBuilder, MULTIPLY, inputTensorDesc,
                                       constantInputNode->outputTensorDesc, inputTensorDesc);
                inputNodes.push_back(constantInputNode);
            } break;
            case op::UnaryOpType::kRelu: {
                CREATE_UNARY_OPERATOR(mGraphBuilder, ACTIVATION_RELU, inputTensorDesc);
            } break;
            case op::UnaryOpType::kSigmoid: {
                CREATE_UNARY_OPERATOR(mGraphBuilder, ACTIVATION_SIGMOID, inputTensorDesc);
            } break;
            case op::UnaryOpType::kSin: {
                CREATE_UNARY_OPERATOR(mGraphBuilder, ELEMENT_WISE_SIN, inputTensorDesc);
            } break;
            case op::UnaryOpType::kSoftmax: {
                CREATE_UNARY_OPERATOR(mGraphBuilder, ACTIVATION_SOFTMAX, inputTensorDesc);
            } break;
            case op::UnaryOpType::kTan: {
                CREATE_UNARY_OPERATOR(mGraphBuilder, ELEMENT_WISE_TAN, inputTensorDesc);
            } break;
            case op::UnaryOpType::kTanh: {
                CREATE_UNARY_OPERATOR(mGraphBuilder, ACTIVATION_TANH, inputTensorDesc);
            } break;
            default:
                return DAWN_UNIMPLEMENTED_ERROR("This Unary op is not implemented.");
        }

        mGraphBuilder->AddNodes(inputNodes);
        mGraphNodesMap[unary->PrimaryOutput()] = mGraphBuilder->CreateNode(inputTensorDesc);
        return {};
    }

    MaybeError Graph::AddSplit(const op::Split* split) {
        DAWN_ASSERT(split->Inputs().size() == 1);
        auto inputOperand = split->Inputs()[0].Get();
        DAWN_ASSERT(mGraphNodesMap.find(inputOperand) != mGraphNodesMap.end());

        auto inputDims = inputOperand->Shape();
        int32_t axis = split->GetAxis();
        // This value must be in the range [0, InputTensor.DimensionCount - 1]. Negative values
        // address dimensions from the end.
        if (axis < 0) {
            axis = axis + inputDims.size();
        }

        size_t outputNum = split->Outputs().size();

        auto inputNode = mGraphNodesMap[inputOperand];
        DML_TENSOR_DESC inputTensorDesc = inputNode->outputTensorDesc;
        std::vector<DML_TENSOR_DESC> outputTensorsDesc;
        outputTensorsDesc.reserve(outputNum);
        for (size_t i = 0; i < outputNum; ++i) {
            DML_TENSOR_DESC outputTensorDesc;
            DAWN_INVALID_IF(
                CreateDmlTensorDesc(outputTensorDesc, &inputNode->outputTensorDesc,
                                    ConvertDimensions(split->Outputs()[i].Get()->Shape()), {}, true)
                    .IsError(),
                "Failed to create DML_TENSOR_DESC.");
            outputTensorsDesc.push_back(outputTensorDesc);
        }

        DML_SPLIT_OPERATOR_DESC dmlSplitOperatorDesc{};
        dmlSplitOperatorDesc.Axis = axis;
        dmlSplitOperatorDesc.InputTensor = &inputTensorDesc;
        dmlSplitOperatorDesc.OutputCount = outputTensorsDesc.size();
        dmlSplitOperatorDesc.OutputTensors = outputTensorsDesc.data();
        mGraphBuilder->CreateOperator(DML_OPERATOR_SPLIT, &dmlSplitOperatorDesc);
        mGraphBuilder->AddNodes({inputNode});
        for (size_t i = 0; i < outputNum; ++i) {
            mGraphNodesMap[split->Outputs()[i].Get()] =
                mGraphBuilder->CreateNode(outputTensorsDesc[i], i);
        }
        return {};
    }

    MaybeError Graph::AddReshape(const op::Reshape* reshape) {
        DAWN_ASSERT(reshape->Inputs().size() == 1);
        const OperandBase* inputOperand = reshape->Inputs()[0].Get();
        DAWN_ASSERT(mGraphNodesMap.find(inputOperand) != mGraphNodesMap.end());

        auto inputNode = mGraphNodesMap[inputOperand];
        auto outputDims = ConvertDimensions(reshape->Outputs()[0].Get()->Shape());
        // Reshape needn't new strides, because the layout has not been changed.
        DML_TENSOR_DESC outputTensorDesc;
        DAWN_INVALID_IF(
            CreateDmlTensorDesc(outputTensorDesc, &inputNode->outputTensorDesc, outputDims)
                .IsError(),
            "Failed to create DML_TENSOR_DESC.");
        // Reshape is not a real node in DML, just need to update its' origin node.
        mGraphNodesMap[reshape->PrimaryOutput()] = updateNode(inputNode, outputTensorDesc);
        return {};
    }

    MaybeError Graph::AddTranspose(const op::Transpose* transpose) {
        DAWN_ASSERT(transpose->Inputs().size() == 1);
        const OperandBase* inputOperand = transpose->Inputs()[0].Get();
        DAWN_ASSERT(mGraphNodesMap.find(inputOperand) != mGraphNodesMap.end());

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

        auto inputNode = mGraphNodesMap[inputOperand];
        // Transpose is not a real node in DML, just need to update its' origin node.
        DML_TENSOR_DESC outputTensorDesc;
        DAWN_INVALID_IF(CreateDmlTensorDesc(outputTensorDesc, &inputNode->outputTensorDesc,
                                            outputDims, transposedStrides)
                            .IsError(),
                        "Failed to create DML_TENSOR_DESC.");
        mGraphNodesMap[transpose->PrimaryOutput()] = updateNode(inputNode, outputTensorDesc);
        return {};
    }

    DML_OPERATOR_DESC* CreateFusedOperator(
        FusionType fusionType,
        DML_ACTIVATION_LINEAR_OPERATOR_DESC& dmlActicationOperatorDesc,
        DML_OPERATOR_DESC& dmlFusedOperatorDesc,
        float alpha = 0.0,
        float beta = 0.0) {
        dmlActicationOperatorDesc.InputTensor = nullptr;
        dmlActicationOperatorDesc.OutputTensor = nullptr;
        dmlActicationOperatorDesc.Alpha = alpha;
        dmlActicationOperatorDesc.Beta = beta;
        switch (fusionType) {
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

    DML_OPERATOR_DESC* CreateFusedOperator(
        FusionOperatorBase* activation,
        DML_ACTIVATION_LINEAR_OPERATOR_DESC& dmlActicationOperatorDesc,
        DML_OPERATOR_DESC& dmlFusedOperatorDesc) {
        if (activation == nullptr) {
            return nullptr;
        }
        float alpha = activation->GetFusionType() == FusionType::LeakyRelu
                          ? reinterpret_cast<op::FusionLeakyRelu*>(activation)->GetAlpha()
                          : 0.0;
        return CreateFusedOperator(activation->GetFusionType(), dmlActicationOperatorDesc,
                                   dmlFusedOperatorDesc, alpha);
    }

    MaybeError Graph::EmulateFusedOperator(FusionOperatorBase* activation,
                                           std::shared_ptr<NodeBase>& inputNode,
                                           const std::vector<UINT>& inputDims) {
        // HardSwish and Clamp are not supported for fusion, so we add them directly to
        // emulate. Currently we implement Relu6 operator by Clamp.
        if (activation == nullptr) {
            return {};
        }
        DAWN_ASSERT(inputNode != nullptr);
        auto fusionType = activation->GetFusionType();
        if (fusionType == FusionType::Clamp) {
            auto clamp = reinterpret_cast<const op::FusionClamp*>(activation);
            inputNode = Clamp(clamp, inputNode);
        } else if (fusionType == FusionType::HardSwish) {
            if (HardSwish(inputNode, inputDims).IsError()) {
                return DAWN_INTERNAL_ERROR("Failed to create the HardSwish.");
            };
        }
        return {};
    }

    std::shared_ptr<NodeBase> Graph::Clamp(const op::ClampBase* clamp,
                                           std::shared_ptr<NodeBase>& inputNode) {
        DAWN_ASSERT(inputNode != nullptr);
        DML_TENSOR_DESC inputTensorDesc = inputNode->outputTensorDesc;

        // Set OutputTensor = InputTensor with the same strides to optimize performance.
        DML_ELEMENT_WISE_CLIP_OPERATOR_DESC desc = {};
        desc.InputTensor = &inputTensorDesc;
        desc.OutputTensor = &inputTensorDesc;
        desc.ScaleBias = nullptr;
        desc.Min = clamp->GetMinValue();
        desc.Max = clamp->GetMaxValue();
        mGraphBuilder->CreateOperator(DML_OPERATOR_ELEMENT_WISE_CLIP, &desc);

        mGraphBuilder->AddNodes({inputNode});
        return mGraphBuilder->CreateNode(inputTensorDesc);
    }

    MaybeError Graph::AddClamp(const op::Clamp* clamp) {
        auto inputsOperand = clamp->Inputs();
        DAWN_ASSERT(inputsOperand.size() == 1);
        auto inputNode = mGraphNodesMap[inputsOperand[0].Get()];
        mGraphNodesMap[clamp->PrimaryOutput()] = Clamp(clamp, inputNode);
        return {};
    }

    std::vector<UINT> transposeStridesToNchw(const std::vector<UINT>& inputDims,
                                             const DML_TENSOR_DESC& inputTensorDesc) {
        const DML_BUFFER_TENSOR_DESC* bufferDesc =
            reinterpret_cast<const DML_BUFFER_TENSOR_DESC*>(inputTensorDesc.Desc);
        DAWN_ASSERT(bufferDesc != nullptr && bufferDesc->DimensionCount == 4);
        auto strides = bufferDesc->Strides;
        if (strides != nullptr) {
            return {strides[0], strides[3], strides[1], strides[2]};
        } else {
            return transposeStrides(NhwcToNchw, inputDims);
        }
    }

    MaybeError Graph::AddConv2d(const op::Conv2d* conv2d) {
        auto inputsOperand = conv2d->Inputs();
        DAWN_ASSERT(inputsOperand.size() == 2 || inputsOperand.size() == 3);
        DAWN_ASSERT(mGraphNodesMap.find(inputsOperand[0].Get()) != mGraphNodesMap.end());
        DAWN_ASSERT(mGraphNodesMap.find(inputsOperand[1].Get()) != mGraphNodesMap.end());

        auto inputNode = mGraphNodesMap[inputsOperand[0].Get()];
        auto filterNode = mGraphNodesMap[inputsOperand[1].Get()];

        auto inputDims = ConvertDimensions(inputsOperand[0].Get()->Shape());
        auto filterDims = ConvertDimensions(inputsOperand[1].Get()->Shape());
        auto outputDims = ConvertDimensions(conv2d->Outputs()[0].Get()->Shape());
        std::vector<UINT> newInputDims = inputDims, newFilterDims = filterDims,
                          newOutputDims = outputDims, newInputStrides, newFilterStrides;

        const Conv2dOptions* options = conv2d->GetOptions();

        DML_TENSOR_DESC inputTensorDesc = inputNode->outputTensorDesc;
        if (options->inputLayout == wnn::InputOperandLayout::Nhwc) {
            newInputDims = transposeDimensions(NhwcToNchw, inputDims);
            newOutputDims = transposeDimensions(NhwcToNchw, outputDims);
            newInputStrides = transposeStridesToNchw(inputDims, inputTensorDesc);
            DAWN_INVALID_IF(CreateDmlTensorDesc(inputTensorDesc, &inputNode->outputTensorDesc,
                                                newInputDims, newInputStrides)
                                .IsError(),
                            "Failed to create DML_TENSOR_DESC.");
        }

        DML_TENSOR_DESC filterTensorDesc = filterNode->outputTensorDesc;
        if (options->filterLayout != wnn::Conv2dFilterOperandLayout::Oihw) {
            newFilterDims = transposeFilterDimensionsAsOihw(options->filterLayout, filterDims);
            newFilterStrides = transposeFilterStridesAsOihw(options->filterLayout, filterDims);
            DAWN_INVALID_IF(CreateDmlTensorDesc(filterTensorDesc, &filterNode->outputTensorDesc,
                                                newFilterDims, newFilterStrides)
                                .IsError(),
                            "Failed to create DML_TENSOR_DESC.");
        }

        std::vector<std::shared_ptr<NodeBase>> inputNodes = {inputNode, filterNode};

        const DML_TENSOR_DESC* biasTensorDescPtr = nullptr;
        DML_TENSOR_DESC newBiasTensorDesc = {};
        if (options->bias != nullptr) {
            DAWN_ASSERT(mGraphNodesMap.find(inputsOperand[2].Get()) != mGraphNodesMap.end());
            auto biasNode = mGraphNodesMap[inputsOperand[2].Get()];
            auto biasDims = ConvertDimensions(conv2d->Inputs()[2].Get()->Shape());
            if (biasDims[0] != newFilterDims[0] || biasDims.size() != 1) {
                return DAWN_INTERNAL_ERROR(
                    "The bias should be 1-D tensor with the shape of [output_channels].");
            }

            // Reshape bias from 1-D to 4-D for NCHW layout.
            std::vector<UINT> newBiasDims = {1, biasDims[0], 1, 1};
            DAWN_INVALID_IF(
                CreateDmlTensorDesc(newBiasTensorDesc, &biasNode->outputTensorDesc, newBiasDims)
                    .IsError(),
                "Failed to create DML_TENSOR_DESC.");
            biasTensorDescPtr = &newBiasTensorDesc;
            inputNodes.push_back(biasNode);
        }
        DML_TENSOR_DESC outputTensorDesc;
        DAWN_INVALID_IF(CreateDmlTensorDesc(outputTensorDesc, &inputNode->outputTensorDesc,
                                            newOutputDims, {}, true)
                            .IsError(),
                        "Failed to create DML_TENSOR_DESC.");

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

        DML_CONVOLUTION_OPERATOR_DESC operatorDesc{};
        operatorDesc.InputTensor = &inputTensorDesc;
        operatorDesc.FilterTensor = &filterTensorDesc;
        operatorDesc.BiasTensor = biasTensorDescPtr;
        operatorDesc.OutputTensor = &outputTensorDesc;

        operatorDesc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
        operatorDesc.Direction = DML_CONVOLUTION_DIRECTION_FORWARD;
        operatorDesc.DimensionCount = inputDims.size() - 2;
        operatorDesc.Strides = strides.data();
        operatorDesc.Dilations = dilations.data();
        operatorDesc.StartPadding = startPadding.data();
        operatorDesc.EndPadding = endPadding.data();
        operatorDesc.OutputPadding = defaultOutPadding.data();
        operatorDesc.GroupCount = static_cast<UINT>(options->groups);
        operatorDesc.FusedActivation = fusedActivation;
        mGraphBuilder->CreateOperator(DML_OPERATOR_CONVOLUTION, &operatorDesc);

        mGraphBuilder->AddNodes(inputNodes);
        auto outputNode = mGraphBuilder->CreateNode(outputTensorDesc);

        // Transpose output from nchw->nhwc.
        if (options->inputLayout == wnn::InputOperandLayout::Nhwc) {
            if (TransposeOutputToNhwc(outputNode, newOutputDims).IsError()) {
                return DAWN_INTERNAL_ERROR("Failed to transpose output from Nchw to Nhwc.");
            };
        }

        if (EmulateFusedOperator(options->activation, outputNode, outputDims).IsError()) {
            return DAWN_INTERNAL_ERROR("Failed to emulate fused operator.");
        }
        mGraphNodesMap[conv2d->PrimaryOutput()] = outputNode;
        return {};
    }

    MaybeError Graph::AddPool2d(const op::Pool2d* pool2d) {
        DAWN_ASSERT(pool2d->Inputs().size() == 1);
        const OperandBase* inputOperand = pool2d->Inputs()[0].Get();
        DAWN_ASSERT(mGraphNodesMap.find(inputOperand) != mGraphNodesMap.end());

        auto inputNode = mGraphNodesMap[inputOperand];
        auto inputDims = ConvertDimensions(inputOperand->Shape());
        auto outputDims = ConvertDimensions(pool2d->Outputs()[0].Get()->Shape());
        std::vector<UINT> newInputDims = inputDims, newOutputDims = outputDims, newInputStrides;
        const Pool2dOptions* options = pool2d->GetOptions();

        DML_TENSOR_DESC inputTensorDesc = inputNode->outputTensorDesc;
        if (options->layout == wnn::InputOperandLayout::Nhwc) {
            newInputDims = transposeDimensions(NhwcToNchw, inputDims);
            newOutputDims = transposeDimensions(NhwcToNchw, outputDims);
            newInputStrides = transposeStridesToNchw(inputDims, inputTensorDesc);
            DAWN_INVALID_IF(CreateDmlTensorDesc(inputTensorDesc, &inputNode->outputTensorDesc,
                                                newInputDims, newInputStrides)
                                .IsError(),
                            "Failed to create DML_TENSOR_DESC.");
        }
        DML_TENSOR_DESC outputTensorDesc;
        DAWN_INVALID_IF(CreateDmlTensorDesc(outputTensorDesc, &inputNode->outputTensorDesc,
                                            newOutputDims, {}, true)
                            .IsError(),
                        "Failed to create DML_TENSOR_DESC.");
        std::vector<UINT> strides, dilations;
        strides.assign(reinterpret_cast<const UINT*>(options->strides),
                       reinterpret_cast<const UINT*>(options->strides) + options->stridesCount);
        dilations.assign(reinterpret_cast<const UINT*>(options->dilations),
                         reinterpret_cast<const UINT*>(options->dilations) + options->stridesCount);

        std::vector<UINT> windowSizes;
        if (options->windowDimensions != nullptr) {
            const UINT* windowDimensions = reinterpret_cast<const UINT*>(options->windowDimensions);
            windowSizes.assign(windowDimensions, windowDimensions + options->windowDimensionsCount);
        } else {
            windowSizes = {newInputDims[2], newInputDims[3]};
        }

        auto padding = options->autoPad == wnn::AutoPad::Explicit
                           ? ExplicitPadding<Pool2dOptions>(options)
                           : ImplicitPadding<Pool2dOptions>(options, newInputDims, windowSizes);
        std::vector<UINT> startPadding = {padding[0], padding[2]};
        std::vector<UINT> endPadding = {padding[1], padding[3]};

        if (pool2d->GetType() == op::Pool2dType::kAveragePool2d) {
            if (dilations[0] != 1 || dilations[1] != 1) {
                return DAWN_INTERNAL_ERROR("The dilations of average pool2d are not supported.");
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
            mGraphBuilder->CreateOperator(DML_OPERATOR_AVERAGE_POOLING, &desc);
        } else if (pool2d->GetType() == op::Pool2dType::kL2Pool2d) {
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
            mGraphBuilder->CreateOperator(DML_OPERATOR_LP_POOLING, &desc);
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
                outputDims = transposeDimensions(NchwToNhwc, newOutputDims);
                // Update output tensor.
                DAWN_INVALID_IF(CreateDmlTensorDesc(outputTensorDesc, newOutputDims).IsError(),
                                "Failed to create DML_TENSOR_DESC.");
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
            mGraphBuilder->CreateOperator(DML_OPERATOR_MAX_POOLING2, &desc);
        } else {
            return DAWN_INTERNAL_ERROR("This pool2d type is not supported.");
        }

        mGraphBuilder->AddNodes({inputNode});
        auto outputNode = mGraphBuilder->CreateNode(outputTensorDesc);

        // Transpose output from nchw->nhwc.
        if (options->layout == wnn::InputOperandLayout::Nhwc) {
            if (TransposeOutputToNhwc(outputNode, newOutputDims).IsError()) {
                return DAWN_INTERNAL_ERROR("Failed to transpose output from Nchw to Nhwc.");
            };
        }

        mGraphNodesMap[pool2d->PrimaryOutput()] = outputNode;
        return {};
    }

    MaybeError Graph::AddPad(const op::Pad* pad) {
        auto inputsOperand = pad->Inputs();
        DAWN_ASSERT(inputsOperand.size() == 2);
        DAWN_ASSERT(mGraphNodesMap.find(inputsOperand[0].Get()) != mGraphNodesMap.end());
        DAWN_ASSERT(mGraphNodesMap.find(inputsOperand[1].Get()) != mGraphNodesMap.end());

        auto inputNode = mGraphNodesMap[inputsOperand[0].Get()];
        auto paddingNode = mGraphNodesMap[inputsOperand[1].Get()];
        auto inputDims = ConvertDimensions(inputsOperand[0].Get()->Shape());
        auto paddingDims = ConvertDimensions(inputsOperand[1].Get()->Shape());
        auto outputDims = ConvertDimensions(pad->Outputs()[0].Get()->Shape());
        size_t inputRank = inputDims.size();

        // Workaround(mingming): If padding was added in mGraph, it must be used.
        // Use "Pad_"+std::to_string(mGraphNodesMap.size()) to generate a unique name for the
        // output node. This may be a dml issue:
        // https://github.com/microsoft/DirectML/issues/133.
        std::string name = "Pad_" + std::to_string(mGraphNodesMap.size());
        auto paddingTensorDesc = paddingNode->outputTensorDesc;

        // Ensure that the DML_TENSOR_FLAGS of output tensor is DML_TENSOR_FLAG_NONE.
        DML_TENSOR_DESC outputTensorDesc;
        DAWN_INVALID_IF(
            CreateDmlTensorDesc(outputTensorDesc, &paddingTensorDesc, paddingDims, {}, true)
                .IsError(),
            "Failed to create DML_TENSOR_DESC.");

        DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC operatorDesc{};
        operatorDesc.InputTensor = &paddingTensorDesc;
        operatorDesc.OutputTensor = &outputTensorDesc;
        operatorDesc.ScaleBias = nullptr;
        mGraphBuilder->CreateOperator(DML_OPERATOR_ELEMENT_WISE_IDENTITY, &operatorDesc);

        mGraphBuilder->AddNodes({paddingNode});
        auto outputNode = mGraphBuilder->CreateNode(paddingTensorDesc);
        outputNode->name = name;
        mGraphBuilder->SetGraphOutput(outputNode, mOutputs.size());
        mOutputs.push_back(*reinterpret_cast<Node*>(outputNode.get()));

        if (mConstantSet.find(inputsOperand[1].Get()) == mConstantSet.end()) {
            return DAWN_INTERNAL_ERROR("The padding constant is not found.");
        }

        const op::Constant* paddingConstant =
            reinterpret_cast<const op::Constant*>(inputsOperand[1]->Operator());
        const uint32_t* paddingData = static_cast<const uint32_t*>(paddingConstant->GetBuffer());
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
        auto inputTensorDesc = inputNode->outputTensorDesc;
        DAWN_INVALID_IF(CreateDmlTensorDesc(outputTensorDesc, &inputNode->outputTensorDesc,
                                            outputDims, {}, true)
                            .IsError(),
                        "Failed to create DML_TENSOR_DESC.");

        DML_PADDING_OPERATOR_DESC desc = {};
        desc.InputTensor = &inputTensorDesc;
        desc.OutputTensor = &outputTensorDesc;
        desc.PaddingMode = paddingMode;
        desc.PaddingValue = options->value;
        desc.DimensionCount = static_cast<UINT>(startPadding.size());
        desc.StartPadding = startPadding.data();
        desc.EndPadding = endPadding.data();
        mGraphBuilder->CreateOperator(DML_OPERATOR_PADDING, &desc);

        mGraphBuilder->AddNodes({inputNode});
        mGraphNodesMap[pad->PrimaryOutput()] = mGraphBuilder->CreateNode(outputTensorDesc);
        return {};
    }

    MaybeError Graph::AddBatchNorm(const op::BatchNorm* batchNorm) {
        auto inputs = batchNorm->Inputs();
        DAWN_ASSERT(inputs.size() == 3 || inputs.size() == 4 || inputs.size() == 5);
        DAWN_ASSERT(mGraphNodesMap.find(batchNorm->Inputs()[0].Get()) != mGraphNodesMap.end());
        auto inputNode = mGraphNodesMap[batchNorm->Inputs()[0].Get()];
        auto inputDims = ConvertDimensions(inputs[0].Get()->Shape());
        auto outputDims = ConvertDimensions(batchNorm->Outputs()[0].Get()->Shape());
        std::vector<UINT> newInputDims = inputDims, newOutputDims = outputDims, newInputStrides;
        const BatchNormOptions* options = batchNorm->GetOptions();

        // When input is a 4-D tensor of the "nchw" or "nhwc" layout, options.axis should be set
        // to 1 or 3 respectively.
        uint32_t axis = options->axis;
        DML_TENSOR_DESC inputTensorDesc = inputNode->outputTensorDesc;
        if (options->axis == 3) {
            axis = 1;
            newInputDims = transposeDimensions(NhwcToNchw, inputDims);
            newOutputDims = transposeDimensions(NhwcToNchw, outputDims);
            newInputStrides = transposeStridesToNchw(inputDims, inputTensorDesc);
            DAWN_INVALID_IF(CreateDmlTensorDesc(inputTensorDesc, &inputNode->outputTensorDesc,
                                                newInputDims, newInputStrides)
                                .IsError(),
                            "Failed to create DML_TENSOR_DESC.");
        }

        // Reshape 1D mean, variance, scale, bias to 4D with setting 1 to automatically
        // broadcast.
        std::vector<DML_TENSOR_DESC> tensorsDesc;
        std::vector<std::shared_ptr<NodeBase>> inputNodes;
        for (size_t i = 1; i < inputs.size(); ++i) {
            DAWN_ASSERT(mGraphNodesMap.find(batchNorm->Inputs()[i].Get()) != mGraphNodesMap.end());
            auto node = mGraphNodesMap[batchNorm->Inputs()[i].Get()];
            auto dims = ConvertDimensions(inputs[i].Get()->Shape());
            DAWN_ASSERT(dims.size() == 1);
            if (dims[0] != newInputDims[axis]) {
                return DAWN_INTERNAL_ERROR(
                    "The 1-D tensor of the values whose length size is not equal to the size "
                    "of "
                    "the input dimension denoted by options.axis.");
            }
            // This tensor's dimensions should be { BatchCount, ChannelCount, Height,Width}.
            // Set 1 to automatically broadcast those dimensions across the input.
            std::vector<UINT> expandDims(4, 1);
            expandDims[axis] = dims[0];
            DML_TENSOR_DESC tensorDesc;
            DAWN_INVALID_IF(
                CreateDmlTensorDesc(tensorDesc, &node->outputTensorDesc, expandDims).IsError(),
                "Failed to create DML_TENSOR_DESC.");
            tensorsDesc.push_back(tensorDesc);
            inputNodes.push_back(updateNode(node, tensorDesc));
        }

        if (options->scale == nullptr) {
            float scale = 1.0;
            std::vector<UINT> scaleDims = {1, newInputDims[1], 1, 1};
            // Create a constant scale.
            std::shared_ptr<InputNode> constantInputNode;
            DAWN_INVALID_IF(CreateConstantInput(constantInputNode, &scale, sizeof(float),
                                                {1, 1, 1, 1}, {}, DML_TENSOR_DATA_TYPE_FLOAT32)
                                .IsError(),
                            "Failed to create constant input.");
            tensorsDesc.insert(
                options->bias == nullptr ? tensorsDesc.end() : tensorsDesc.begin() + 2,
                constantInputNode->outputTensorDesc);
            inputNodes.insert(options->bias == nullptr ? inputNodes.end() : inputNodes.begin() + 2,
                              constantInputNode);
        }

        if (options->bias == nullptr) {
            float bias = 0;
            std::vector<UINT> biasDims = {1, newInputDims[1], 1, 1};
            // Create a constant scale.
            std::shared_ptr<InputNode> constantInputNode;
            DAWN_INVALID_IF(CreateConstantInput(constantInputNode, &bias, sizeof(float),
                                                {1, 1, 1, 1}, {}, DML_TENSOR_DATA_TYPE_FLOAT32)
                                .IsError(),
                            "Failed to create constant input.");
            tensorsDesc.push_back(constantInputNode->outputTensorDesc);
            inputNodes.push_back(constantInputNode);
        }
        DML_TENSOR_DESC outputTensorDesc;
        DAWN_INVALID_IF(CreateDmlTensorDesc(outputTensorDesc, &inputNode->outputTensorDesc,
                                            newOutputDims, {}, true)
                            .IsError(),
                        "Failed to create DML_TENSOR_DESC.");

        DML_ACTIVATION_LINEAR_OPERATOR_DESC dmlActicationOperatorDesc{};
        DML_OPERATOR_DESC dmlFusedOperatorDesc = {};
        DML_OPERATOR_DESC* fusedActivation = CreateFusedOperator(
            options->activation, dmlActicationOperatorDesc, dmlFusedOperatorDesc);

        DML_BATCH_NORMALIZATION_OPERATOR_DESC desc = {};
        desc.InputTensor = &inputTensorDesc;
        desc.MeanTensor = &tensorsDesc[0];
        desc.VarianceTensor = &tensorsDesc[1];
        desc.ScaleTensor = &tensorsDesc[2];
        desc.BiasTensor = &tensorsDesc[3];
        desc.OutputTensor = &outputTensorDesc;
        desc.Spatial = true;
        desc.Epsilon = options->epsilon;
        desc.FusedActivation = fusedActivation;
        mGraphBuilder->CreateOperator(DML_OPERATOR_BATCH_NORMALIZATION, &desc);

        mGraphBuilder->AddNodes(
            {inputNode, inputNodes[0], inputNodes[1], inputNodes[2], inputNodes[3]});
        auto outputNode = mGraphBuilder->CreateNode(outputTensorDesc);

        // Transpose output from nchw->nhwc.
        if (options->axis == 3) {
            if (TransposeOutputToNhwc(outputNode, newOutputDims).IsError()) {
                return DAWN_INTERNAL_ERROR("Failed to transpose output from Nchw to Nhwc.");
            };
        }

        if (EmulateFusedOperator(options->activation, outputNode, outputDims).IsError()) {
            return DAWN_INTERNAL_ERROR("Failed to emulate fused operator.");
        };
        mGraphNodesMap[batchNorm->PrimaryOutput()] = outputNode;
        return {};
    }

    MaybeError Graph::AddConvTranspose2d(const op::ConvTranspose2d* convTranspose2d) {
        auto inputsOperand = convTranspose2d->Inputs();
        DAWN_ASSERT(inputsOperand.size() == 2 || inputsOperand.size() == 3);
        DAWN_ASSERT(mGraphNodesMap.find(inputsOperand[0].Get()) != mGraphNodesMap.end());
        DAWN_ASSERT(mGraphNodesMap.find(inputsOperand[1].Get()) != mGraphNodesMap.end());

        auto inputNode = mGraphNodesMap[inputsOperand[0].Get()];
        auto filterNode = mGraphNodesMap[inputsOperand[1].Get()];

        auto inputDims = ConvertDimensions(inputsOperand[0].Get()->Shape());
        auto filterDims = ConvertDimensions(inputsOperand[1].Get()->Shape());
        std::vector<UINT> newInputDims = inputDims, newFilterDims = filterDims, newInputStrides,
                          newFilterStrides;

        const ConvTranspose2dOptions* options = convTranspose2d->GetOptions();

        DML_TENSOR_DESC inputTensorDesc = inputNode->outputTensorDesc;
        if (options->inputLayout == wnn::InputOperandLayout::Nhwc) {
            newInputDims = transposeDimensions(NhwcToNchw, inputDims);
            newInputStrides = transposeStridesToNchw(inputDims, inputTensorDesc);
            DAWN_INVALID_IF(CreateDmlTensorDesc(inputTensorDesc, &inputNode->outputTensorDesc,
                                                newInputDims, newInputStrides)
                                .IsError(),
                            "Failed to create DML_TENSOR_DESC.");
        }

        DML_TENSOR_DESC filterTensorDesc = filterNode->outputTensorDesc;
        if (options->filterLayout != wnn::ConvTranspose2dFilterOperandLayout::Iohw) {
            newFilterDims = transposeFilterDimensionsAsIohw(options->filterLayout, filterDims);
            newFilterStrides = transposeFilterStridesAsIohw(options->filterLayout, filterDims);
            DAWN_INVALID_IF(CreateDmlTensorDesc(filterTensorDesc, &filterNode->outputTensorDesc,
                                                newFilterDims, newFilterStrides)
                                .IsError(),
                            "Failed to create DML_TENSOR_DESC.");
        }

        std::vector<std::shared_ptr<NodeBase>> inputNodes = {inputNode, filterNode};

        const DML_TENSOR_DESC* biasTensorDescPtr = nullptr;
        DML_TENSOR_DESC newBiasTensorDesc = {};
        if (options->bias != nullptr) {
            DAWN_ASSERT(mGraphNodesMap.find(inputsOperand[2].Get()) != mGraphNodesMap.end());
            auto biasNode = mGraphNodesMap[inputsOperand[2].Get()];
            auto biasDims = ConvertDimensions(convTranspose2d->Inputs()[2].Get()->Shape());
            if (biasDims[0] != newFilterDims[0] || biasDims.size() != 1) {
                return DAWN_INTERNAL_ERROR(
                    "The bias should be 1-D tensor with the shape of [output_channels].");
            }

            // Reshape bias from 1-D to 4-D for NCHW layout.
            std::vector<UINT> newBiasDims = {1, biasDims[0], 1, 1};
            DAWN_INVALID_IF(
                CreateDmlTensorDesc(newBiasTensorDesc, &biasNode->outputTensorDesc, newBiasDims)
                    .IsError(),
                "Failed to create DML_TENSOR_DESC.");
            biasTensorDescPtr = &newBiasTensorDesc;
            inputNodes.push_back(biasNode);
        }

        std::vector<UINT> outputDims(4);
        if (options->outputSizes != nullptr) {
            std::vector<UINT> outputSizes;
            outputSizes.assign(options->outputSizes,
                               options->outputSizes + options->outputSizesCount);
            if (options->inputLayout == wnn::InputOperandLayout::Nchw) {
                outputDims = {inputDims[0], newFilterDims[1], outputSizes[0], outputSizes[1]};
            } else {
                outputDims = {inputDims[0], outputSizes[0], outputSizes[1], newFilterDims[1]};
            }
        } else {
            outputDims = ConvertDimensions(convTranspose2d->Outputs()[0]->Shape());
        }
        std::vector<UINT> newOutputDims = outputDims;
        if (options->inputLayout == wnn::InputOperandLayout::Nhwc) {
            newOutputDims = transposeDimensions(NhwcToNchw, outputDims);
        }
        DML_TENSOR_DESC outputTensorDesc;
        DAWN_INVALID_IF(CreateDmlTensorDesc(outputTensorDesc, &inputNode->outputTensorDesc,
                                            newOutputDims, {}, true)
                            .IsError(),
                        "Failed to create DML_TENSOR_DESC.");

        // FIXME(nhu): strides, dilations, padding should be uint32_t
        // need to fix the spec.
        std::vector<UINT> strides, dilations, outputPadding;
        strides.assign(options->strides, options->strides + options->stridesCount);
        dilations.assign(options->dilations, options->dilations + options->dilationsCount);
        outputPadding.assign(options->outputPadding,
                             options->outputPadding + options->outputPaddingCount);

        std::vector<UINT> padding(4);
        if (options->autoPad == wnn::AutoPad::Explicit) {
            padding = ExplicitPadding<ConvTranspose2dOptions>(options);
        } else {
            std::vector<UINT> inputSize = {inputDims[2], inputDims[3]};
            std::vector<UINT> filterSize = {filterDims[2], filterDims[3]};
            padding = webnn::native::utils::ComputeImplicitPaddingForConvTranspose2dAutoPad(
                options, inputSize, filterSize);
        }
        std::vector<UINT> startPadding = {padding[0], padding[2]};
        std::vector<UINT> endPadding = {padding[1], padding[3]};

        DML_ACTIVATION_LINEAR_OPERATOR_DESC dmlActicationOperatorDesc{};
        DML_OPERATOR_DESC dmlFusedOperatorDesc = {};
        DML_OPERATOR_DESC* fusedActivation = CreateFusedOperator(
            options->activation, dmlActicationOperatorDesc, dmlFusedOperatorDesc);

        DML_CONVOLUTION_OPERATOR_DESC operatorDesc{};
        operatorDesc.InputTensor = &inputTensorDesc;
        operatorDesc.FilterTensor = &filterTensorDesc;
        operatorDesc.BiasTensor = biasTensorDescPtr;
        operatorDesc.OutputTensor = &outputTensorDesc;
        operatorDesc.Mode = DML_CONVOLUTION_MODE_CONVOLUTION;
        operatorDesc.Direction = DML_CONVOLUTION_DIRECTION_BACKWARD;
        operatorDesc.DimensionCount = inputDims.size() - 2;
        operatorDesc.Strides = strides.data();
        operatorDesc.Dilations = dilations.data();
        operatorDesc.StartPadding = startPadding.data();
        operatorDesc.EndPadding = endPadding.data();
        operatorDesc.OutputPadding = outputPadding.data();
        operatorDesc.GroupCount = static_cast<UINT>(options->groups);
        operatorDesc.FusedActivation = fusedActivation;
        mGraphBuilder->CreateOperator(DML_OPERATOR_CONVOLUTION, &operatorDesc);

        mGraphBuilder->AddNodes(inputNodes);
        auto outputNode = mGraphBuilder->CreateNode(outputTensorDesc);

        // Transpose output from nchw->nhwc.
        if (options->inputLayout == wnn::InputOperandLayout::Nhwc) {
            if (TransposeOutputToNhwc(outputNode, newOutputDims).IsError()) {
                return DAWN_INTERNAL_ERROR("Failed to transpose output from Nchw to Nhwc.");
            };
        }

        if (EmulateFusedOperator(options->activation, outputNode, outputDims).IsError()) {
            return DAWN_INTERNAL_ERROR("Failed to emulate fused operator.");
        }
        mGraphNodesMap[convTranspose2d->PrimaryOutput()] = outputNode;
        return {};
    }

    MaybeError Graph::AddGru(const op::Gru* gru) {
        const auto inputsOperand = gru->Inputs();
        DAWN_ASSERT(inputsOperand.size() >= 3 && inputsOperand.size() <= 6);
        DAWN_ASSERT(mGraphNodesMap.find(inputsOperand[0].Get()) != mGraphNodesMap.end());
        DAWN_ASSERT(mGraphNodesMap.find(inputsOperand[1].Get()) != mGraphNodesMap.end());
        DAWN_ASSERT(mGraphNodesMap.find(inputsOperand[2].Get()) != mGraphNodesMap.end());
        std::vector<std::shared_ptr<NodeBase>> inputNodes;

        // Input: 4D tensor with the Sizes of { 1, seq_length, batch_size, input_size }.
        // Need to reshape input from WebNN 3-D to DML 4-D.
        auto inputNode = mGraphNodesMap[inputsOperand[0].Get()];
        auto webnnInputDims = ConvertDimensions(inputsOperand[0].Get()->Shape());
        std::vector<UINT> inputDims = {1, webnnInputDims[0], webnnInputDims[1], webnnInputDims[2]};
        DML_TENSOR_DESC inputTensorDesc;
        DAWN_INVALID_IF(
            CreateDmlTensorDesc(inputTensorDesc, &inputNode->outputTensorDesc, inputDims).IsError(),
            "Failed to create DML_TENSOR_DESC.");
        inputNodes.push_back(inputNode);

        // Weight: 4D tensor with the Sizes of { 1, num_directions, 3 * hidden_size, input_size
        // }. Need to reshape weight from WebNN 3-D to DML 4-D. The TENSOR_FLAGS of weight, bias
        // and hiddenInit in gru must be DML_TENSOR_FLAG_NONE.
        auto constantWeightNode = mGraphNodesMap[inputsOperand[1].Get()];
        auto webnnWeightDims = ConvertDimensions(inputsOperand[1].Get()->Shape());
        std::vector<UINT> weightDims = {1, webnnWeightDims[0], webnnWeightDims[1],
                                        webnnWeightDims[2]};
        // Workaround: append identity to convert constant input tensor with
        // DML_TENSOR_FLAG_OWNED_BY_DML falg to input tenor with DML_TENSOR_FLAG_NONE flag.
        DML_TENSOR_DESC constantweightTensorDesc, weightTensorDesc;
        DAWN_INVALID_IF(CreateDmlTensorDesc(constantweightTensorDesc,
                                            &constantWeightNode->outputTensorDesc, weightDims)
                            .IsError(),
                        "Failed to create DML_TENSOR_DESC.");
        DAWN_INVALID_IF(AppendIdentity(weightTensorDesc, constantweightTensorDesc).IsError(),
                        "Failed to append identity.");
        mGraphBuilder->AddNodes({constantWeightNode});
        auto weightNode = mGraphBuilder->CreateNode(weightTensorDesc);
        inputNodes.push_back(weightNode);

        // Recurrence: 4D tensor with the Sizes { 1, num_directions, 3 * hidden_size,
        // hidden_size }. Need to reshape recurrence from WebNN 3-D to DML 4-D. Need to convert
        // tensor flag to NONE.
        auto constantRecurrenceNode = mGraphNodesMap[inputsOperand[2].Get()];
        auto webnnRecurrenceDims = ConvertDimensions(inputsOperand[2].Get()->Shape());
        std::vector<UINT> recurrenceDims = {1, webnnRecurrenceDims[0], webnnRecurrenceDims[1],
                                            webnnRecurrenceDims[2]};
        DML_TENSOR_DESC constantRecurrenceTensorDesc, recurrenceTensorDesc;
        DAWN_INVALID_IF(
            CreateDmlTensorDesc(constantRecurrenceTensorDesc,
                                &constantRecurrenceNode->outputTensorDesc, recurrenceDims)
                .IsError(),
            "Failed to create DML_TENSOR_DESC.");
        DAWN_INVALID_IF(
            AppendIdentity(recurrenceTensorDesc, constantRecurrenceTensorDesc).IsError(),
            "Failed to append identity.");
        mGraphBuilder->AddNodes({constantRecurrenceNode});
        auto recurrenceNode = mGraphBuilder->CreateNode(recurrenceTensorDesc);
        inputNodes.push_back(recurrenceNode);

        const GruOptions* options = gru->GetOptions();
        UINT operandIndex = 3;

        // Bias: 4D tensor with the Sizes of { 1, 1, num_directions, 6 * hidden_size }.
        // Need to concat bias tensor and recurrentBias tensor.
        // Need to reshape bias from WebNN 2-D to DML 4-D.
        std::vector<UINT> webnnBiasDims = {weightDims[1],
                                           weightDims[2]};  // { num_directions, 3 * hidden_size }
        uint32_t webnnBiasLength = SizeOfShape(webnnBiasDims);
        std::vector<float> biasConstantData(webnnBiasLength, 0);
        std::shared_ptr<webnn::native::dml::NodeBase> webnnBiasNode;
        if (options->bias != nullptr) {
            DAWN_ASSERT(mGraphNodesMap.find(inputsOperand[operandIndex].Get()) !=
                        mGraphNodesMap.end());
            webnnBiasNode = mGraphNodesMap[inputsOperand[operandIndex].Get()];
            operandIndex++;
        } else {
            std::shared_ptr<InputNode> constantInputNode;
            DAWN_INVALID_IF(CreateConstantInput(constantInputNode, biasConstantData.data(),
                                                webnnBiasLength * sizeof(float), webnnBiasDims, {},
                                                DML_TENSOR_DATA_TYPE_FLOAT32)
                                .IsError(),
                            "Failed to create constant input.");
            webnnBiasNode = constantInputNode;
        }

        std::shared_ptr<webnn::native::dml::NodeBase> webnnRecurrentBiasNode;
        if (options->recurrentBias != nullptr) {
            DAWN_ASSERT(mGraphNodesMap.find(inputsOperand[operandIndex].Get()) !=
                        mGraphNodesMap.end());
            webnnRecurrentBiasNode = mGraphNodesMap[inputsOperand[operandIndex].Get()];
            operandIndex++;
        } else {
            std::shared_ptr<InputNode> constantInputNode;
            DAWN_INVALID_IF(CreateConstantInput(constantInputNode, biasConstantData.data(),
                                                webnnBiasLength * sizeof(float), webnnBiasDims, {},
                                                DML_TENSOR_DATA_TYPE_FLOAT32)
                                .IsError(),
                            "Failed to create constant input.");
            webnnRecurrentBiasNode = constantInputNode;
        }
        // Concat
        std::vector<DML_TENSOR_DESC> joinInputTensorDescs = {
            webnnBiasNode->outputTensorDesc, webnnRecurrentBiasNode->outputTensorDesc};
        DML_TENSOR_DESC joinOutputTensorDesc;
        DAWN_INVALID_IF(CreateDmlTensorDesc(joinOutputTensorDesc, &webnnBiasNode->outputTensorDesc,
                                            {webnnBiasDims[0], webnnBiasDims[1] * 2}, {}, true)
                            .IsError(),
                        "Failed to create DML_TENSOR_DESC.");
        DML_JOIN_OPERATOR_DESC joinDesc = {};
        joinDesc.Axis = 1;
        joinDesc.InputCount = static_cast<uint32_t>(joinInputTensorDescs.size());
        joinDesc.InputTensors = joinInputTensorDescs.data();
        joinDesc.OutputTensor = &joinOutputTensorDesc;
        mGraphBuilder->CreateOperator(DML_OPERATOR_JOIN, &joinDesc);

        mGraphBuilder->AddNodes({webnnBiasNode, webnnRecurrentBiasNode});
        auto biasNode = mGraphBuilder->CreateNode(joinOutputTensorDesc);
        // Reshape
        std::vector<UINT> biasDims = {1, 1, webnnBiasDims[0],
                                      webnnBiasDims[1] * 2};  // { num_directions, 6 * hidden_size }

        DML_TENSOR_DESC biasTensorDesc;
        DAWN_INVALID_IF(
            CreateDmlTensorDesc(biasTensorDesc, &biasNode->outputTensorDesc, biasDims).IsError(),
            "Failed to create DML_TENSOR_DESC.");
        inputNodes.push_back(biasNode);

        // HiddenInit: 4D tensor with the Sizes of { 1, num_directions, batch_size, hidden_size
        // }. Need to reshape hiddenInit from WebNN 3-D to DML 4-D. Need to convert tensor flag
        // to NONE.
        DML_TENSOR_DESC constantHiddenInitTensorDesc, hiddenInitTensorDesc;
        DML_TENSOR_DESC* hiddenInitTensorDescPtr = nullptr;
        if (options->initialHiddenState != nullptr) {
            DAWN_ASSERT(mGraphNodesMap.find(inputsOperand[operandIndex].Get()) !=
                        mGraphNodesMap.end());
            auto constantHiddenInitNode = mGraphNodesMap[inputsOperand[operandIndex].Get()];
            auto webnnHiddenInitDims =
                ConvertDimensions(inputsOperand[operandIndex].Get()->Shape());
            std::vector<UINT> hiddenInitDims = {1, webnnHiddenInitDims[0], webnnHiddenInitDims[1],
                                                webnnHiddenInitDims[2]};
            DAWN_INVALID_IF(
                CreateDmlTensorDesc(constantHiddenInitTensorDesc,
                                    &constantHiddenInitNode->outputTensorDesc, hiddenInitDims)
                    .IsError(),
                "Failed to create DML_TENSOR_DESC.");
            DAWN_INVALID_IF(
                AppendIdentity(hiddenInitTensorDesc, constantHiddenInitTensorDesc).IsError(),
                "Failed to append identity.");
            hiddenInitTensorDescPtr = &hiddenInitTensorDesc;
            mGraphBuilder->AddNodes({constantHiddenInitNode});
            auto hiddenInitNode = mGraphBuilder->CreateNode(hiddenInitTensorDesc);
            inputNodes.push_back(hiddenInitNode);
        }

        // Outputs Tensor
        DML_TENSOR_DESC outputSequenceTensorDesc = {};
        DML_TENSOR_DESC* outputSequenceTensorDescPtr = nullptr;
        if (options->returnSequence) {
            std::vector<UINT> outputSequenceSizes(4);
            outputSequenceSizes[0] = inputDims[1];       // SequenceLength
            outputSequenceSizes[1] = recurrenceDims[1];  // NumDirections
            outputSequenceSizes[2] = inputDims[2];       // BatchSize
            outputSequenceSizes[3] = recurrenceDims[3];  // HiddenSize
            DAWN_INVALID_IF(
                CreateDmlTensorDesc(outputSequenceTensorDesc, outputSequenceSizes).IsError(),
                "Failed to create DML_TENSOR_DESC.");
            outputSequenceTensorDescPtr = &outputSequenceTensorDesc;
        }

        std::vector<UINT> outputSingleSizes(4);
        outputSingleSizes[0] = 1;
        outputSingleSizes[1] = recurrenceDims[1];
        outputSingleSizes[2] = inputDims[2];
        outputSingleSizes[3] = recurrenceDims[3];
        DML_TENSOR_DESC outputSingleTensorDesc;
        DAWN_INVALID_IF(CreateDmlTensorDesc(outputSingleTensorDesc, outputSingleSizes).IsError(),
                        "Failed to create DML_TENSOR_DESC.");

        // Attributes
        DML_RECURRENT_NETWORK_DIRECTION direction =
            getRecurrentSequenceDirection(options->direction);
        DML_ACTIVATION_LINEAR_OPERATOR_DESC fActicationOperatorDesc{}, gActicationOperatorDesc{};
        DML_OPERATOR_DESC fFusedOperatorDesc = {}, gFusedOperatorDesc = {}, *fActivation,
                          *gActivation;
        if (options->activations == nullptr) {
            fActivation = CreateFusedOperator(FusionType::Sigmoid, fActicationOperatorDesc,
                                              fFusedOperatorDesc);
            gActivation =
                CreateFusedOperator(FusionType::Tanh, gActicationOperatorDesc, gFusedOperatorDesc);
        } else {
            fActivation = CreateFusedOperator(options->activations->Get(0), fActicationOperatorDesc,
                                              fFusedOperatorDesc);
            gActivation = CreateFusedOperator(options->activations->Get(1), gActicationOperatorDesc,
                                              gFusedOperatorDesc);
        }
        UINT activationDescCount;
        std::vector<DML_OPERATOR_DESC> activations;
        if (direction == DML_RECURRENT_NETWORK_DIRECTION_BIDIRECTIONAL) {
            activationDescCount = 4;
            activations = {*fActivation, *gActivation, *fActivation, *gActivation};
        } else {
            activationDescCount = 2;
            activations = {*fActivation, *gActivation};
        }
        bool linearBeforeReset = options->resetAfter;

        DML_GRU_OPERATOR_DESC desc = {};
        desc.InputTensor = &inputTensorDesc;
        desc.WeightTensor = &weightTensorDesc;
        desc.RecurrenceTensor = &recurrenceTensorDesc;
        desc.BiasTensor = &biasTensorDesc;
        desc.HiddenInitTensor = hiddenInitTensorDescPtr;
        desc.SequenceLengthsTensor = nullptr;
        desc.OutputSequenceTensor = outputSequenceTensorDescPtr;
        desc.OutputSingleTensor = &outputSingleTensorDesc;
        desc.ActivationDescCount = activationDescCount;
        desc.ActivationDescs = activations.data();
        desc.Direction = direction;
        desc.LinearBeforeReset = linearBeforeReset;
        mGraphBuilder->CreateOperator(DML_OPERATOR_GRU, &desc);

        mGraphBuilder->AddNodes(inputNodes);
        auto outputSingleNode = mGraphBuilder->CreateNode(outputSingleTensorDesc, 1);
        auto webnnOutputSingleDims = ConvertDimensions(gru->Outputs()[0].Get()->Shape());
        DML_TENSOR_DESC webnnOutputSingleTensorDesc;
        DAWN_INVALID_IF(
            CreateDmlTensorDesc(webnnOutputSingleTensorDesc, &outputSingleNode->outputTensorDesc,
                                webnnOutputSingleDims)
                .IsError(),
            "Failed to create DML_TENSOR_DESC.");
        mGraphNodesMap[gru->PrimaryOutput()] =
            updateNode(outputSingleNode, webnnOutputSingleTensorDesc);
        if (options->returnSequence) {
            auto outputSequenceNode = mGraphBuilder->CreateNode(outputSequenceTensorDesc, 0);
            mGraphNodesMap[gru->Outputs()[1].Get()] = outputSequenceNode;
        }
        return {};
    }

    MaybeError Graph::AddReduce(const op::Reduce* reduce) {
        DAWN_ASSERT(reduce->Inputs().size() == 1);
        const OperandBase* inputOperand = reduce->Inputs()[0].Get();
        DAWN_ASSERT(mGraphNodesMap.find(inputOperand) != mGraphNodesMap.end());

        auto inputNode = mGraphNodesMap[inputOperand];
        const ReduceOptions* options = reduce->GetOptions();
        std::vector<std::uint32_t> axes;
        auto inputDims = ConvertDimensions(inputOperand->Shape());
        auto outputDims = ConvertDimensions(reduce->Outputs()[0].Get()->Shape());

        auto inputTensorDesc = inputNode->outputTensorDesc;
        auto reducedDims = inputDims;
        for (size_t i = 0; i < options->axesCount; ++i) {
            // Axes values must be in the range [0, InputTensor.DimensionCount - 1].
            // The dimensions to reduce where -1 means the last dimension.
            uint32_t axis = options->axes[i] == -1 ? inputDims.size() - 1 : options->axes[i];
            axes.push_back(axis);
            reducedDims[axis] = 1;
        }
        DML_TENSOR_DESC outputTensorDesc;
        DAWN_INVALID_IF(
            CreateDmlTensorDesc(outputTensorDesc, &inputTensorDesc, reducedDims, {}, true)
                .IsError(),
            "Failed to create DML_TENSOR_DESC.");
        switch (reduce->GetType()) {
            case op::ReduceType::kReduceL1: {
                CREATE_REDUCE_OPERATOR(mGraphBuilder, L1, inputTensorDesc, outputTensorDesc, axes);
            } break;
            case op::ReduceType::kReduceL2: {
                CREATE_REDUCE_OPERATOR(mGraphBuilder, L2, inputTensorDesc, outputTensorDesc, axes);
            } break;
            case op::ReduceType::kReduceMax: {
                CREATE_REDUCE_OPERATOR(mGraphBuilder, MAX, inputTensorDesc, outputTensorDesc, axes);
            } break;
            case op::ReduceType::kReduceMean: {
                CREATE_REDUCE_OPERATOR(mGraphBuilder, AVERAGE, inputTensorDesc, outputTensorDesc,
                                       axes);
            } break;
            case op::ReduceType::kReduceMin: {
                CREATE_REDUCE_OPERATOR(mGraphBuilder, MIN, inputTensorDesc, outputTensorDesc, axes);
            } break;
            case op::ReduceType::kReduceProduct: {
                CREATE_REDUCE_OPERATOR(mGraphBuilder, MULTIPLY, inputTensorDesc, outputTensorDesc,
                                       axes);
            } break;
            case op::ReduceType::kReduceSum: {
                CREATE_REDUCE_OPERATOR(mGraphBuilder, SUM, inputTensorDesc, outputTensorDesc, axes);
            } break;
            default:
                return DAWN_INTERNAL_ERROR("The reduce op type isn't supported.");
        }
        // Reshape if dimensions needn't be kept. Output node has been updated with new output
        // dims.
        if (!options->keepDimensions) {
            DAWN_INVALID_IF(
                CreateDmlTensorDesc(outputTensorDesc, &outputTensorDesc, outputDims).IsError(),
                "Failed to create DML_TENSOR_DESC.");
        }
        mGraphBuilder->AddNodes({inputNode});
        mGraphNodesMap[reduce->PrimaryOutput()] = mGraphBuilder->CreateNode(outputTensorDesc);
        ;
        return {};
    }

    MaybeError Graph::AddResample2d(const op::Resample2d* resample2d) {
        DAWN_ASSERT(resample2d->Inputs().size() == 1);
        const OperandBase* inputOperand = resample2d->Inputs()[0].Get();
        DAWN_ASSERT(mGraphNodesMap.find(inputOperand) != mGraphNodesMap.end());

        auto inputNode = mGraphNodesMap[inputOperand];
        auto inputDims = ConvertDimensions(inputOperand->Shape());
        auto outputDims = ConvertDimensions(resample2d->Outputs()[0].Get()->Shape());

        auto inputTensorDesc = inputNode->outputTensorDesc;
        DML_TENSOR_DESC outputTensorDesc;
        DAWN_INVALID_IF(
            CreateDmlTensorDesc(outputTensorDesc, &inputTensorDesc, outputDims).IsError(),
            "Failed to create DML_TENSOR_DESC.");

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
        mGraphBuilder->CreateOperator(DML_OPERATOR_RESAMPLE1, &desc);

        mGraphBuilder->AddNodes({inputNode});
        mGraphNodesMap[resample2d->PrimaryOutput()] = mGraphBuilder->CreateNode(outputTensorDesc);
        return {};
    }

    MaybeError Graph::AddSlice(const op::Slice* slice) {
        DAWN_ASSERT(slice->Inputs().size() == 1);
        const OperandBase* inputOperand = slice->Inputs()[0].Get();
        DAWN_ASSERT(mGraphNodesMap.find(inputOperand) != mGraphNodesMap.end());

        auto inputNode = mGraphNodesMap[inputOperand];
        auto inputDims = ConvertDimensions(inputOperand->Shape());
        auto outputDims = ConvertDimensions(slice->Outputs()[0].Get()->Shape());

        auto inputTensorDesc = inputNode->outputTensorDesc;
        DML_TENSOR_DESC outputTensorDesc;
        DAWN_INVALID_IF(
            CreateDmlTensorDesc(outputTensorDesc, &inputTensorDesc, outputDims, {}, true).IsError(),
            "Failed to create DML_TENSOR_DESC.");

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
        mGraphBuilder->CreateOperator(DML_OPERATOR_SLICE1, &desc);

        mGraphBuilder->AddNodes({inputNode});
        mGraphNodesMap[slice->PrimaryOutput()] = mGraphBuilder->CreateNode(outputTensorDesc);
        return {};
    }

    MaybeError Graph::AddSqueeze(const op::Squeeze* squeeze) {
        DAWN_ASSERT(squeeze->Inputs().size() == 1);
        const OperandBase* inputOperand = squeeze->Inputs()[0].Get();
        DAWN_ASSERT(mGraphNodesMap.find(inputOperand) != mGraphNodesMap.end());

        auto inputNode = mGraphNodesMap[inputOperand];
        auto outputDims = ConvertDimensions(squeeze->Outputs()[0].Get()->Shape());
        // Squeeze perform like reshape which needn't new strides, because the layout has not
        // been changed.
        DML_TENSOR_DESC outputTensorDesc;
        DAWN_INVALID_IF(
            CreateDmlTensorDesc(outputTensorDesc, &inputNode->outputTensorDesc, outputDims)
                .IsError(),
            "Failed to create DML_TENSOR_DESC.");
        // Squeeze is not a real node in DML, just need to update its' origin node.
        mGraphNodesMap[squeeze->PrimaryOutput()] = updateNode(inputNode, outputTensorDesc);
        return {};
    }

    MaybeError Graph::AddInstanceNorm(const op::InstanceNorm* instanceNorm) {
        auto inputs = instanceNorm->Inputs();
        DAWN_ASSERT(inputs.size() == 1 || inputs.size() == 2 || inputs.size() == 3);
        DAWN_ASSERT(mGraphNodesMap.find(instanceNorm->Inputs()[0].Get()) != mGraphNodesMap.end());
        auto inputNode = mGraphNodesMap[instanceNorm->Inputs()[0].Get()];
        auto inputDims = ConvertDimensions(inputs[0].Get()->Shape());
        auto outputDims = ConvertDimensions(instanceNorm->Outputs()[0].Get()->Shape());
        std::vector<UINT> newInputDims = inputDims, newOutputDims = outputDims, newInputStrides;
        const InstanceNormOptions* options = instanceNorm->GetOptions();

        DML_TENSOR_DESC inputTensorDesc = inputNode->outputTensorDesc;
        if (options->layout == wnn::InputOperandLayout::Nhwc) {
            newInputDims = transposeDimensions(NhwcToNchw, inputDims);
            newOutputDims = transposeDimensions(NhwcToNchw, outputDims);
            newInputStrides = transposeStridesToNchw(inputDims, inputTensorDesc);
            DAWN_INVALID_IF(CreateDmlTensorDesc(inputTensorDesc, &inputNode->outputTensorDesc,
                                                newInputDims, newInputStrides)
                                .IsError(),
                            "Failed to create DML_TENSOR_DESC.");
        }
        DML_TENSOR_DESC outputTensorDesc;
        DAWN_INVALID_IF(CreateDmlTensorDesc(outputTensorDesc, &inputNode->outputTensorDesc,
                                            newOutputDims, {}, true)
                            .IsError(),
                        "Failed to create DML_TENSOR_DESC.");

        std::vector<DML_TENSOR_DESC> tensorsDesc;
        std::vector<std::shared_ptr<NodeBase>> inputNodes;
        // Reshape 1D scale, bias to 4D with setting 1 to automatically broadcast.
        for (size_t i = 1; i < inputs.size(); ++i) {
            DAWN_ASSERT(mGraphNodesMap.find(instanceNorm->Inputs()[i].Get()) !=
                        mGraphNodesMap.end());
            auto node = mGraphNodesMap[inputs[i].Get()];
            auto dims = ConvertDimensions(inputs[i].Get()->Shape());
            DAWN_ASSERT(dims.size() == 1);
            if (dims[0] != newInputDims[1]) {
                return DAWN_INTERNAL_ERROR(
                    "The 1-D tensor of the values whose length size is not equal to the size "
                    "of "
                    "feature dimension of the input ");
            }
            // This tensor's dimensions should be {BatchCount, ChannelCount, Height, Width}.
            // Set 1 to automatically broadcast those dimensions across the input.
            std::vector<UINT> expandDims(4, 1);
            expandDims[1] = dims[0];
            DML_TENSOR_DESC tensorDesc;
            DAWN_INVALID_IF(
                CreateDmlTensorDesc(tensorDesc, &node->outputTensorDesc, expandDims).IsError(),
                "Failed to create DML_TENSOR_DESC.");
            tensorsDesc.push_back(tensorDesc);
            inputNodes.push_back(updateNode(node, tensorDesc));
        }

        // Set tensor's dimensions to {1, channel, 1, 1} if scale or bias is null.
        if (options->scale == nullptr) {
            std::vector<float> scale(newInputDims[1], 1.0);
            std::vector<UINT> scaleDims = {1, newInputDims[1], 1, 1};
            // Create a constant scale.
            std::shared_ptr<InputNode> constantInputNode;
            DAWN_INVALID_IF(CreateConstantInput(constantInputNode, scale.data(),
                                                newInputDims[1] * sizeof(float), scaleDims, {},
                                                DML_TENSOR_DATA_TYPE_FLOAT32)
                                .IsError(),
                            "Failed to create constant input.");
            tensorsDesc.insert(tensorsDesc.begin(), constantInputNode->outputTensorDesc);
            inputNodes.insert(inputNodes.begin(), constantInputNode);
        }

        if (options->bias == nullptr) {
            std::vector<float> bias(newInputDims[1], 0.0);
            std::vector<UINT> biasDims = {1, newInputDims[1], 1, 1};
            // Create a constant scale.
            std::shared_ptr<InputNode> constantInputNode;
            DAWN_INVALID_IF(
                CreateConstantInput(constantInputNode, bias.data(), newInputDims[1] * sizeof(float),
                                    biasDims, {}, DML_TENSOR_DATA_TYPE_FLOAT32)
                    .IsError(),
                "Failed to create constant input.");
            tensorsDesc.push_back(constantInputNode->outputTensorDesc);
            inputNodes.push_back(constantInputNode);
        }

        std::vector<const uint32_t> axes({2, 3});

        DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC desc = {};
        desc.InputTensor = &inputTensorDesc;
        desc.ScaleTensor = &tensorsDesc[0];
        desc.BiasTensor = &tensorsDesc[1];
        desc.OutputTensor = &outputTensorDesc;
        desc.AxisCount = static_cast<UINT>(axes.size());
        desc.Axes = axes.data();
        desc.NormalizeVariance = true;
        desc.Epsilon = options->epsilon;
        mGraphBuilder->CreateOperator(DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1, &desc);

        mGraphBuilder->AddNodes({inputNode, inputNodes[0], inputNodes[1]});
        auto outputNode = mGraphBuilder->CreateNode(outputTensorDesc);

        // Transpose output from nchw->nhwc.
        if (options->layout == wnn::InputOperandLayout::Nhwc) {
            if (TransposeOutputToNhwc(outputNode, newOutputDims).IsError()) {
                return DAWN_INTERNAL_ERROR("Failed to transpose output from Nchw to Nhwc.");
            };
        }

        mGraphNodesMap[instanceNorm->PrimaryOutput()] = outputNode;
        return {};
    }

    MaybeError Graph::AddConcat(const op::Concat* concat) {
        DAWN_ASSERT(concat->Inputs().size() >= 1);
        auto inputsOperand = concat->Inputs();
        std::vector<std::shared_ptr<NodeBase>> inputNodes;
        std::shared_ptr<NodeBase> primaryNode = mGraphNodesMap[inputsOperand[0].Get()];
        auto primaryDims = ConvertDimensions(inputsOperand[0].Get()->Shape());

        std::vector<DML_TENSOR_DESC> inputTensorsDesc;
        for (auto& inputOperand : inputsOperand) {
            DAWN_ASSERT(mGraphNodesMap.find(inputOperand.Get()) != mGraphNodesMap.end());
            auto inputNode = mGraphNodesMap[inputOperand.Get()];
            auto inputDims = ConvertDimensions(inputOperand.Get()->Shape());
            inputNodes.push_back(inputNode);

            // Expand dimensions to DML_TENSOR_DIMENSION_COUNT_MAX if needed.
            if (inputDims.size() < DML_TENSOR_DIMENSION_COUNT_MAX) {
                auto newInputDims = ExpandDimensions(inputDims, DML_TENSOR_DIMENSION_COUNT_MAX);
                auto newInputStrides = CalculateStridesForBroadcast(inputDims, newInputDims,
                                                                    inputNode->outputTensorDesc);
                DML_TENSOR_DESC inputTensorDesc;
                DAWN_INVALID_IF(CreateDmlTensorDesc(inputTensorDesc, &inputNode->outputTensorDesc,
                                                    newInputDims, newInputStrides)
                                    .IsError(),
                                "Failed to create DML_TENSOR_DESC.");
                inputTensorsDesc.push_back(inputTensorDesc);
            } else if (inputDims.size() == DML_TENSOR_DIMENSION_COUNT_MAX) {
                inputTensorsDesc.push_back(inputNode->outputTensorDesc);
            } else {
                return DAWN_INTERNAL_ERROR("The size of input dimensions is greater than max");
            }
        }

        auto outputDims = ConvertDimensions(concat->Outputs()[0].Get()->Shape());
        auto newOutputDims = outputDims;
        if (outputDims.size() < DML_TENSOR_DIMENSION_COUNT_MAX) {
            newOutputDims = ExpandDimensions(outputDims, DML_TENSOR_DIMENSION_COUNT_MAX);
        }
        DML_TENSOR_DESC outputTensorDesc;
        DAWN_INVALID_IF(CreateDmlTensorDesc(outputTensorDesc, &primaryNode->outputTensorDesc,
                                            newOutputDims, {}, true)
                            .IsError(),
                        "Failed to create DML_TENSOR_DESC.");

        // Update the axis to align with the DML_TENSOR_DIMENSION_COUNT_MAX.
        uint32_t axis = concat->GetAxis();
        axis += DML_TENSOR_DIMENSION_COUNT_MAX - primaryDims.size();

        DML_JOIN_OPERATOR_DESC desc = {};
        desc.Axis = axis;
        desc.InputCount = static_cast<uint32_t>(inputTensorsDesc.size());
        desc.InputTensors = inputTensorsDesc.data();
        desc.OutputTensor = &outputTensorDesc;
        mGraphBuilder->CreateOperator(DML_OPERATOR_JOIN, &desc);

        // Reshape back according to output rank if needed to update the output node.
        if (outputDims.size() < newOutputDims.size()) {
            DAWN_INVALID_IF(
                CreateDmlTensorDesc(outputTensorDesc, &primaryNode->outputTensorDesc, outputDims)
                    .IsError(),
                "Failed to create DML_TENSOR_DESC.");
        }

        mGraphBuilder->AddNodes({inputNodes});
        mGraphNodesMap[concat->PrimaryOutput()] = mGraphBuilder->CreateNode(outputTensorDesc);
        return {};
    }

    MaybeError Graph::AddGemm(const op::Gemm* gemm) {
        auto inputsOperand = gemm->Inputs();
        DAWN_ASSERT(inputsOperand.size() == 2 || inputsOperand.size() == 3);
        DAWN_ASSERT(mGraphNodesMap.find(inputsOperand[0].Get()) != mGraphNodesMap.end());
        auto aNode = mGraphNodesMap[inputsOperand[0].Get()];
        auto aDims = ConvertDimensions(inputsOperand[0].Get()->Shape());
        DAWN_ASSERT(mGraphNodesMap.find(inputsOperand[1].Get()) != mGraphNodesMap.end());
        auto bNode = mGraphNodesMap[inputsOperand[1].Get()];
        auto bDims = ConvertDimensions(inputsOperand[1].Get()->Shape());
        auto outputDims = ConvertDimensions(gemm->Outputs()[0].Get()->Shape());
        std::vector<std::shared_ptr<NodeBase>> inputNodes = {aNode, bNode};

        // The shape of a tensor is 2D definited in WebNN Spec, but DML only support 4D,
        // so expand dimensions to 4D.
        DAWN_ASSERT(aDims.size() == 2);
        aDims = ExpandDimensions(aDims, 4);
        DML_TENSOR_DESC aTensorDesc;
        DAWN_INVALID_IF(CreateDmlTensorDesc(aTensorDesc, &aNode->outputTensorDesc, aDims).IsError(),
                        "Failed to create DML_TENSOR_DESC.");

        DAWN_ASSERT(bDims.size() == 2);
        bDims = ExpandDimensions(bDims, 4);
        DML_TENSOR_DESC bTensorDesc;
        DAWN_INVALID_IF(CreateDmlTensorDesc(bTensorDesc, &bNode->outputTensorDesc, bDims).IsError(),
                        "Failed to create DML_TENSOR_DESC.");

        DAWN_ASSERT(outputDims.size() == 2);
        auto expandedOutputDims = ExpandDimensions(outputDims, 4);
        DML_TENSOR_DESC outputTensorDesc;
        DAWN_INVALID_IF(CreateDmlTensorDesc(outputTensorDesc, &aNode->outputTensorDesc,
                                            expandedOutputDims, {}, true)
                            .IsError(),
                        "Failed to create DML_TENSOR_DESC.");

        // The operand c is optional.
        DML_TENSOR_DESC* cTensorDescPtr = nullptr;
        DML_TENSOR_DESC cTensorDesc;
        if (inputsOperand.size() == 3) {
            DAWN_ASSERT(mGraphNodesMap.find(inputsOperand[2].Get()) != mGraphNodesMap.end());
            auto cNode = mGraphNodesMap[inputsOperand[2].Get()];
            auto cDims = ConvertDimensions(inputsOperand[2].Get()->Shape());
            // It is either a scalar, or of the shape that is unidirectionally broadcastable to
            // the shape [M, N] definited in WebNN Spec, DML only support 4D, so broadCast the
            // Shape of optional C to {1, 1, M, N } supported in DML.
            auto cNewDims = expandedOutputDims;
            auto cNewStrides =
                CalculateStridesForBroadcast(cDims, cNewDims, cNode->outputTensorDesc);
            DAWN_INVALID_IF(
                CreateDmlTensorDesc(cTensorDesc, &cNode->outputTensorDesc, cNewDims, cNewStrides)
                    .IsError(),
                "Failed to create DML_TENSOR_DESC.");
            cTensorDescPtr = &cTensorDesc;
            inputNodes.push_back(cNode);
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
        mGraphBuilder->CreateOperator(DML_OPERATOR_GEMM, &desc);
        // Reshape back according to output rank if needed to update the output node.
        if (outputDims.size() < expandedOutputDims.size()) {
            DAWN_INVALID_IF(CreateDmlTensorDesc(outputTensorDesc, &aNode->outputTensorDesc,
                                                outputDims, {}, true)
                                .IsError(),
                            "Failed to create DML_TENSOR_DESC.");
        }

        mGraphBuilder->AddNodes({inputNodes});
        mGraphNodesMap[gemm->PrimaryOutput()] = mGraphBuilder->CreateNode(outputTensorDesc);
        return {};
    }

    MaybeError Graph::AddOutput(std::string_view name, const OperandBase* output) {
        DAWN_ASSERT(mGraphNodesMap.find(output) != mGraphNodesMap.end());
        auto outputNode = mGraphNodesMap[output];
        DAWN_ASSERT(outputNode != nullptr);

        const DML_BUFFER_TENSOR_DESC* bufferDesc =
            reinterpret_cast<const DML_BUFFER_TENSOR_DESC*>(outputNode->outputTensorDesc.Desc);
        DAWN_ASSERT(bufferDesc != nullptr);
        auto strides = bufferDesc->Strides;

        // Append identity to avoid directly using graph input as output, and avoid lack of
        // considering the impacts of strides if there are.
        if (outputNode->type == NodeType::ConstantInput ||
            outputNode->type == NodeType::NonConstantInput || strides != nullptr) {
            auto node = outputNode;
            DML_TENSOR_DESC outputTensorDesc;
            DAWN_INVALID_IF(
                AppendIdentity(outputTensorDesc, outputNode->outputTensorDesc).IsError(),
                "Failed to append identity.");
            mGraphBuilder->AddNodes({node});
            outputNode = mGraphBuilder->CreateNode(outputTensorDesc);
        }
        outputNode->name = name;
        mGraphBuilder->SetGraphOutput(outputNode, mOutputs.size());
        mOutputs.push_back(*reinterpret_cast<Node*>(outputNode.get()));
        return {};
    }

    MaybeError Graph::Finish() {
        if (mInputs.empty()) {
            return DAWN_VALIDATION_ERROR("Model inputs must be set.");
        }
        return {};
    }

    MaybeError Graph::CompileImpl() {
        DML_GRAPH_DESC graphDesc = mGraphBuilder->GetGraphDesc(mInputs.size(), mOutputs.size());
        // Compiles a graph of DirectML operators into an object that can be dispatched to the
        // GPU.
        ComPtr<IDMLDevice1> device1;
        IDMLDevice* device = mDevice->GetIDMLDevice();
        DAWN_INVALID_IF(FAILED(device->QueryInterface(IID_PPV_ARGS(&device1))),
                        "Failed to query interface");
        DAWN_INVALID_IF(FAILED(device1->CompileGraph(&graphDesc, DML_EXECUTION_FLAG_NONE,
                                                     IID_PPV_ARGS(&mCompiledGraph))),
                        "Failed to comile graph.");
        mGraphBuilder.reset(nullptr);
        DAWN_INVALID_IF(FAILED(mDevice->InitializeGraph(mCompiledGraph, mInputs, mOutputs)),
                        "Failed to initialize graph.");
        return {};
    }

    MaybeError Graph::ComputeImpl(NamedInputsBase* inputs, NamedOutputsBase* outputs) {
        DAWN_ASSERT(outputs != nullptr);
        auto namedInputs = inputs->GetRecords();
        auto namedOutputs = outputs->GetRecords();
        for (auto& input : mInputs) {
            // All the inputs must be set.
            if (input->type == NodeType::NonConstantInput &&
                namedInputs.find(input->name) == namedInputs.end()) {
                return DAWN_INTERNAL_ERROR("The input must be set.");
            }
        }

        DAWN_INVALID_IF(FAILED(mDevice->ExecuteGraph(mCompiledGraph, mInputs, mOutputs, namedInputs,
                                                     namedOutputs)),
                        "Failed to execute graph.");
        return {};
    }

}  // namespace webnn::native::dml
