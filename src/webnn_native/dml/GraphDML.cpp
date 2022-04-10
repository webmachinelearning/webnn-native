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
#include <numeric>

#include "common/Assert.h"
#include "common/Log.h"
#include "webnn_native/ErrorData.h"
#include "webnn_native/NamedInputs.h"
#include "webnn_native/NamedOutputs.h"
#include "webnn_native/Utils.h"
#include "webnn_native/dml/ContextDML.h"

namespace webnn_native::dml {

    namespace {
        enum TransposeType { NhwcToNchw, NchwToNhwc };

        uint32_t SizeOfShape(::dml::TensorDimensions& dims) {
            uint32_t prod = 1;
            for (size_t i = 0; i < dims.size(); ++i)
                prod *= dims[i];
            return prod;
        }

        bool CheckShape(const ::dml::Expression& expression,
                        const OperatorBase* operatorBase,
                        size_t index = 0) {
            DAWN_ASSERT(index < operatorBase->Outputs().size());
            auto expectedShape = operatorBase->Outputs()[index]->Shape();
            ::dml::TensorDimensions dmlShape = expression.GetOutputDesc().sizes;
            // Shape {1} equals to shape {} for a scalar.
            if (dmlShape == std::vector<uint32_t>{1} && expectedShape.size() == 0) {
                return true;
            }
            if (expectedShape.size() != dmlShape.size()) {
                dawn::ErrorLog() << "The size of output shape is expected as "
                                 << expectedShape.size() << ", but got " << dmlShape.size();
                return false;
            }
            for (size_t i = 0; i < dmlShape.size(); ++i) {
                if (expectedShape[i] < 0 || static_cast<size_t>(expectedShape[i]) != dmlShape[i]) {
                    dawn::ErrorLog() << "The output shape at index " << i << " is expected as "
                                     << expectedShape[i] << ", but got " << dmlShape[i];
                    return false;
                }
            }
            return true;
        }

        bool CheckShape(const std::vector<::dml::Expression>& expressions,
                        const OperatorBase* operatorBase) {
            DAWN_ASSERT(expressions.size() == operatorBase->Outputs().size());
            for (size_t i = 0; i < expressions.size(); ++i) {
                if (!CheckShape(expressions[i], operatorBase, i)) {
                    return false;
                }
            }
            return true;
        }

        bool GetDmlTensorDataType(wnn::OperandType operandType,
                                  DML_TENSOR_DATA_TYPE& dmlTensorDataType) {
            if (operandType == wnn::OperandType::Float32) {
                dmlTensorDataType = DML_TENSOR_DATA_TYPE_FLOAT32;
            } else if (operandType == wnn::OperandType::Float16) {
                dmlTensorDataType = DML_TENSOR_DATA_TYPE_FLOAT16;
            } else if (operandType == wnn::OperandType::Int32) {
                dmlTensorDataType = DML_TENSOR_DATA_TYPE_INT32;
            } else if (operandType == wnn::OperandType::Uint32) {
                dmlTensorDataType = DML_TENSOR_DATA_TYPE_UINT32;
            } else {
                return false;
            }
            return true;
        }

        bool GetDmlTensorDimensions(int32_t const* dimensions,
                                    uint32_t dimensionsCount,
                                    ::dml::TensorDimensions& dmlTensorDimensions) {
            if (dimensionsCount > DML_TENSOR_DIMENSION_COUNT_MAX) {
                dawn::ErrorLog() << "Tensor dimension count " << dimensionsCount
                                 << " is greater than DML_TENSOR_DIMENSION_COUNT_MAX "
                                 << DML_TENSOR_DIMENSION_COUNT_MAX;
                return false;
            }
            // for scale
            if (dimensionsCount == 0) {
                dmlTensorDimensions.resize(1);
                dmlTensorDimensions[0] = 1;
            } else {
                dmlTensorDimensions.resize(dimensionsCount);
                for (uint32_t i = 0; i < dimensionsCount; ++i) {
                    int32_t d = dimensions[i];
                    if (d < 0) {
                        dawn::ErrorLog() << "DML doesn't support the negative dimension value";
                        return false;
                    }
                    dmlTensorDimensions[i] = d;
                }
            }
            return true;
        }

        ::dml::TensorDimensions ExpandDimensions(const ::dml::TensorDimensions& dims, size_t rank) {
            DAWN_ASSERT(rank >= dims.size());
            ::dml::TensorDimensions newDims(rank, 1);
            for (size_t i = 0; i < dims.size(); ++i) {
                newDims[newDims.size() - i - 1] = dims[dims.size() - i - 1];
            }
            return newDims;
        }

        ::dml::TensorDimensions ShrinkDimensions(const ::dml::TensorDimensions& dims, size_t rank) {
            DAWN_ASSERT(rank <= dims.size());
            ::dml::TensorDimensions newDims(rank);
            for (size_t i = 0; i < rank; ++i) {
                newDims[newDims.size() - i - 1] = dims[dims.size() - i - 1];
            }
            return newDims;
        }

        // Strides are used to express broadcasting (by specifying a stride of 0) as well as
        // padding. If Strides is not specified, each dimension in the tensor is considered to
        // be contiguously packed, with no additional padding. The calculated strides refer to
        // https://docs.microsoft.com/en-us/windows/win32/direct3d12/dml-helper-functions#calculatestrides
        ::dml::TensorDimensions CalculateBroadcastStrides(::dml::TensorDimensions dims,
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
            ::dml::TensorDimensions strides(rank);
            strides[rank - 1] = broadcast[rank - 1] ? 0 : 1;
            size_t elements = 1;
            for (size_t i = 1; i < rank; i++) {
                size_t j = dims.size() - i - 1;
                elements *= dims[j + 1];
                strides[j] = broadcast[j] ? 0 : elements;
            }
            return strides;
        }

        ::dml::TensorDimensions CalculateFilterLayoutStrides(
            wnn::Conv2dFilterOperandLayout filterLayout,
            ::dml::TensorDimensions sizes) {
            uint32_t hStride = 0, wStride = 0, iStride = 0, oStride = 0;
            switch (filterLayout) {
                case wnn::Conv2dFilterOperandLayout::Hwio:
                    hStride = sizes[1] * sizes[2] * sizes[3];
                    wStride = sizes[2] * sizes[3];
                    iStride = sizes[3];
                    oStride = 1;
                    break;
                case wnn::Conv2dFilterOperandLayout::Ohwi:
                    oStride = sizes[1] * sizes[2] * sizes[3];
                    hStride = sizes[2] * sizes[3];
                    wStride = sizes[3];
                    iStride = 1;
                    break;
                case wnn::Conv2dFilterOperandLayout::Ihwo:
                    iStride = sizes[1] * sizes[2] * sizes[3];
                    hStride = sizes[2] * sizes[3];
                    wStride = sizes[3];
                    oStride = 1;
                    break;
                default:
                    DAWN_ASSERT(0);
                    break;
            }
            return {oStride, iStride, hStride, wStride};
        }

        ::dml::Expression ReinterpretFilterLayoutAsOihw(wnn::Conv2dFilterOperandLayout filterLayout,
                                                        ::dml::Expression filter) {
            ::dml::TensorDimensions filterDims = filter.GetOutputDesc().sizes;
            ::dml::TensorDimensions newFilterDims;
            newFilterDims.resize(4);
            switch (filterLayout) {
                case wnn::Conv2dFilterOperandLayout::Ohwi:
                    newFilterDims.resize(4);
                    newFilterDims[0] = filterDims[0];
                    newFilterDims[1] = filterDims[3];
                    newFilterDims[2] = filterDims[1];
                    newFilterDims[3] = filterDims[2];
                    filter =
                        ::dml::Reinterpret(filter, newFilterDims,
                                           CalculateFilterLayoutStrides(
                                               wnn::Conv2dFilterOperandLayout::Ohwi, filterDims));
                    break;
                case wnn::Conv2dFilterOperandLayout::Hwio:
                    newFilterDims[0] = filterDims[3];
                    newFilterDims[1] = filterDims[2];
                    newFilterDims[2] = filterDims[0];
                    newFilterDims[3] = filterDims[1];
                    filter =
                        ::dml::Reinterpret(filter, newFilterDims,
                                           CalculateFilterLayoutStrides(
                                               wnn::Conv2dFilterOperandLayout::Hwio, filterDims));
                    break;
                case wnn::Conv2dFilterOperandLayout::Ihwo:
                    newFilterDims[0] = filterDims[3];
                    newFilterDims[1] = filterDims[0];
                    newFilterDims[2] = filterDims[1];
                    newFilterDims[3] = filterDims[2];
                    filter =
                        ::dml::Reinterpret(filter, newFilterDims,
                                           CalculateFilterLayoutStrides(
                                               wnn::Conv2dFilterOperandLayout::Ihwo, filterDims));
                    break;
                default:
                    DAWN_ASSERT(0);
                    break;
            }
            return filter;
        }

        ::dml::TensorDimensions CalculateInputLayoutStrides(TransposeType transposeType,
                                                            ::dml::TensorDimensions sizes) {
            uint32_t nStride = 0, cStride = 0, hStride = 0, wStride = 0;
            switch (transposeType) {
                case NhwcToNchw:
                    nStride = sizes[1] * sizes[2] * sizes[3];
                    hStride = sizes[2] * sizes[3];
                    wStride = sizes[3];
                    cStride = 1;
                    return {nStride, cStride, hStride, wStride};
                case NchwToNhwc:
                    nStride = sizes[1] * sizes[2] * sizes[3];
                    cStride = sizes[2] * sizes[3];
                    hStride = sizes[3];
                    wStride = 1;
                    return {nStride, hStride, wStride, cStride};
                default:
                    DAWN_ASSERT(0);
                    break;
            }
        }

        ::dml::Expression ReinterpretInputLayout(TransposeType transposeType,
                                                 ::dml::Expression input) {
            ::dml::TensorDimensions inputDims = input.GetOutputDesc().sizes;
            ::dml::TensorDimensions newInputDims;
            newInputDims.resize(4);
            switch (transposeType) {
                case NhwcToNchw:
                    newInputDims[0] = inputDims[0];
                    newInputDims[1] = inputDims[3];
                    newInputDims[2] = inputDims[1];
                    newInputDims[3] = inputDims[2];
                    input = ::dml::Reinterpret(input, newInputDims,
                                               CalculateInputLayoutStrides(NhwcToNchw, inputDims));
                    break;
                case NchwToNhwc:
                    newInputDims.resize(4);
                    newInputDims[0] = inputDims[0];
                    newInputDims[1] = inputDims[2];
                    newInputDims[2] = inputDims[3];
                    newInputDims[3] = inputDims[1];
                    input = ::dml::Reinterpret(input, newInputDims,
                                               CalculateInputLayoutStrides(NchwToNhwc, inputDims));
                    break;
                default:
                    DAWN_ASSERT(0);
                    break;
            }
            return input;
        }

        bool BroadcastDimensions(const ::dml::TensorDimensions& aDims,
                                 const ::dml::TensorDimensions& bDims,
                                 bool& aBroadcasted,
                                 ::dml::TensorDimensions& aNewDims,
                                 ::dml::TensorDimensions& aNewStrides,
                                 bool& bBroadcasted,
                                 ::dml::TensorDimensions& bNewDims,
                                 ::dml::TensorDimensions& bNewStrides,
                                 size_t skipAxis = 0) {
            auto aRank = aDims.size();
            auto bRank = bDims.size();
            auto newRank = std::max(aRank, bRank);
            aNewDims.resize(newRank);
            aNewStrides.resize(newRank);
            std::vector<bool> aBroadcast(newRank, false);
            bNewDims.resize(newRank);
            bNewStrides.resize(newRank);
            std::vector<bool> bBroadcast(newRank, false);
            if (newRank > aRank) {
                aNewDims = ExpandDimensions(aDims, newRank);
                aBroadcasted = true;
            } else {
                aNewDims = aDims;
            }
            if (newRank > bRank) {
                bNewDims = ExpandDimensions(bDims, newRank);
                bBroadcasted = true;
            } else {
                bNewDims = bDims;
            }
            for (size_t i = 0; i < newRank - skipAxis; i++) {
                if (aNewDims[i] == 1 && bNewDims[i] != 1) {
                    aNewDims[i] = bNewDims[i];
                    aBroadcast[i] = true;
                    aBroadcasted = true;
                } else if (bNewDims[i] == 1 && aNewDims[i] != 1) {
                    bNewDims[i] = aNewDims[i];
                    bBroadcast[i] = true;
                    bBroadcasted = true;
                } else if (aNewDims[i] != bNewDims[i]) {
                    return false;
                }
            }
            aNewStrides = CalculateBroadcastStrides(aNewDims, aBroadcast);
            bNewStrides = CalculateBroadcastStrides(bNewDims, bBroadcast);
            return true;
        }

        DML_RECURRENT_NETWORK_DIRECTION getRecurrentSequenceDirection(
            wnn::RecurrentNetworkDirection direction) {
            DML_RECURRENT_NETWORK_DIRECTION dml_direction;
            switch (direction) {
                case wnn::RecurrentNetworkDirection::Forward:
                    dml_direction = DML_RECURRENT_NETWORK_DIRECTION_FORWARD;
                    break;
                case wnn::RecurrentNetworkDirection::Backward:
                    dml_direction = DML_RECURRENT_NETWORK_DIRECTION_BACKWARD;
                    break;
                case wnn::RecurrentNetworkDirection::Both:
                    dml_direction = DML_RECURRENT_NETWORK_DIRECTION_BIDIRECTIONAL;
                    break;
                default:
                    assert(0);
            }
            return dml_direction;
        }

        std::string OpTypeToString(op::BinaryOpType type) {
            if (type == op::BinaryOpType::kAdd) {
                return "add";
            } else if (type == op::BinaryOpType::kMul) {
                return "mul";
            } else if (type == op::BinaryOpType::kSub) {
                return "sub";
            } else if (type == op::BinaryOpType::kDiv) {
                return "div";
            } else if (type == op::BinaryOpType::kMatMul) {
                return "matmul";
            }
            return std::to_string(type);
        }

        std::string OpTypeToString(op::UnaryOpType type) {
            if (type == op::UnaryOpType::kRelu) {
                return "relu";
            } else if (type == op::UnaryOpType::kSoftmax) {
                return "softmax";
            } else if (type == op::UnaryOpType::kSigmoid) {
                return "sigmoid";
            } else if (type == op::UnaryOpType::kTanh) {
                return "tanh";
            }
            return std::to_string(type);
        }

        template <typename T>
        std::vector<const uint32_t> ImplicitPadding(const T* options,
                                                    ::dml::Expression input,
                                                    std::vector<uint32_t> filterSize) {
            ::dml::Span<const uint32_t> strides(reinterpret_cast<const uint32_t*>(options->strides),
                                                options->stridesCount);
            ::dml::Span<const uint32_t> dilations(
                reinterpret_cast<const uint32_t*>(options->dilations), options->dilationsCount);
            ::dml::TensorDimensions inputDims = input.GetOutputDesc().sizes;
            // {paddingTop, paddingBottom, paddingLeft, paddingRight}
            int32_t paddingTop, paddingBottom, paddingLeft, paddingRight;
            utils::ComputeImplicitPaddingForAutoPad(options->autoPad, dilations[0], inputDims[2],
                                                    filterSize[0], strides[0], paddingTop,
                                                    paddingBottom);
            utils::ComputeImplicitPaddingForAutoPad(options->autoPad, dilations[1], inputDims[3],
                                                    filterSize[1], strides[1], paddingLeft,
                                                    paddingRight);
            return {static_cast<const uint32_t>(paddingTop),
                    static_cast<const uint32_t>(paddingBottom),
                    static_cast<const uint32_t>(paddingLeft),
                    static_cast<const uint32_t>(paddingRight)};
        }

        template <typename T>
        std::vector<const uint32_t> ImplicitPadding(const T* options,
                                                    ::dml::Expression input,
                                                    ::dml::Expression filter) {
            ::dml::TensorDimensions filterDims = filter.GetOutputDesc().sizes;
            return ImplicitPadding(options, input, {filterDims[2], filterDims[3]});
        }

        template <typename T>
        std::vector<const uint32_t> ExplicitPadding(const T* options) {
            uint32_t paddingTop = static_cast<uint32_t>(options->padding[0]);
            uint32_t paddingBottom = static_cast<uint32_t>(options->padding[1]);
            uint32_t paddingLeft = static_cast<uint32_t>(options->padding[2]);
            uint32_t paddingRight = static_cast<uint32_t>(options->padding[3]);

            return {static_cast<const uint32_t>(paddingTop),
                    static_cast<const uint32_t>(paddingBottom),
                    static_cast<const uint32_t>(paddingLeft),
                    static_cast<const uint32_t>(paddingRight)};
        }

    }  // namespace

    std::string DmlTensorDimensionsToString(const ::dml::TensorDimensions& dimensions) {
        std::string output = "[";
        for (size_t i = 0; i < dimensions.size(); ++i) {
            output.append(std::to_string(dimensions[i]));
            if (i != dimensions.size() - 1) {
                output.append(",");
            }
        }
        output.append("]");
        return output;
    }

    template <typename T>
    std::string DmlSpanToString(const ::dml::Span<T>& span) {
        std::string output = "[";
        for (size_t i = 0; i < span.size(); ++i) {
            output.append(std::to_string(span[i]));
            if (i != span.size() - 1) {
                output.append(",");
            }
        }
        output.append("]");
        return output;
    }

    std::string DmlTensorDataTypeToString(DML_TENSOR_DATA_TYPE type) {
        if (type == DML_TENSOR_DATA_TYPE_UNKNOWN) {
            return "UNKNOWN";
        } else if (type == DML_TENSOR_DATA_TYPE_FLOAT32) {
            return "FLOAT32";
        } else if (type == DML_TENSOR_DATA_TYPE_FLOAT16) {
            return "FLOAT16";
        } else if (type == DML_TENSOR_DATA_TYPE_UINT32) {
            return "UINT32";
        } else if (type == DML_TENSOR_DATA_TYPE_UINT16) {
            return "UINT16";
        } else if (type == DML_TENSOR_DATA_TYPE_UINT8) {
            return "UINT8";
        } else if (type == DML_TENSOR_DATA_TYPE_INT32) {
            return "INT32";
        } else if (type == DML_TENSOR_DATA_TYPE_INT16) {
            return "INT16";
        } else if (type == DML_TENSOR_DATA_TYPE_INT8) {
            return "INT8";
        } else if (type == DML_TENSOR_DATA_TYPE_FLOAT64) {
            return "FLOAT64";
        } else if (type == DML_TENSOR_DATA_TYPE_UINT64) {
            return "UINT64";
        } else if (type == DML_TENSOR_DATA_TYPE_INT64) {
            return "INT64";
        }
        return std::to_string(type);
    }

    Graph::Graph(Context* context) : GraphBase(context) {
#ifdef WEBNN_ENABLE_GPU_BUFFER
        mDevice.reset(new ::pydml::Device(context->GetWGPUDevice()));
#else
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
#    if defined(_DEBUG)
        mDevice.reset(new ::pydml::Device(useGpu, true, gpuPreference));
#    else
        mDevice.reset(new ::pydml::Device(useGpu, false, gpuPreference));
#    endif
#endif
        mDevice->Init();
        mGraph.reset(new ::dml::Graph(mDevice->GetDevice()));
    }

    ::dml::Expression Graph::BindingConstant(DML_TENSOR_DATA_TYPE dmlTensorType,
                                             ::dml::TensorDimensions dmlTensorDims,
                                             void const* value,
                                             size_t size
#ifdef WEBNN_ENABLE_GPU_BUFFER
                                             ,
                                             WGPUBuffer wgpuBuffer
#endif
    ) {
        ::dml::TensorDesc dmlTensorDesc(dmlTensorType,
                                        ::DML_TENSOR_FLAGS::DML_TENSOR_FLAG_OWNED_BY_DML,
                                        dmlTensorDims, ::dml::TensorPolicy::Default());
        ::dml::Expression dmlConstant =
            ::dml::InputTensor(*mGraph, mInputBindings.size(), dmlTensorDesc);
        std::unique_ptr<::pydml::Binding> binding;
        // Input data is array buffer view.
        if (value != nullptr) {
#ifdef WEBNN_ENABLE_GPU_BUFFER
            UNREACHABLE();
#else
            std::unique_ptr<char> buffer(new char[size]);
            memcpy(buffer.get(), value, size);
            binding.reset(
                new ::pydml::Binding(dmlConstant, static_cast<void*>(buffer.get()), size));
            mConstantBuffers.push_back(std::move(buffer));
#endif
        } else {
#ifdef WEBNN_ENABLE_GPU_BUFFER
            binding.reset(new ::pydml::Binding(dmlConstant, wgpuBuffer, size, 0));
#endif
        }
        mInputBindings.push_back(std::move(binding));
        return dmlConstant;
    }

    MaybeError Graph::AddConstant(const op::Constant* constant) {
        const OperandDescriptor* desc = constant->GetOperandDescriptor();
        DML_TENSOR_DATA_TYPE dmlTensorType;
        if (!GetDmlTensorDataType(desc->type, dmlTensorType)) {
            return DAWN_INTERNAL_ERROR("Failed to get DML tensor type.");
        }
        ::dml::TensorDimensions dmlTensorDims;
        if (!GetDmlTensorDimensions(desc->dimensions, desc->dimensionsCount, dmlTensorDims)) {
            return DAWN_INTERNAL_ERROR("Failed to get DML tensor dimensions.");
        }

#ifdef WEBNN_ENABLE_GPU_BUFFER
        auto dmlConstant = BindingConstant(dmlTensorType, dmlTensorDims, nullptr,
                                           constant->GetByteLength(), constant->GetWGPUBuffer());
        Ref<OperandBase> constantOperand = AcquireRef<OperandBase>(constant->PrimaryOutput());
        constantOperand->Reference();
        mConstants.push_back(std::move(constantOperand));
#else
        auto dmlConstant = BindingConstant(dmlTensorType, dmlTensorDims, constant->GetBuffer(),
                                           constant->GetByteLength());
        mConstantSet.insert(constant->PrimaryOutput());
#endif
        mExpression.insert(std::make_pair(constant->PrimaryOutput(), dmlConstant));
        DAWN_ASSERT(CheckShape(dmlConstant, constant));
        return {};
    }

    ::dml::FusedActivation CreateFusedActivation(FusionOperatorBase* activation) {
        ::dml::FusedActivation dmlActivation = ::dml::FusedActivation::None();
        if (activation == nullptr) {
            return dmlActivation;
        }

        switch (activation->GetFusionType()) {
            case FusionType::Clamp:
            case FusionType::HardSwish:
                return dmlActivation;
            case FusionType::Relu:
                dmlActivation = ::dml::FusedActivation::Relu();
                break;
            case FusionType::Sigmoid:
                dmlActivation = ::dml::FusedActivation::Sigmoid();
                break;
            case FusionType::Tanh:
                dmlActivation = ::dml::FusedActivation::Tanh();
                break;
            case FusionType::LeakyRelu:
                dmlActivation = ::dml::FusedActivation::LeakyRelu(
                    reinterpret_cast<op::FusionLeakyRelu*>(activation)->GetAlpha());
                break;
            default:
                DAWN_ASSERT(0);
        }
        return dmlActivation;
    }

    ::dml::Expression Graph::EmulateFusedActivation(FusionOperatorBase* activation,
                                                    ::dml::Expression& input) {
        if (activation == nullptr) {
            return input;
        }
        // HardSwish and Clamp are not supported for fusion, so we add them directly to emulate.
        // Currently we implement Relu6 operator by Clamp.
        auto type = activation->GetFusionType();
        if (type == FusionType::HardSwish) {
            return HardSwish(input);
        } else if (type == FusionType::Clamp) {
            auto clamp = reinterpret_cast<const op::FusionClamp*>(activation);
            return ::dml::Clip(input, clamp->GetMinValue(), clamp->GetMaxValue());
        }
        return input;
    }

    MaybeError Graph::AddInput(const op::Input* input) {
        const OperandDescriptor* desc = input->GetOperandDescriptor();
        DML_TENSOR_DATA_TYPE dmlTensorType;
        if (!GetDmlTensorDataType(desc->type, dmlTensorType)) {
            return DAWN_INTERNAL_ERROR("Failed to get DML tensor type.");
        }
        ::dml::TensorDimensions dmlTensorDims;
        if (!GetDmlTensorDimensions(desc->dimensions, desc->dimensionsCount, dmlTensorDims)) {
            return DAWN_INTERNAL_ERROR("Failed to get DML tensor dimensions.");
        }
        ::dml::TensorDesc dmlTensorDesc(dmlTensorType, dmlTensorDims,
                                        ::dml::TensorPolicy::Default());
        ::dml::Expression dmlInput =
            ::dml::InputTensor(*mGraph, mInputBindings.size(), dmlTensorDesc);
        mExpression.insert(std::make_pair(input->PrimaryOutput(), dmlInput));
        std::unique_ptr<::pydml::Binding> binding(new ::pydml::Binding(dmlInput, nullptr, 0));
        mInputBindings.push_back(std::move(binding));
        mInputBindingMap.insert(std::make_pair(input->GetName(), mInputBindings.back().get()));
        DAWN_ASSERT(CheckShape(dmlInput, input));
        return {};
    }

    MaybeError Graph::AddOutput(std::string_view name, const OperandBase* output) {
        DAWN_ASSERT(mExpression.find(output) != mExpression.end());
        ::dml::Expression dmlOutput = mExpression.at(output);
        mOutputExpressionMap.insert(std::make_pair(name.data(), dmlOutput));
        std::unique_ptr<::pydml::Binding> binding(new ::pydml::Binding(dmlOutput, nullptr, 0));
        mOutputBindings.push_back(std::move(binding));
        mOutputBindingMap.insert(std::make_pair(name, mOutputBindings.back().get()));
        return {};
    }

    MaybeError Graph::AddBatchNorm(const op::BatchNorm* batchNorm) {
        auto inputs = batchNorm->Inputs();
        // input
        DAWN_ASSERT(inputs.size() == 3 || inputs.size() == 4 || inputs.size() == 5);
        DAWN_ASSERT(mExpression.find(batchNorm->Inputs()[0].Get()) != mExpression.end());
        ::dml::Expression input = mExpression.at(batchNorm->Inputs()[0].Get());
        const BatchNormOptions* options = batchNorm->GetOptions();
        // When input is a 4-D tensor of the "nchw" or "nhwc" layout, options.axis should be set to
        // 1 or 3 respectively.
        uint32_t axis = options->axis;
        if (options->axis == 3) {
            input = ReinterpretInputLayout(NhwcToNchw, input);
            axis = 1;
        }
        ::dml::TensorDimensions inputDims = input.GetOutputDesc().sizes;

        // Reshape 1D mean, variance, scale, bias to 4D with setting 1 to automatically broadcast.
        std::vector<::dml::Expression> expressions;
        expressions.reserve(inputs.size());
        for (size_t i = 1; i < inputs.size(); ++i) {
            DAWN_ASSERT(mExpression.find(batchNorm->Inputs()[i].Get()) != mExpression.end());
            ::dml::Expression expression = mExpression.at(batchNorm->Inputs()[i].Get());
            ::dml::TensorDimensions dimensions = expression.GetOutputDesc().sizes;
            DAWN_ASSERT(dimensions.size() == 1);
            if (dimensions[0] != inputDims[axis]) {
                return DAWN_INTERNAL_ERROR(
                    "The 1-D tensor of the values whose length size is not equal to the size of "
                    "the input dimension denoted by options.axis.");
            }
            // This tensor's dimensions should be { BatchCount, ChannelCount, Height,Width}.
            // Set 1 to automatically broadcast those dimensions across the input.
            ::dml::TensorDimensions expandDimens(4, 1);
            expandDimens[axis] = dimensions[0];
            expressions.push_back(::dml::Reinterpret(expression, expandDimens, ::dml::NullOpt));
        }
        // Set tensor's dimensions to {1, 1, 1, 1} if scale or bias is null.
        const DML_TENSOR_DATA_TYPE type = DML_TENSOR_DATA_TYPE_FLOAT32;
        if (options->scale == nullptr) {
            float scale = 1.0;
            expressions.insert(
                options->bias == nullptr ? expressions.end() : expressions.begin() + 2,
                BindingConstant(type, {1, 1, 1, 1}, &scale, sizeof(float)));
        }
        if (options->bias == nullptr) {
            float bias = 0;
            expressions.push_back(BindingConstant(type, {1, 1, 1, 1}, &bias, sizeof(float)));
        }
        ::dml::Expression output = ::dml::BatchNormalization(
            input, expressions[0], expressions[1], expressions[2], expressions[3], true,
            options->epsilon, CreateFusedActivation(options->activation));
        if (options->axis == 3) {
            output = ReinterpretInputLayout(NchwToNhwc, output);
        }
        output = EmulateFusedActivation(options->activation, output);
        mExpression.insert(std::make_pair(batchNorm->PrimaryOutput(), output));
        DAWN_ASSERT(CheckShape(output, batchNorm));
        return {};
    }

    MaybeError Graph::AddBinary(const op::Binary* binary) {
        DAWN_ASSERT(binary->Inputs().size() == 2);
        DAWN_ASSERT(mExpression.find(binary->Inputs()[0].Get()) != mExpression.end());
        ::dml::Expression a = mExpression.at(binary->Inputs()[0].Get());
        DAWN_ASSERT(mExpression.find(binary->Inputs()[1].Get()) != mExpression.end());
        ::dml::Expression b = mExpression.at(binary->Inputs()[1].Get());
        ::dml::Expression c;
        ::dml::TensorDimensions aDims = a.GetOutputDesc().sizes;
        const size_t aRank = aDims.size();
        ::dml::TensorDimensions bDims = b.GetOutputDesc().sizes;
        const size_t bRank = bDims.size();
        ::dml::TensorDimensions aNewDims, bNewDims;
        ::dml::TensorDimensions aNewStrides, bNewStrides;
        bool aDimsChanged = false, bDimsChanged = false;
        size_t cRank = 0;
        bool needBroadcast = false;
        size_t broadcastSkipAxis = 0;

        if (binary->GetType() == op::BinaryOpType::kMatMul) {
            // DML GEMM requires inputs are either 4D or 5D. We use 4D.
            if (aRank > 4 || bRank > 4) {
                return DAWN_INTERNAL_ERROR("The size of input dimensions is greater than 4.");
            }

            if (aRank == 1 && bRank == 1) {
                // If both a and b are 1-D, the operation is a vector dot-product,
                // which produces a scalar output.
                cRank = 1;
            } else {
                // The output is a N-D tensor whose rank is the maximum rank of the
                // input tensors.
                cRank = std::max(aRank, bRank);
            }

            if (aRank < 4) {
                aDims = ExpandDimensions(aDims, 4);
                aDimsChanged = true;
                aNewDims = aDims;
                aNewStrides = CalculateBroadcastStrides(aNewDims);
            }

            if (bRank < 4) {
                if (bRank == 1) {
                    // If b is 1-D, it is converted to a 2-D tensor by by appending a 1 to
                    // its dimensions.
                    bDims.push_back(1);
                }
                bDims = ExpandDimensions(bDims, 4);
                bDimsChanged = true;
                bNewDims = bDims;
                bNewStrides = CalculateBroadcastStrides(bNewDims);
            }

            if (aRank > 2 || bRank > 2) {
                // If either a or b is N-D, N > 2, it is treated as a stack of matrices
                // with dimensions corresponding to the last two indices. The matrix
                // multiplication will be broadcasted accordingly by following
                // [numpy-broadcasting-rule].
                needBroadcast = true;
                broadcastSkipAxis = 2;
            }
        } else {
            // The element-wise binary operation will be broadcasted according to
            // [numpy-broadcasting-rule].
            needBroadcast = true;
            broadcastSkipAxis = 0;
        }

        if (needBroadcast) {
            if (!BroadcastDimensions(aDims, bDims, aDimsChanged, aNewDims, aNewStrides,
                                     bDimsChanged, bNewDims, bNewStrides, broadcastSkipAxis)) {
                return DAWN_INTERNAL_ERROR("Failed to broadcast a and b.");
            }
        }

        if (aDimsChanged) {
            a = ::dml::Reinterpret(a, aNewDims, aNewStrides);
        }
        if (bDimsChanged) {
            b = ::dml::Reinterpret(b, bNewDims, bNewStrides);
        }

        if (binary->GetType() == op::BinaryOpType::kMatMul) {
            c = ::dml::Gemm(a, b);
        } else if (binary->GetType() == op::BinaryOpType::kAdd) {
            c = ::dml::Add(a, b);
        } else if (binary->GetType() == op::BinaryOpType::kDiv) {
            c = ::dml::Divide(a, b);
        } else if (binary->GetType() == op::BinaryOpType::kMul) {
            c = ::dml::Multiply(a, b);
        } else if (binary->GetType() == op::BinaryOpType::kSub) {
            c = ::dml::Subtract(a, b);
        } else if (binary->GetType() == op::BinaryOpType::kMax) {
            c = ::dml::Max(a, b);
        } else if (binary->GetType() == op::BinaryOpType::kMin) {
            c = ::dml::Min(a, b);
        } else if (binary->GetType() == op::BinaryOpType::kPower) {
            c = ::dml::Pow(a, b);
        } else {
            std::string errorMessage = std::string(" Binary op ") +
                                       OpTypeToString(binary->GetType()) +
                                       std::string(" is not implemented.");
            return DAWN_UNIMPLEMENTED_ERROR(errorMessage);
        }

        // Reshape back according to c rank if needed.
        ::dml::TensorDimensions cDims = c.GetOutputDesc().sizes;
        if (cRank != 0 && cRank < cDims.size()) {
            ::dml::TensorDimensions cNewDims = ShrinkDimensions(cDims, cRank);
            c = ::dml::Reinterpret(c, cNewDims, ::dml::NullOpt);
        }
        mExpression.insert(std::make_pair(binary->PrimaryOutput(), c));
        DAWN_ASSERT(CheckShape(c, binary));
        return {};
    }

    MaybeError Graph::AddConv2d(const op::Conv2d* conv2d) {
        auto inputsOperand = conv2d->Inputs();
        DAWN_ASSERT(inputsOperand.size() == 2 || inputsOperand.size() == 3);
        DAWN_ASSERT(mExpression.find(inputsOperand[0].Get()) != mExpression.end());
        ::dml::Expression input = mExpression.at(inputsOperand[0].Get());
        DAWN_ASSERT(mExpression.find(inputsOperand[1].Get()) != mExpression.end());
        ::dml::Expression filter = mExpression.at(inputsOperand[1].Get());
        const Conv2dOptions* options = conv2d->GetOptions();

        if (options->inputLayout == wnn::InputOperandLayout::Nhwc) {
            input = ReinterpretInputLayout(NhwcToNchw, input);
        }

        if (options->filterLayout != wnn::Conv2dFilterOperandLayout::Oihw) {
            filter = ReinterpretFilterLayoutAsOihw(options->filterLayout, filter);
        }

        // FIXME(nhu): strides, dilations, padding should be uint32_t
        // need to fix the spec.
        ::dml::Span<const uint32_t> strides(reinterpret_cast<const uint32_t*>(options->strides),
                                            options->stridesCount);
        ::dml::Span<const uint32_t> dilations(reinterpret_cast<const uint32_t*>(options->dilations),
                                              options->dilationsCount);

        auto padding = options->autoPad == wnn::AutoPad::Explicit
                           ? ExplicitPadding<Conv2dOptions>(options)
                           : ImplicitPadding<Conv2dOptions>(options, input, filter);
        // dml::Span just holds the refernces, need a variable to hold the memory.
        std::vector<const uint32_t> startPaddingVector = {padding[0], padding[2]};
        ::dml::Span<const uint32_t> startPadding(startPaddingVector);
        std::vector<const uint32_t> endPaddingVector = {padding[1], padding[3]};
        ::dml::Span<const uint32_t> endPadding(endPaddingVector);

        ::dml::Optional<::dml::Expression> bias = ::dml::NullOpt;
        if (options->bias != nullptr) {
            DAWN_ASSERT(mExpression.find(inputsOperand[2].Get()) != mExpression.end());
            bias = mExpression.at(inputsOperand[2].Get());
            ::dml::TensorDimensions biasDims = bias->GetOutputDesc().sizes;
            if (biasDims[0] != filter.GetOutputDesc().sizes[0] || biasDims.size() != 1) {
                return DAWN_INTERNAL_ERROR(
                    "The bias should be 1-D tensor with the shape of [output_channels].");
            }
            // Reshape bias from 1-D to 4-D for NCHW layout.
            ::dml::TensorDimensions expandDimens = {1, biasDims[0], 1, 1};
            bias = ::dml::Reinterpret(*bias, expandDimens, ::dml::NullOpt);
        }
        ::dml::Expression output = ::dml::Convolution(
            input, filter, bias, DML_CONVOLUTION_MODE_CROSS_CORRELATION,
            DML_CONVOLUTION_DIRECTION_FORWARD, strides, dilations, startPadding, endPadding,
            // outPadding
            {},
            // groupCount
            options->groups, CreateFusedActivation(options->activation));
        if (options->inputLayout == wnn::InputOperandLayout::Nhwc) {
            output = ::dml::Identity(ReinterpretInputLayout(NchwToNhwc, output));
        }
        output = EmulateFusedActivation(options->activation, output);
        mExpression.insert(std::make_pair(conv2d->PrimaryOutput(), output));
        DAWN_ASSERT(CheckShape(output, conv2d));
        return {};
    }

    MaybeError Graph::AddConvTranspose2d(const op::ConvTranspose2d* convTranspose2d) {
        return DAWN_UNIMPLEMENTED_ERROR("ConvTranspose2D has not been supported on DirectML.");
    }

    MaybeError Graph::AddPad(const op::Pad* pad) {
        auto inputsOperand = pad->Inputs();
        DAWN_ASSERT(mExpression.find(inputsOperand[0].Get()) != mExpression.end());
        ::dml::Expression input = mExpression.at(inputsOperand[0].Get());
        ::dml::TensorDimensions inputDims = input.GetOutputDesc().sizes;
        uint32_t inputRank = inputDims.size();
        // dml::Span just holds the refernces, need a variable to hold the memory.
        std::vector<uint32_t> startPaddingVector;
        std::vector<uint32_t> endPaddingVector;
        if (inputsOperand.size() == 2) {
            DAWN_ASSERT(mExpression.find(inputsOperand[1].Get()) != mExpression.end());
            if (mConstantSet.find(inputsOperand[1].Get()) == mConstantSet.end()) {
                return DAWN_INTERNAL_ERROR("The padding constant is not found.");
            }
            ::dml::Expression padding = mExpression.at(inputsOperand[1].Get());

            // Workaround(mingming): If padding was added in mGraph, it must be used.
            // Use "Pad_"+std::to_string(mExpression.size()) to generate a unique name for the
            // output node. This may be a dml issue:
            // https://github.com/microsoft/DirectML/issues/133.
            std::string name = "Pad_" + std::to_string(mExpression.size());
            mOutputExpressionMap[name] = ::dml::Identity(padding);

            const op::Constant* paddingConstant =
                reinterpret_cast<const op::Constant*>(inputsOperand[1]->Operator());
            ::dml::TensorDimensions paddingDims = padding.GetOutputDesc().sizes;
            if (paddingDims[1] != 2 || paddingDims[0] != inputRank) {
                return DAWN_INTERNAL_ERROR(
                    "The padding should has shape [n, 2], where n is the rank of the input tensor");
            }
            const uint32_t* paddingData =
                static_cast<const uint32_t*>(paddingConstant->GetBuffer());
            for (size_t i = 0; i < inputRank; ++i) {
                startPaddingVector.push_back(paddingData[2 * i]);
                endPaddingVector.push_back(paddingData[2 * i + 1]);
            }
        } else {
            for (size_t i = 0; i < inputRank; ++i) {
                startPaddingVector.push_back(pad->GetPadding()[2 * i]);
                endPaddingVector.push_back(pad->GetPadding()[2 * i + 1]);
            }
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
        float paddingValue = options->value;
        ::dml::Span<const uint32_t> startPadding(startPaddingVector);
        ::dml::Span<const uint32_t> endPadding(endPaddingVector);
        ::dml::Expression output =
            ::dml::Padding(input, paddingMode, paddingValue, startPadding, endPadding);
        mExpression.insert(std::make_pair(pad->PrimaryOutput(), output));
        DAWN_ASSERT(CheckShape(output, pad));
        return {};
    }

    //  DirectMlX.h hasn't supported setting correct outputSizes which need to match the specified
    //  roundingType, then pool2d always performs as specifying floor roundingType. Track this by
    //  issue: https://github.com/microsoft/DirectML/issues/205 and
    //  https://github.com/webmachinelearning/webnn-native/issues/217.
    MaybeError Graph::AddPool2d(const op::Pool2d* pool2d) {
        DAWN_ASSERT(pool2d->Inputs().size() == 1);
        const OperandBase* inputOperand = pool2d->Inputs()[0].Get();
        DAWN_ASSERT(mExpression.find(inputOperand) != mExpression.end());
        ::dml::Expression input = mExpression.at(inputOperand);
        const Pool2dOptions* options = pool2d->GetOptions();
        if (options->layout == wnn::InputOperandLayout::Nhwc) {
            input = ReinterpretInputLayout(NhwcToNchw, input);
        }
        ::dml::TensorDimensions inputDims = input.GetOutputDesc().sizes;

        ::dml::Span<const uint32_t> strides(reinterpret_cast<const uint32_t*>(options->strides),
                                            options->stridesCount);
        std::vector<std::uint32_t> windowSizesVector;
        if (options->windowDimensions != nullptr) {
            const uint32_t* windowDimensions =
                reinterpret_cast<const uint32_t*>(options->windowDimensions);
            windowSizesVector.assign(windowDimensions,
                                     windowDimensions + options->windowDimensionsCount);
        } else {
            windowSizesVector = {inputDims[2], inputDims[3]};
        }
        ::dml::Span<const uint32_t> windowSizes(windowSizesVector);
        ::dml::Span<const uint32_t> dilations(reinterpret_cast<const uint32_t*>(options->dilations),
                                              options->dilationsCount);
        auto padding = options->autoPad == wnn::AutoPad::Explicit
                           ? ExplicitPadding<Pool2dOptions>(options)
                           : ImplicitPadding<Pool2dOptions>(options, input, windowSizesVector);
        // dml::Span just holds the refernces, need a variable to hold the memory.
        std::vector<const uint32_t> startPaddingVector = {padding[0], padding[2]};
        ::dml::Span<const uint32_t> startPadding(startPaddingVector);
        std::vector<const uint32_t> endPaddingVector = {padding[1], padding[3]};
        ::dml::Span<const uint32_t> endPadding(endPaddingVector);
        ::dml::Expression output;
        auto outputSizes = pool2d->GetOutputSizes();
        ::dml::TensorDimensions outputShape = {inputDims[0], inputDims[1],
                                               static_cast<uint32_t>(outputSizes[0]),
                                               static_cast<uint32_t>(outputSizes[1])};
        if (pool2d->GetType() == op::Pool2dType::kAveragePool2d) {
            if (dilations[0] != 1 || dilations[1] != 1) {
                return DAWN_INTERNAL_ERROR("The dilations of average pool2d are not supported.");
            }
            output = ::dml::AveragePooling(input, strides, windowSizes, startPadding, endPadding,
                                           false, outputShape);
        }
        // L2Pool2d is not supported, emulate it by referring to
        // https://github.com/tensorflow/tfjs/issues/5539.
        else if (pool2d->GetType() == op::Pool2dType::kL2Pool2d) {
            uint32_t length = SizeOfShape(inputDims);
            std::vector<float> constant(length, 2);
            auto pow = ::dml::Pow(input, BindingConstant(DML_TENSOR_DATA_TYPE_FLOAT32, inputDims,
                                                         constant.data(), sizeof(float) * length));
            auto avgPool2d = ::dml::AveragePooling(pow, strides, windowSizes, startPadding,
                                                   endPadding, false, outputShape);
            output = ::dml::Sqrt(avgPool2d);
        } else if (pool2d->GetType() == op::Pool2dType::kMaxPool2d) {
            if (dilations[0] != 1 || dilations[1] != 1) {
                for (size_t i = 0; i < windowSizes.size(); ++i) {
                    uint32_t paddedInputSize = inputDims[2 + i] + startPadding[i] + endPadding[i];
                    uint32_t dilatedWindowSize = 1 + (windowSizes[i] - 1) * dilations[i];
                    outputShape[2 + i] =
                        (dilatedWindowSize >= paddedInputSize)
                            ? 1
                            : (paddedInputSize - dilatedWindowSize) / strides[i] + 1;
                }
            }
            output = ::dml::MaxPooling(input, windowSizes, strides, startPadding, endPadding,
                                       dilations, false, outputShape)
                         .values;
        } else {
            return DAWN_INTERNAL_ERROR("This pool2d type is not supported.");
        }

        if (options->layout == wnn::InputOperandLayout::Nhwc) {
            output = ::dml::Identity(ReinterpretInputLayout(NchwToNhwc, output));
        }
        mExpression.insert(std::make_pair(pool2d->PrimaryOutput(), output));
        DAWN_ASSERT(CheckShape(output, pool2d));
        return {};
    }

    MaybeError Graph::AddClamp(const op::Clamp* clamp) {
        auto inputsOperand = clamp->Inputs();
        DAWN_ASSERT(inputsOperand.size() == 1);
        ::dml::Expression input = mExpression.at(inputsOperand[0].Get());
        ::dml::TensorDimensions inputDims = input.GetOutputDesc().sizes;
        if (inputDims.size() > DML_TENSOR_DIMENSION_COUNT_MAX1) {
            return DAWN_INTERNAL_ERROR("The size of input dimensions is greater than max");
        }
        auto output = ::dml::Clip(input, clamp->GetMinValue(), clamp->GetMaxValue());
        mExpression.insert(std::make_pair(clamp->PrimaryOutput(), output));
        DAWN_ASSERT(CheckShape(output, clamp));
        return {};
    }

    MaybeError Graph::AddReduce(const op::Reduce* reduce) {
        DAWN_ASSERT(reduce->Inputs().size() == 1);
        const OperandBase* inputOperand = reduce->Inputs()[0].Get();
        DAWN_ASSERT(mExpression.find(inputOperand) != mExpression.end());
        ::dml::Expression input = mExpression.at(inputOperand);
        const ReduceOptions* options = reduce->GetOptions();
        std::vector<std::uint32_t> axesVector;
        size_t inputRank = input.GetOutputDesc().sizes.size();
        for (size_t i = 0; i < options->axesCount; ++i) {
            // Axes values must be in the range [0, InputTensor.DimensionCount - 1].
            // The dimensions to reduce where -1 means the last dimension.
            uint32_t axis = options->axes[i] == -1 ? inputRank - 1 : options->axes[i];
            axesVector.push_back(axis);
        }
        ::dml::Span<const uint32_t> axes(axesVector);
        ::dml::Expression output;
        switch (reduce->GetType()) {
            case op::ReduceType::kReduceL1:
                output = ::dml::Reduce(input, DML_REDUCE_FUNCTION_L1, axes);
                break;
            case op::ReduceType::kReduceL2:
                output = ::dml::Reduce(input, DML_REDUCE_FUNCTION_L2, axes);
                break;
            case op::ReduceType::kReduceMax:
                output = ::dml::Reduce(input, DML_REDUCE_FUNCTION_MAX, axes);
                break;
            case op::ReduceType::kReduceMean:
                output = ::dml::Reduce(input, DML_REDUCE_FUNCTION_AVERAGE, axes);
                break;
            case op::ReduceType::kReduceMin:
                output = ::dml::Reduce(input, DML_REDUCE_FUNCTION_MIN, axes);
                break;
            case op::ReduceType::kReduceProduct:
                output = ::dml::Reduce(input, DML_REDUCE_FUNCTION_MULTIPLY, axes);
                break;
            case op::ReduceType::kReduceSum:
                output = ::dml::Reduce(input, DML_REDUCE_FUNCTION_SUM, axes);
                break;
            case op::ReduceType::kReduceArgMax:
                output = ::dml::Reduce(input, DML_REDUCE_FUNCTION_ARGMAX, axes);
                break;
            case op::ReduceType::kReduceArgMin:
                output = ::dml::Reduce(input, DML_REDUCE_FUNCTION_ARGMIN, axes);
                break;
            default:
                return DAWN_INTERNAL_ERROR("The reduce op type isn't supported.");
        }
        ::dml::TensorDimensions outputDims = output.GetOutputDesc().sizes;
        if (!options->keepDimensions) {
            ::dml::TensorDimensions newDims;
            for (size_t i = 0; i < outputDims.size(); ++i) {
                // Reduce in DML always keep dimensions,
                // manually remove the reduced dimension whose value is 1.
                if (!(outputDims[i] == 1 && std::find(axes.begin(), axes.end(), i) != axes.end())) {
                    newDims.push_back(outputDims[i]);
                }
            }
            // DML doesn't support reinterpret a node for empty shape.
            if (newDims.empty()) {
                newDims.push_back(1);
            }
            output = ::dml::Reinterpret(output, newDims, ::dml::NullOpt);
        }
        mExpression.insert(std::make_pair(reduce->PrimaryOutput(), output));
        DAWN_ASSERT(CheckShape(output, reduce));
        return {};
    }

    MaybeError Graph::AddResample2d(const op::Resample2d* resample2d) {
        DAWN_ASSERT(resample2d->Inputs().size() == 1);
        const OperandBase* inputOperand = resample2d->Inputs()[0].Get();
        DAWN_ASSERT(mExpression.find(inputOperand) != mExpression.end());
        ::dml::Expression input = mExpression.at(inputOperand);
        ::dml::TensorDimensions inputDims = input.GetOutputDesc().sizes;
        const Resample2dOptions* options = resample2d->GetOptions();
        // axes.
        auto axes = resample2d->GetAxes();
        // size.
        auto outputShape = resample2d->GetOutputShape();
        ::dml::TensorDimensions outputSizes(outputShape.begin(), outputShape.end());

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

        // If not specified, parameters are defaulted to the following values:
        // Scales = computed by dividing the output sizes by the input sizes
        // InputPixelOffsets = 0.5f for each dimension
        // OutputPixelOffsets = -0.5f for each dimension
        ::dml::Expression output = ::dml::Resample(input, outputSizes, mode, {}, {}, {});
        mExpression.insert(std::make_pair(resample2d->PrimaryOutput(), output));
        DAWN_ASSERT(CheckShape(output, resample2d));
        return {};
    }

    MaybeError Graph::AddReshape(const op::Reshape* reshape) {
        DAWN_ASSERT(reshape->Inputs().size() == 1);
        const OperandBase* inputOperand = reshape->Inputs()[0].Get();
        DAWN_ASSERT(mExpression.find(inputOperand) != mExpression.end());
        ::dml::Expression input = mExpression.at(inputOperand);
        auto newShape = reshape->GetNewShape();
        if (newShape.size() > DML_TENSOR_DIMENSION_COUNT_MAX) {
            return DAWN_INTERNAL_ERROR("The size of new shape is not supported by DML.");
        }
        ::dml::TensorDimensions newSizes(newShape.size());
        uint32_t outputElementCount = 1;
        int32_t inferAxis = -1;

        ::dml::TensorDimensions inputDims = input.GetOutputDesc().sizes;
        uint32_t inputElementCount =
            std::accumulate(inputDims.begin(), inputDims.end(), 1, std::multiplies<uint32_t>());

        for (size_t i = 0; i < newShape.size(); ++i) {
            if (newShape[i] == -1) {
                if (inferAxis != -1) {
                    return DAWN_VALIDATION_ERROR("New shape should contain only one -1 value.");
                } else {
                    inferAxis = i;
                }
            } else if (newShape[i] <= 0) {
                return DAWN_VALIDATION_ERROR("Argument new shape is invalid");
            } else {
                newSizes[i] = newShape[i];
                outputElementCount *= newSizes[i];
            }
        }

        if (inferAxis != -1) {
            newSizes[inferAxis] = inputElementCount / outputElementCount;
        }

        ::dml::Expression output = ::dml::Reinterpret(input, newSizes, ::dml::NullOpt);
        mExpression.insert(std::make_pair(reshape->PrimaryOutput(), output));
        DAWN_ASSERT(CheckShape(output, reshape));
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
        DAWN_ASSERT(mExpression.find(inputOperand) != mExpression.end());
        ::dml::Expression input = mExpression.at(inputOperand);
        ::dml::TensorDimensions inputDims = input.GetOutputDesc().sizes;

        // dml::Span just holds the refernces, need a variable to hold the memory.
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
        std::vector<int32_t> inputWindwStrides(inputDims.size(), 1);
        ::dml::Expression output =
            ::dml::Slice(input, ::dml::Span<const uint32_t>(inputWindowOffsets),
                         ::dml::Span<const uint32_t>(inputWindowSizes),
                         ::dml::Span<const int32_t>(inputWindwStrides));
        mExpression.insert(std::make_pair(slice->PrimaryOutput(), output));
        DAWN_ASSERT(CheckShape(output, slice));
        return {};
    }

    MaybeError Graph::AddSplit(const op::Split* split) {
        DAWN_ASSERT(split->Inputs().size() == 1);
        const OperandBase* inputOperand = split->Inputs()[0].Get();
        DAWN_ASSERT(mExpression.find(inputOperand) != mExpression.end());
        ::dml::Expression input = mExpression.at(inputOperand);
        ::dml::TensorDimensions inputDims = input.GetOutputDesc().sizes;

        // dml::Span just holds the refernces, need a variable to hold the memory.
        std::vector<uint32_t> splits = split->GetSplits();
        int32_t axis = split->GetAxis();
        // This value must be in the range [0, InputTensor.DimensionCount - 1]. Negative values
        // address dimensions from the end.
        if (axis < 0) {
            axis = axis + inputDims.size();
        }
        if (splits.size() == 1) {
            if (inputDims[axis] % splits[0] != 0) {
                return DAWN_INTERNAL_ERROR("the axis " + std::to_string(axis) + " with size " +
                                           std::to_string(splits.size()) +
                                           " can't be divisible by splits[0] " +
                                           std::to_string(splits[0]) + ".");
            }
            splits = std::vector(splits[0], inputDims[axis] / splits[0]);
        }
        ::dml::Span<const uint32_t> splitsSpan(splits);
        std::vector<::dml::Expression> output = ::dml::Split(input, axis, splitsSpan);
        size_t outputSize = split->Outputs().size();
        DAWN_ASSERT(outputSize == output.size());
        for (size_t i = 0; i < outputSize; ++i) {
            mExpression.insert(std::make_pair(split->Outputs()[i].Get(), output[i]));
        }
        DAWN_ASSERT(CheckShape(output, split));
        return {};
    }

    MaybeError Graph::AddSqueeze(const op::Squeeze* squeeze) {
        DAWN_ASSERT(squeeze->Inputs().size() == 1);
        const OperandBase* inputOperand = squeeze->Inputs()[0].Get();
        DAWN_ASSERT(mExpression.find(inputOperand) != mExpression.end());
        ::dml::Expression input = mExpression.at(inputOperand);
        ::dml::TensorDimensions inputDims = input.GetOutputDesc().sizes;
        ::dml::TensorDimensions squeezeDims;
        std::vector<int32_t> axes = squeeze->GetAxes();
        if (axes.empty()) {
            for (auto& dim : inputDims) {
                if (dim != 1) {
                    squeezeDims.push_back(dim);
                }
            }
        } else {
            squeezeDims = inputDims;
            // Descending sort the axes so that they can be erased orderly.
            std::sort(axes.begin(), axes.end(), std::greater<int>());
            for (auto& axis : axes) {
                if (inputDims[axis] != 1) {
                    return DAWN_INTERNAL_ERROR(
                        "The size of the axis is not 1 that can't be squeezed.");
                } else {
                    squeezeDims.erase(squeezeDims.begin() + axis);
                }
            }
        }
        ::dml::Expression output =
            ::dml::Identity(::dml::Reinterpret(input, squeezeDims, ::dml::NullOpt));
        mExpression.insert(std::make_pair(squeeze->PrimaryOutput(), output));
        DAWN_ASSERT(CheckShape(output, squeeze));
        return {};
    }

    MaybeError Graph::AddTranspose(const op::Transpose* transpose) {
        DAWN_ASSERT(transpose->Inputs().size() == 1);
        const OperandBase* inputOperand = transpose->Inputs()[0].Get();
        DAWN_ASSERT(mExpression.find(inputOperand) != mExpression.end());
        ::dml::Expression input = mExpression.at(inputOperand);
        std::vector<int32_t> permutation = transpose->GetPermutation();
        if (permutation.size() > DML_TENSOR_DIMENSION_COUNT_MAX) {
            return DAWN_INTERNAL_ERROR("The size of permutation is not supported by DML.");
        }

        // Transpose is implemented by dml::Reinterpret and dml::Identity
        // See details at: https://github.com/microsoft/DirectML/issues/75
        const size_t inputRank = input.GetOutputDesc().sizes.size();
        ::dml::TensorDimensions inputStrides;
        if (!input.GetOutputDesc().strides) {
            inputStrides.resize(inputRank);
            uint32_t stride = 1;
            for (size_t i = inputStrides.size(); i-- > 0;) {
                inputStrides[i] = stride;
                stride *= input.GetOutputDesc().sizes[i];
            }
        } else {
            inputStrides = input.GetOutputDesc().strides.value();
        }

        ::dml::TensorDimensions transposedSizes;
        ::dml::TensorDimensions transposedStrides;
        // Permute the shape and strides.
        for (auto dimPermuted : permutation) {
            transposedSizes.push_back(input.GetOutputDesc().sizes[dimPermuted]);
            transposedStrides.push_back(inputStrides[dimPermuted]);
        }

        ::dml::Expression output =
            ::dml::Identity(::dml::Reinterpret(input, transposedSizes, transposedStrides));
        mExpression.insert(std::make_pair(transpose->PrimaryOutput(), output));
        DAWN_ASSERT(CheckShape(output, transpose));
        return {};
    }

    MaybeError Graph::AddInstanceNorm(const op::InstanceNorm* instanceNorm) {
        auto inputs = instanceNorm->Inputs();
        DAWN_ASSERT(inputs.size() == 1 || inputs.size() == 2 || inputs.size() == 3);
        DAWN_ASSERT(mExpression.find(instanceNorm->Inputs()[0].Get()) != mExpression.end());
        ::dml::Expression input = mExpression.at(instanceNorm->Inputs()[0].Get());
        const InstanceNormOptions* options = instanceNorm->GetOptions();
        if (options->layout == wnn::InputOperandLayout::Nhwc) {
            input = ReinterpretInputLayout(NhwcToNchw, input);
        }

        // The mean reductions happen over the spatial dimensions of the input
        ::dml::Span<const uint32_t> axes({2, 3});
        ::dml::TensorDimensions inputDims = input.GetOutputDesc().sizes;
        // Reshape 1D scale, bias to 4D with setting 1 to automatically broadcast.
        std::vector<::dml::Expression> expressions;
        expressions.reserve(inputs.size());
        for (size_t i = 1; i < inputs.size(); ++i) {
            DAWN_ASSERT(mExpression.find(instanceNorm->Inputs()[i].Get()) != mExpression.end());
            ::dml::Expression expression = mExpression.at(instanceNorm->Inputs()[i].Get());
            ::dml::TensorDimensions dimensions = expression.GetOutputDesc().sizes;
            DAWN_ASSERT(dimensions.size() == 1);
            if (dimensions[0] != inputDims[1]) {
                return DAWN_INTERNAL_ERROR(
                    "The 1-D tensor of the values whose length size is not equal to the size of "
                    "feature dimension of the input ");
            }
            // This tensor's dimensions should be {BatchCount, ChannelCount, Height,Width}.
            // Set 1 to automatically broadcast those dimensions across the input.
            ::dml::TensorDimensions expandDimens(4, 1);
            expandDimens[1] = dimensions[0];
            expressions.push_back(::dml::Reinterpret(expression, expandDimens, ::dml::NullOpt));
        }
        // Set tensor's dimensions to {1, channel, 1, 1} if scale or bias is null.
        const DML_TENSOR_DATA_TYPE type = DML_TENSOR_DATA_TYPE_FLOAT32;
        if (options->scale == nullptr) {
            std::vector<float> scale(inputDims[1], 1.0);
            expressions.insert(expressions.begin(),
                               BindingConstant(type, {1, inputDims[1], 1, 1}, scale.data(),
                                               sizeof(float) * inputDims[1]));
        }
        if (options->bias == nullptr) {
            std::vector<float> bias(inputDims[1], 0.0);
            expressions.push_back(BindingConstant(type, {1, inputDims[1], 1, 1}, bias.data(),
                                                  sizeof(float) * inputDims[1]));
        }

        ::dml::Expression output = ::dml::MeanVarianceNormalization(
            input, expressions[0], expressions[1], axes, true, options->epsilon);

        if (options->layout == wnn::InputOperandLayout::Nhwc) {
            output = ReinterpretInputLayout(NchwToNhwc, output);
        }
        mExpression.insert(std::make_pair(instanceNorm->PrimaryOutput(), output));
        DAWN_ASSERT(CheckShape(output, instanceNorm));
        return {};
    }

    ::dml::Expression Graph::HardSwish(::dml::Expression& input) {
        ::dml::TensorDimensions inputDims = input.GetOutputDesc().sizes;
        // x + 3
        uint32_t length = SizeOfShape(inputDims);
        std::vector<float> constant(length, 3);
        ::dml::Expression output =
            ::dml::Add(input, BindingConstant(DML_TENSOR_DATA_TYPE_FLOAT32, inputDims,
                                              constant.data(), sizeof(float) * length));
        // min(6, (x + 3))
        constant = std::vector<float>(length, 6);
        auto expressionSix = BindingConstant(DML_TENSOR_DATA_TYPE_FLOAT32, inputDims,
                                             constant.data(), sizeof(float) * length);
        output = ::dml::Min(output, expressionSix);
        constant = std::vector<float>(length, 0);
        // max(0, min(6, (x + 3)))
        output = ::dml::Max(output, BindingConstant(DML_TENSOR_DATA_TYPE_FLOAT32, inputDims,
                                                    constant.data(), sizeof(float) * length));
        // x * max(0, min(6, (x + 3)))
        output = ::dml::Multiply(input, output);
        // x * max(0, min(6, (x + 3))) / 6
        output = ::dml::Divide(output, expressionSix);
        return output;
    }

    MaybeError Graph::AddUnary(const op::Unary* unary) {
        DAWN_ASSERT(unary->Inputs().size() == 1);
        const OperandBase* inputOperand = unary->Inputs()[0].Get();
        DAWN_ASSERT(mExpression.find(inputOperand) != mExpression.end());
        ::dml::Expression input = mExpression.at(inputOperand);
        ::dml::TensorDimensions inputDims = input.GetOutputDesc().sizes;
        if (inputDims.size() > DML_TENSOR_DIMENSION_COUNT_MAX1) {
            return DAWN_INTERNAL_ERROR("The size of input dimensions isn't supported.");
        }

        ::dml::Expression output;
        switch (unary->GetType()) {
            case op::UnaryOpType::kAbs:
                output = ::dml::Abs(input);
                break;
            case op::UnaryOpType::kCeil:
                output = ::dml::Ceil(input);
                break;
            case op::UnaryOpType::kCos:
                output = ::dml::Cos(input);
                break;
            case op::UnaryOpType::kExp:
                output = ::dml::Exp(input);
                break;
            case op::UnaryOpType::kFloor:
                output = ::dml::Floor(input);
                break;
            case op::UnaryOpType::kHardSwish:
                dawn::WarningLog() << "The hardSwish is emulated from other operations, maybe the "
                                      "performance isn't best";
                output = HardSwish(input);
                break;
            case op::UnaryOpType::kLog:
                output = ::dml::Log(input);
                break;
            case op::UnaryOpType::kLeakyRelu:
                output = ::dml::ActivationLeakyRelu(
                    input, reinterpret_cast<const op::LeakyRelu*>(unary)->GetAlpha());
                break;
            // DML doesn't support element-wise negative, emulated it from multiplying input by -1.
            case op::UnaryOpType::kNeg: {
                uint32_t length = SizeOfShape(inputDims);
                auto dataType = input.GetOutputDesc().dataType;
                if (dataType == DML_TENSOR_DATA_TYPE_FLOAT32) {
                    std::vector<float> constant(length, -1);
                    output =
                        ::dml::Multiply(input, BindingConstant(dataType, inputDims, constant.data(),
                                                               sizeof(float) * length));
                } else if (dataType == DML_TENSOR_DATA_TYPE_INT32) {
                    std::vector<int32_t> constant(length, -1);
                    output =
                        ::dml::Multiply(input, BindingConstant(dataType, inputDims, constant.data(),
                                                               sizeof(int32_t) * length));
                } else {
                    return DAWN_UNIMPLEMENTED_ERROR(" Unary op " +
                                                    OpTypeToString(unary->GetType()) +
                                                    " with this data type is not implemented.");
                }
                break;
            }
            case op::UnaryOpType::kRelu:
                output = ::dml::ActivationRelu(input);
                break;
            case op::UnaryOpType::kSigmoid:
                output = ::dml::ActivationSigmoid(input);
                break;
            case op::UnaryOpType::kSin:
                output = ::dml::Sin(input);
                break;
            case op::UnaryOpType::kSoftmax:
                output = ::dml::ActivationSoftmax(input);
                break;
            case op::UnaryOpType::kTan:
                output = ::dml::Tan(input);
                break;
            case op::UnaryOpType::kTanh:
                output = ::dml::ActivationTanh(input);
                break;
            default:
                return DAWN_UNIMPLEMENTED_ERROR(" Unary op " + OpTypeToString(unary->GetType()) +
                                                " is not implemented.");
        }
        mExpression.insert(std::make_pair(unary->PrimaryOutput(), output));
        DAWN_ASSERT(CheckShape(output, unary));
        return {};
    }

    MaybeError Graph::AddConcat(const op::Concat* concat) {
        DAWN_ASSERT(concat->Inputs().size() >= 1);
        auto inputsOperand = concat->Inputs();
        std::vector<::dml::Expression> inputs;
        inputs.reserve(inputsOperand.size());
        const ::dml::Expression primary = mExpression.at(inputsOperand[0].Get());
        const ::dml::TensorDimensions primaryDims = primary.GetOutputDesc().sizes;
        if (primaryDims.size() > DML_TENSOR_DIMENSION_COUNT_MAX) {
            return DAWN_INTERNAL_ERROR("The size of input dimensions is greater than max");
        }

        uint32_t axis = concat->GetAxis();
        for (auto& inputOperand : inputsOperand) {
            DAWN_ASSERT(mExpression.find(inputOperand.Get()) != mExpression.end());
            ::dml::Expression input = mExpression.at(inputOperand.Get());
            ::dml::TensorDimensions inputDims = input.GetOutputDesc().sizes;
            DAWN_ASSERT(inputDims.size() == primaryDims.size());
            // All input tensors must have the same shape, except for the size of the dimension to
            // concatenate on.
            for (uint32_t i = 0; i < inputDims.size(); ++i) {
                if (i == axis)
                    continue;
                if (inputDims[i] != primaryDims[i]) {
                    return DAWN_VALIDATION_ERROR(
                        "All input tensors must have the same shape except the axis.");
                }
            }
            // Expand dimensions to DML_TENSOR_DIMENSION_COUNT_MAX if needed.
            if (inputDims.size() < DML_TENSOR_DIMENSION_COUNT_MAX) {
                auto newDims = ExpandDimensions(inputDims, DML_TENSOR_DIMENSION_COUNT_MAX);
                input = ::dml::Reinterpret(input, newDims, ::dml::NullOpt);
            }
            inputs.push_back(input);
        }

        // Update the axis to align with the DML_TENSOR_DIMENSION_COUNT_MAX.
        axis += DML_TENSOR_DIMENSION_COUNT_MAX - primaryDims.size();
        ::dml::Expression output = ::dml::Join(inputs, axis);
        ::dml::TensorDimensions outputDims = output.GetOutputDesc().sizes;
        // Reshape back according to output rank if needed.
        if (primaryDims.size() < outputDims.size()) {
            auto dims = ShrinkDimensions(outputDims, primaryDims.size());
            output = ::dml::Reinterpret(output, dims, ::dml::NullOpt);
        }
        mExpression.insert(std::make_pair(concat->PrimaryOutput(), output));
        DAWN_ASSERT(CheckShape(output, concat));
        return {};
    }

    MaybeError Graph::AddGemm(const op::Gemm* gemm) {
        std::vector<uint32_t> outputDims;
        outputDims.reserve(2);
        auto inputs = gemm->Inputs();
        DAWN_ASSERT(inputs.size() == 2 || inputs.size() == 3);
        DAWN_ASSERT(mExpression.find(inputs[0].Get()) != mExpression.end());
        ::dml::Expression a = mExpression.at(inputs[0].Get());
        ::dml::TensorDimensions aDims = a.GetOutputDesc().sizes;
        const GemmOptions* options = gemm->GetOptions();
        outputDims.push_back(options->aTranspose ? aDims[1] : aDims[0]);
        // The shape of a tensor is 2D definited in WebNN Spec, but DML only support 4D,
        // so expand dimensions to 4D.
        DAWN_ASSERT(aDims.size() == 2);
        auto expandDims = ExpandDimensions(aDims, 4);
        a = ::dml::Reinterpret(a, expandDims, ::dml::NullOpt);

        DAWN_ASSERT(mExpression.find(inputs[1].Get()) != mExpression.end());
        ::dml::Expression b = mExpression.at(inputs[1].Get());
        ::dml::TensorDimensions bDims = b.GetOutputDesc().sizes;
        outputDims.push_back(options->bTranspose ? bDims[0] : bDims[1]);
        // The shape of b tensor is 2D definited in WebNN Spec, but DML only support 4D,
        // so expand dimensions to 4D.
        DAWN_ASSERT(bDims.size() == 2);
        expandDims = ExpandDimensions(bDims, 4);
        b = ::dml::Reinterpret(b, expandDims, ::dml::NullOpt);

        // The operand c is optional.
        ::dml::Optional<::dml::Expression> c = ::dml::NullOpt;
        if (inputs.size() == 3) {
            DAWN_ASSERT(mExpression.find(inputs[2].Get()) != mExpression.end());
            c = mExpression.at(inputs[2].Get());
            ::dml::TensorDimensions cDims = c->GetOutputDesc().sizes;
            if (cDims.size() != 2) {
                cDims = ExpandDimensions(cDims, 2);
            }
            // BroadCast the Shape of optional C to {1, 1, M, N } supported in DML.
            std::vector<bool> broadcast(4, false);
            for (size_t i = 0; i < 2; ++i) {
                if (outputDims[i] != cDims[i]) {
                    if (cDims[i] == 1) {
                        broadcast[i + 2] = true;
                        cDims[i] = outputDims[i];
                    } else {
                        return DAWN_INTERNAL_ERROR("The optional c can't be broadcast.");
                    }
                }
            }
            // The shape of c tensor is 2D definited in WebNN Spec, but DML only support 4D,
            // so expand dimensions to 4D.
            DAWN_ASSERT(cDims.size() == 2);
            expandDims = ExpandDimensions(cDims, 4);
            auto expandStrides = CalculateBroadcastStrides(expandDims, broadcast);
            c = ::dml::Reinterpret(c->Impl(), expandDims, expandStrides);
        }

        DML_MATRIX_TRANSFORM aTranspose = gemm->GetOptions()->aTranspose
                                              ? DML_MATRIX_TRANSFORM_TRANSPOSE
                                              : DML_MATRIX_TRANSFORM_NONE;
        DML_MATRIX_TRANSFORM bTranspose = gemm->GetOptions()->bTranspose
                                              ? DML_MATRIX_TRANSFORM_TRANSPOSE
                                              : DML_MATRIX_TRANSFORM_NONE;
        ::dml::Expression output =
            ::dml::Gemm(a, b, c, aTranspose, bTranspose, options->alpha, options->beta);
        // Reshape back according to output rank.
        auto shrinkDims = ShrinkDimensions(output.GetOutputDesc().sizes, 2);
        output = ::dml::Reinterpret(output, shrinkDims, ::dml::NullOpt);
        mExpression.insert(std::make_pair(gemm->PrimaryOutput(), output));
        DAWN_ASSERT(CheckShape(output, gemm));
        return {};
    }

    MaybeError Graph::AddGru(const op::Gru* gru) {
        auto inputs = gru->Inputs();
        auto options = gru->GetOptions();
        DAWN_ASSERT(inputs.size() >= 3 && inputs.size() <= 6);

        DAWN_ASSERT(mExpression.find(inputs[0].Get()) != mExpression.end());
        ::dml::Expression input = mExpression.at(inputs[0].Get());
        ::dml::TensorDimensions inputDims = input.GetOutputDesc().sizes;
        // Reshape input from 3-D to 4-D layout.
        ::dml::TensorDimensions expandInputDimens = ExpandDimensions(inputDims, 4);
        input = ::dml::Reinterpret(input, expandInputDimens, ::dml::NullOpt);

        DAWN_ASSERT(mExpression.find(inputs[1].Get()) != mExpression.end());
        ::dml::Expression weight = mExpression.at(inputs[1].Get());
        ::dml::TensorDimensions weightDims = weight.GetOutputDesc().sizes;
        // Reshape weight from 3-D to 4-D layout.
        ::dml::TensorDimensions expandWeightDimens = ExpandDimensions(weightDims, 4);
        weight = ::dml::ActivationIdentity(
            ::dml::Reinterpret(weight, expandWeightDimens, ::dml::NullOpt));

        DAWN_ASSERT(mExpression.find(inputs[2].Get()) != mExpression.end());
        ::dml::Expression recurrentWeight = mExpression.at(inputs[2].Get());
        ::dml::TensorDimensions recurrentWeightDims = recurrentWeight.GetOutputDesc().sizes;
        // Reshape recurrentWeight from 3-D to 4-D layout.
        ::dml::TensorDimensions expandRecurrentWeightDimens =
            ExpandDimensions(recurrentWeightDims, 4);
        recurrentWeight = ::dml::ActivationIdentity(
            ::dml::Reinterpret(recurrentWeight, expandRecurrentWeightDimens, ::dml::NullOpt));

        int n = 3;
        ::dml::Optional<::dml::Expression> bias = ::dml::NullOpt;
        ::dml::Expression normalBias;
        ::dml::Expression recurrentBias;
        ::dml::TensorDimensions halfBiasDimens = {1, 1, weightDims[0], weightDims[1]};
        if (options->bias != nullptr) {
            DAWN_ASSERT(mExpression.find(inputs[n].Get()) != mExpression.end());
            normalBias = mExpression.at(inputs[n++].Get());
            ::dml::TensorDimensions normalBiasDims = normalBias.GetOutputDesc().sizes;
            // Reshape normal bias from 2-D to 4-D layout.
            normalBias = ::dml::Reinterpret(normalBias, halfBiasDimens, ::dml::NullOpt);
        } else {
            uint32_t length = SizeOfShape(halfBiasDimens);
            std::vector<float> constant(length, 0);
            normalBias = BindingConstant(DML_TENSOR_DATA_TYPE_FLOAT32, halfBiasDimens,
                                         constant.data(), sizeof(float) * length);
        }
        if (options->recurrentBias != nullptr) {
            DAWN_ASSERT(mExpression.find(inputs[n].Get()) != mExpression.end());
            recurrentBias = mExpression.at(inputs[n++].Get());
            ::dml::TensorDimensions recurrentBiasDims = recurrentBias.GetOutputDesc().sizes;
            // Reshape recurrent bias from 2-D to 4-D layout.
            recurrentBias = ::dml::Reinterpret(recurrentBias, halfBiasDimens, ::dml::NullOpt);
        } else {
            uint32_t length = SizeOfShape(halfBiasDimens);
            std::vector<float> constant(length, 0);
            recurrentBias = BindingConstant(DML_TENSOR_DATA_TYPE_FLOAT32, halfBiasDimens,
                                            constant.data(), sizeof(float) * length);
        }
        std::vector<::dml::Expression> biasExpressions = {normalBias, recurrentBias};
        bias = ::dml::Join(biasExpressions, 3);

        ::dml::Optional<::dml::Expression> initialHiddenState = ::dml::NullOpt;
        if (options->initialHiddenState != nullptr) {
            DAWN_ASSERT(mExpression.find(inputs[n].Get()) != mExpression.end());
            initialHiddenState = mExpression.at(inputs[n++].Get());
            ::dml::TensorDimensions initialHiddenStateDims =
                initialHiddenState->GetOutputDesc().sizes;
            // Reshape initialHiddenState from 3-D to 4-D layout.
            ::dml::TensorDimensions expandInitialHiddenStateDimens =
                ExpandDimensions(initialHiddenStateDims, 4);
            initialHiddenState = ::dml::Reinterpret(*initialHiddenState,
                                                    expandInitialHiddenStateDimens, ::dml::NullOpt);
            initialHiddenState = ::dml::ActivationIdentity(initialHiddenState->Impl());
        }

        ::dml::Optional<::dml::Expression> SequenceLength = ::dml::NullOpt;
        DML_RECURRENT_NETWORK_DIRECTION direction =
            getRecurrentSequenceDirection(options->direction);

        // TODO: layout
        if (options->layout == wnn::RecurrentNetworkWeightLayout::Rzn) {
            return DAWN_INTERNAL_ERROR(
                "layout defaults to 'zrn'. Only 'zrn' is currently supported.");
        }

        ::dml::FusedActivation fActivation, gActivation;
        if (options->activations == nullptr) {
            fActivation = ::dml::FusedActivation::Sigmoid();
            gActivation = ::dml::FusedActivation::Tanh();
        } else {
            fActivation = CreateFusedActivation(options->activations->Get(0));
            gActivation = CreateFusedActivation(options->activations->Get(1));
        }
        std::vector<::dml::FusedActivation> activations;
        if (direction ==
            DML_RECURRENT_NETWORK_DIRECTION::DML_RECURRENT_NETWORK_DIRECTION_BIDIRECTIONAL) {
            activations = {fActivation, gActivation, fActivation, gActivation};
        } else {
            activations = {fActivation, gActivation};
        }
        ::dml::Span<const ::dml::FusedActivation> activationDescs(activations);
        bool linearBeforeReset = options->resetAfter;
        ::dml::GRUOutputOptions outputOption = ::dml::GRUOutputOptions::Both;

        ::dml::GRUOutputs outputs =
            ::dml::GRU(input, weight, recurrentWeight, bias, initialHiddenState, SequenceLength,
                       activationDescs, direction, linearBeforeReset, outputOption);
        ::dml::Expression singleOutput = outputs.single;
        ::dml::TensorDimensions singleOutputDims = singleOutput.GetOutputDesc().sizes;
        // Reshape initialHiddenState from 4-D to 3-D layout.
        ::dml::TensorDimensions shrinkDimens = ShrinkDimensions(singleOutputDims, 3);
        singleOutput = ::dml::Reinterpret(singleOutput, shrinkDimens, ::dml::NullOpt);
        mExpression.insert(std::make_pair(gru->Outputs()[0].Get(), singleOutput));
        if (options->returnSequence) {
            ::dml::Expression sequenceOutput = outputs.sequence;
            mExpression.insert(std::make_pair(gru->Outputs()[1].Get(), sequenceOutput));
        }
        return {};
    }

    MaybeError Graph::Finish() {
        if (mInputBindingMap.empty()) {
            return DAWN_VALIDATION_ERROR("Model inputs must be set.");
        }
        if (mOutputExpressionMap.size() == 1) {
            std::string name = mOutputExpressionMap.begin()->first;
            auto outputExp = mOutputExpressionMap.begin()->second;
#if defined(WEBNN_ENABLE_GPU_BUFFER)
            auto builder = outputExp.Impl()->GetGraphBuilder();
#endif
            auto node = outputExp.Impl()->GetNode();
            if (node.type == ::dml::detail::NodeType::Reinterpret
#if defined(WEBNN_ENABLE_GPU_BUFFER)
                && builder->m_reinterpretNodes[node.index].input->GetNode().type ==
                       ::dml::detail::NodeType::Input
#endif
            ) {
                // Deal with a graph with single reshape node.
                // https://github.com/microsoft/DirectML/issues/71
                mOutputExpressionMap[name] = ::dml::ActivationIdentity(outputExp);
            }
        }

        return {};
    }

    MaybeError Graph::CompileImpl() {
        // FIXME(nhu): implement async
        std::vector<::dml::Expression> outputs;
        for (auto& output : mOutputExpressionMap) {
            outputs.push_back(output.second);
        }
        // TODO(nhu): investigate other execution flag,
        // e.g. DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION
        mCompiledModel.reset(new pydml::CompiledModel(*(mGraph), DML_EXECUTION_FLAG_NONE, outputs));

        std::vector<pydml::Binding*> inputBindings;
        for (auto& binding : mInputBindings) {
            inputBindings.push_back(binding.get());
        }
        std::lock_guard<std::mutex> lock(mMutex);
        if (FAILED(mDevice->InitializeOperator(mCompiledModel->op.Get(), inputBindings))) {
            return DAWN_INTERNAL_ERROR("Failed to compile graph.");
        }
        return {};
    }

    WNNComputeGraphStatus Graph::ComputeImpl(NamedInputsBase* inputs, NamedOutputsBase* outputs) {
        auto namedInputs = inputs->GetRecords();
        for (auto& [name, inputBinding] : mInputBindingMap) {
            // All the inputs must be set.
            if (namedInputs.find(name) == namedInputs.end()) {
                dawn::ErrorLog() << "The input must be set.";
                return WNNComputeGraphStatus_Error;
            }

            auto& resource = namedInputs[name].resource;
            if (resource.arrayBufferView.buffer != nullptr) {
#ifndef WEBNN_ENABLE_GPU_BUFFER
                auto& arrayBuffer = resource.arrayBufferView;
                inputBinding->data.buffer =
                    static_cast<int8_t*>(arrayBuffer.buffer) + arrayBuffer.byteOffset;
                inputBinding->data.size = arrayBuffer.byteLength;
#endif
            } else {
#ifdef WEBNN_ENABLE_GPU_BUFFER
                auto& gpuBuffer = resource.gpuBufferView;
                DAWN_ASSERT(gpuBuffer.id != 0);
                inputBinding->data.buffer = reinterpret_cast<WGPUBuffer>(gpuBuffer.buffer);
                inputBinding->data.offset = gpuBuffer.offset;
                inputBinding->data.size = gpuBuffer.size;
#endif
            }
        }
        std::vector<pydml::Binding*> inputBindings;
        for (auto& binding : mInputBindings) {
            inputBindings.push_back(binding.get());
        }

        std::vector<::dml::Expression*> outputExpressions;
        std::vector<std::string> outputNames;
        for (auto& [name, output] : mOutputExpressionMap) {
            outputNames.push_back(name);
            outputExpressions.push_back(&output);
        }
#ifdef WEBNN_ENABLE_GPU_BUFFER
        std::vector<pydml::Binding*> outputBindings;
        auto namedOutputs = outputs->GetRecords();
        for (auto& output : mOutputBindingMap) {
            ::pydml::Binding* binding = output.second;
            auto& bufferView = namedOutputs[output.first];
            if (bufferView.arrayBufferView.) {
                dawn::InfoLog()
                    << "Array Buffer input use expression parameters in DispatchOperator.";
            } else {
                WGPUBuffer gpuBuffer =
                    reinterpret_cast<WGPUBuffer>(bufferView.gpuBufferView.buffer);
                binding->data.buffer = gpuBuffer;
                binding->data.offset = bufferView.gpuBufferView.offset;
                binding->data.size = bufferView.gpuBufferView.size;
            }
            outputBindings.push_back(binding);
        }
#endif

        std::lock_guard<std::mutex> lock(mMutex);
#ifdef WEBNN_ENABLE_GPU_BUFFER
        if (FAILED(mDevice->DispatchOperator(mCompiledModel->op.Get(), inputBindings,
                                             outputBindings))) {
            dawn::ErrorLog() << "Failed to dispatch operator.";
        }
#else
        std::vector<pydml::TensorData*> outputTensors;
        if (FAILED(mDevice->DispatchOperator(mCompiledModel->op.Get(), inputBindings,
                                             outputExpressions, outputTensors))) {
            dawn::ErrorLog() << "Failed to dispatch operator.";
            return WNNComputeGraphStatus_Error;
        }

        for (size_t i = 0; i < outputNames.size(); ++i) {
            std::string outputName = outputNames[i];
            pydml::TensorData* tensor = outputTensors[i];
            void* outputBuffer = tensor->Get();
            size_t bufferLength = tensor->Size();
            auto namedOutputs = outputs->GetRecords();
            if (namedOutputs.find(outputName) != namedOutputs.end()) {
                ArrayBufferView output = namedOutputs[outputName].arrayBufferView;
                if (output.byteLength >= bufferLength) {
                    memcpy(static_cast<int8_t*>(output.buffer) + output.byteOffset, outputBuffer,
                           bufferLength);
                }
            }
            free(outputBuffer);
            delete tensor;
        }
#endif
        return WNNComputeGraphStatus_Success;
    }

}  // namespace webnn_native::dml
