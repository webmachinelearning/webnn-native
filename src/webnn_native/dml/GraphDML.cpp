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

#include "common/Assert.h"
#include "common/Log.h"
#include "webnn_native/ErrorData.h"
#include "webnn_native/NamedInputs.h"
#include "webnn_native/NamedOutputs.h"
#include "webnn_native/NamedResults.h"
#include "webnn_native/Operand.h"
#include "webnn_native/Result.h"
#include "webnn_native/dml/ContextDML.h"
#include "webnn_native/dml/deps/src/precomp.h"

namespace webnn_native { namespace dml {
    class Result : public ResultBase {
      public:
        explicit Result(void* buffer, uint32_t buffer_size, std::vector<int32_t>& dimensions)
            : ResultBase(buffer, buffer_size, dimensions) {
        }
        ~Result() {
            free(mBuffer);
        }
    };

    namespace {
        bool GetDmlTensorDataType(ml::OperandType operandType,
                                  DML_TENSOR_DATA_TYPE& dmlTensorDataType) {
            if (operandType == ml::OperandType::Float32) {
                dmlTensorDataType = DML_TENSOR_DATA_TYPE_FLOAT32;
            } else if (operandType == ml::OperandType::Float16) {
                dmlTensorDataType = DML_TENSOR_DATA_TYPE_FLOAT16;
            } else if (operandType == ml::OperandType::Int32) {
                dmlTensorDataType = DML_TENSOR_DATA_TYPE_INT32;
            } else if (operandType == ml::OperandType::Uint32) {
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
            dmlTensorDimensions.resize(dimensionsCount);
            for (uint32_t i = 0; i < dimensionsCount; ++i) {
                int32_t d = dimensions[i];
                if (d < 0) {
                    dawn::ErrorLog() << "DML doesn't support the negative dimension value";
                    return false;
                }
                dmlTensorDimensions[i] = d;
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

        // Refer to
        // https://docs.microsoft.com/en-us/windows/win32/direct3d12/dml-helper-functions#calculatestrides
        ::dml::TensorDimensions CalculateStrides(::dml::TensorDimensions dims,
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
            aNewStrides = CalculateStrides(aNewDims, aBroadcast);
            bNewStrides = CalculateStrides(bNewDims, bBroadcast);
            return true;
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
            }
            return std::to_string(type);
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
        mDevice = context->GetDevice();
        mGraph.reset(new ::dml::Graph(mDevice->GetDevice()));
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
        ::dml::TensorDesc dmlTensorDesc(dmlTensorType,
                                        ::DML_TENSOR_FLAGS::DML_TENSOR_FLAG_OWNED_BY_DML,
                                        dmlTensorDims, ::dml::TensorPolicy::Default());
        ::dml::Expression dmlConstant =
            ::dml::InputTensor(*mGraph, mBindings.size(), dmlTensorDesc);
        mExpression.insert(std::make_pair(constant, dmlConstant));
        std::unique_ptr<char> buffer(new char[constant->GetSize()]);
        memcpy(buffer.get(), constant->GetValue(), constant->GetSize());
        std::unique_ptr<::pydml::Binding> binding(new ::pydml::Binding(
            dmlConstant, static_cast<void*>(buffer.get()), constant->GetSize()));
        mConstantBuffers.push_back(std::move(buffer));
        mBindings.push_back(std::move(binding));
        return {};
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
        ::dml::Expression dmlInput = ::dml::InputTensor(*mGraph, mBindings.size(), dmlTensorDesc);
        mExpression.insert(std::make_pair(input, dmlInput));
        std::unique_ptr<::pydml::Binding> binding(new ::pydml::Binding(dmlInput, nullptr, 0));
        mBindings.push_back(std::move(binding));
        mInputs.insert(std::make_pair(input->GetName(), mBindings.back().get()));
        return {};
    }

    MaybeError Graph::AddOutput(const std::string& name, const OperandBase* output) {
        DAWN_ASSERT(mExpression.find(output) != mExpression.end());
        ::dml::Expression dmlOutput = mExpression.at(output);
        mOutputs.insert(std::make_pair(name, dmlOutput));
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
                aNewStrides = CalculateStrides(aNewDims);
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
                bNewStrides = CalculateStrides(bNewDims);
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
        } else if (binary->GetType() == op::BinaryOpType::kMul) {
            c = ::dml::Multiply(a, b);
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
            ::dml::TensorDimensions cNewStrides = CalculateStrides(cNewDims);
            c = ::dml::Reinterpret(c, cNewDims, cNewStrides);
        }
        mExpression.insert(std::make_pair(binary, c));
        return {};
    }

    MaybeError Graph::AddConv2d(const op::Conv2d* conv2d) {
        DAWN_ASSERT(conv2d->Inputs().size() == 2);
        const OperandBase* inputOperand = conv2d->Inputs()[0].Get();
        DAWN_ASSERT(mExpression.find(inputOperand) != mExpression.end());
        ::dml::Expression input = mExpression.at(inputOperand);
        const OperandBase* filterOperand = conv2d->Inputs()[1].Get();
        DAWN_ASSERT(mExpression.find(filterOperand) != mExpression.end());
        ::dml::Expression filter = mExpression.at(filterOperand);
        const Conv2dOptions* options = conv2d->GetOptions();
        // FIXME(nhu): strides, dilations, padding should be uint32_t
        // need to fix the spec.
        ::dml::Span<const uint32_t> strides(reinterpret_cast<const uint32_t*>(options->strides),
                                            options->stridesCount);
        ::dml::Span<const uint32_t> dilations(reinterpret_cast<const uint32_t*>(options->dilations),
                                              options->dilationsCount);
        // dml::Span just holds the refernces, need a variable to hold the memory.
        std::vector<const uint32_t> startPaddingVector(
            {static_cast<const uint32_t>(options->padding[0]),
             static_cast<const uint32_t>(options->padding[2])});
        std::vector<const uint32_t> endPaddingVector(
            {static_cast<const uint32_t>(options->padding[1]),
             static_cast<const uint32_t>(options->padding[3])});
        ::dml::Span<const uint32_t> startPadding(startPaddingVector);
        ::dml::Span<const uint32_t> endPadding(endPaddingVector);
        ::dml::Expression output = ::dml::Convolution(
            input, filter, ::dml::NullOpt, DML_CONVOLUTION_MODE_CROSS_CORRELATION,
            DML_CONVOLUTION_DIRECTION_FORWARD, strides, dilations, startPadding, endPadding,
            // outPadding
            {},
            // groupCount
            options->groups);
        mExpression.insert(std::make_pair(conv2d, output));
        return {};
    }

    MaybeError Graph::AddPool2d(const op::Pool2d* pool2d) {
        DAWN_ASSERT(pool2d->Inputs().size() == 1);
        const OperandBase* inputOperand = pool2d->Inputs()[0].Get();
        DAWN_ASSERT(mExpression.find(inputOperand) != mExpression.end());
        ::dml::Expression input = mExpression.at(inputOperand);
        const Pool2dOptions* options = pool2d->GetOptions();
        ::dml::Span<const uint32_t> strides(reinterpret_cast<const uint32_t*>(options->strides),
                                            options->stridesCount);
        std::vector<std::uint32_t> windowSizesVector;
        if (options->windowDimensions != nullptr) {
            const uint32_t* windowDimensions =
                reinterpret_cast<const uint32_t*>(options->windowDimensions);
            windowSizesVector.assign(windowDimensions,
                                     windowDimensions + options->windowDimensionsCount);
        } else {
            const ::dml::TensorDimensions& inputSizes = input.GetOutputDesc().sizes;
            // FIXME(nhu): deal wtih NHWC layout
            windowSizesVector = {inputSizes[2], inputSizes[3]};
        }
        ::dml::Span<const uint32_t> windowSizes(windowSizesVector);
        ::dml::Span<const uint32_t> dilations(reinterpret_cast<const uint32_t*>(options->dilations),
                                              options->dilationsCount);
        std::vector<const uint32_t> startPaddingVector(
            {static_cast<const uint32_t>(options->padding[0]),
             static_cast<const uint32_t>(options->padding[2])});
        std::vector<const uint32_t> endPaddingVector(
            {static_cast<const uint32_t>(options->padding[1]),
             static_cast<const uint32_t>(options->padding[3])});
        ::dml::Span<const uint32_t> startPadding(startPaddingVector);
        ::dml::Span<const uint32_t> endPadding(endPaddingVector);
        ::dml::Expression output;
        if (pool2d->GetType() == op::Pool2dType::kAveragePool2d) {
            if (dilations[0] != 1 || dilations[1] != 1) {
                return DAWN_INTERNAL_ERROR("The dilations of average pool2d are not supported.");
            }
            output =
                ::dml::AveragePooling(input, strides, windowSizes, startPadding, endPadding, false);
        } else if (pool2d->GetType() == op::Pool2dType::kMaxPool2d) {
            output = ::dml::MaxPooling(input, windowSizes, strides, startPadding, endPadding,
                                       dilations, false)
                         .values;
        } else {
            return DAWN_INTERNAL_ERROR("l2Pool2d is not supported.");
        }
        mExpression.insert(std::make_pair(pool2d, output));
        return {};
    }

    MaybeError Graph::AddReshape(const op::Reshape* reshape) {
        DAWN_ASSERT(reshape->Inputs().size() == 1);
        const OperandBase* inputOperand = reshape->Inputs()[0].Get();
        DAWN_ASSERT(mExpression.find(inputOperand) != mExpression.end());
        ::dml::Expression input = mExpression.at(inputOperand);
        if (reshape->GetNewShapeCount() > DML_TENSOR_DIMENSION_COUNT_MAX) {
            return DAWN_INTERNAL_ERROR("The size of new shape is not supported by DML.");
        }
        std::vector<int32_t> newShape;
        newShape.assign(reshape->GetNewShape(),
                        reshape->GetNewShape() + reshape->GetNewShapeCount());
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
        mExpression.insert(std::make_pair(reshape, output));
        return {};
    }

    MaybeError Graph::AddTranspose(const op::Transpose* transpose) {
        DAWN_ASSERT(transpose->Inputs().size() == 1);
        const OperandBase* inputOperand = transpose->Inputs()[0].Get();
        DAWN_ASSERT(mExpression.find(inputOperand) != mExpression.end());
        ::dml::Expression input = mExpression.at(inputOperand);
        const TransposeOptions* options = transpose->GetOptions();
        if (options->permutationCount > DML_TENSOR_DIMENSION_COUNT_MAX) {
            return DAWN_INTERNAL_ERROR("The size of permutation is not supported by DML.");
        }
        const size_t inputRank = input.GetOutputDesc().sizes.size();
        ::dml::TensorDimensions permutation(inputRank);
        if (options->permutationCount == 0) {
            size_t index = inputRank;
            for (auto& p : permutation) {
                p = --index;
            }
        } else if (options->permutationCount == inputRank) {
            for (size_t i = 0; i < inputRank; ++i) {
                if (options->permutation[i] < 0) {
                    return DAWN_VALIDATION_ERROR("The value of permutation is invalid.");
                } else {
                    permutation[i] = options->permutation[i];
                }
            }
        } else {
            return DAWN_VALIDATION_ERROR("The size of permutation is invalid.");
        }

        // Transpose is implemented by dml::Reinterpret and dml::Identity
        // See details at: https://github.com/microsoft/DirectML/issues/75
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

        ::dml::TensorDimensions transposedSizes(inputRank);
        ::dml::TensorDimensions transposedStrides(inputRank);

        // Permute the shape and strides.
        for (size_t i = 0; i < inputRank; ++i) {
            size_t dimPermuted = permutation[i];
            transposedSizes[i] = input.GetOutputDesc().sizes[dimPermuted];
            transposedStrides[i] = inputStrides[dimPermuted];
        }

        ::dml::Expression output =
            ::dml::Identity(::dml::Reinterpret(input, transposedSizes, transposedStrides));
        mExpression.insert(std::make_pair(transpose, output));
        return {};
    }

    MaybeError Graph::AddUnary(const op::Unary* unary) {
        DAWN_ASSERT(unary->Inputs().size() == 1);
        const OperandBase* inputOperand = unary->Inputs()[0].Get();
        DAWN_ASSERT(mExpression.find(inputOperand) != mExpression.end());
        ::dml::Expression input = mExpression.at(inputOperand);
        ::dml::Expression output;
        if (unary->GetType() == op::UnaryOpType::kRelu) {
            output = ::dml::ActivationRelu(input);
        } else if (unary->GetType() == op::UnaryOpType::kSoftmax) {
            output = ::dml::ActivationSoftmax(input);
        } else {
            std::string errorMessage = std::string(" Unary op ") +
                                       OpTypeToString(unary->GetType()) +
                                       std::string(" is not implemented.");
            return DAWN_UNIMPLEMENTED_ERROR(errorMessage);
        }
        mExpression.insert(std::make_pair(unary, output));
        return {};
    }

    MaybeError Graph::Finish() {
        if (mOutputs.size() == 1) {
            auto output = mOutputs.begin();
            if (output->second.Impl()->GetNode().type == ::dml::detail::NodeType::Reinterpret) {
                // Deal with a graph with single reshape node.
                // https://github.com/microsoft/DirectML/issues/71
                std::string name = output->first;
                ::dml::Expression reshape = output->second;
                mOutputs[name] = ::dml::ActivationIdentity(reshape);
            }
        }

        return {};
    }

    void Graph::CompileImpl(BuildGraphCallbackDelgate delgate) {
        // FIXME(nhu): implement async
        std::vector<::dml::Expression> outputs;
        for (auto& output : mOutputs) {
            outputs.push_back(output.second);
        }
        // TODO(nhu): investigate other execution flag,
        // e.g. DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION
        mCompiledModel.reset(new pydml::CompiledModel(*(mGraph), DML_EXECUTION_FLAG_NONE, outputs));

        std::vector<pydml::Binding*> inputBindings;
        for (auto& binding : mBindings) {
            inputBindings.push_back(binding.get());
        }
        MLBuildGraphStatus status =
            FAILED(mDevice->InitializeOperator(mCompiledModel->op.Get(), inputBindings))
                ? MLBuildGraphStatus_Error
                : MLBuildGraphStatus_Success;
        delgate(status, this);
    }

    void Graph::ComputeImpl(NamedInputsBase* inputs,
                            MLComputeGraphCallback callback,
                            void* userdata,
                            NamedOutputsBase* outputs) {
        for (auto& input : inputs->GetRecords()) {
            ::pydml::Binding* inputBinding = mInputs.at(input.first);
            inputBinding->data.buffer = const_cast<void*>(input.second->buffer);
            inputBinding->data.size = input.second->size;
        }
        std::vector<pydml::Binding*> inputBindings;
        for (auto& binding : mBindings) {
            inputBindings.push_back(binding.get());
        }
        std::vector<::dml::Expression*> outputExpressions;
        std::vector<std::string> outputNames;
        if (outputs != nullptr) {
            for (auto& output : outputs->GetRecords()) {
                outputNames.push_back(output.first);
                outputExpressions.push_back(&(mOutputs.at(output.first)));
            }
        } else {
            for (auto& output : mOutputs) {
                outputNames.push_back(output.first);
                outputExpressions.push_back(&(output.second));
            }
        }
        std::vector<pydml::TensorData*> outputTensors;
        if (FAILED(mDevice->DispatchOperator(mCompiledModel->op.Get(), inputBindings,
                                             outputExpressions, outputTensors))) {
            callback(MLComputeGraphStatus_Error, nullptr, "Failed to dispatch operator", userdata);
            return;
        }

        Ref<NamedResultsBase> results = AcquireRef(new NamedResultsBase());
        for (size_t i = 0; i < outputNames.size(); ++i) {
            std::string outputName = outputNames[i];
            pydml::TensorData* tensor = outputTensors[i];
            void* outputBuffer = tensor->Get();
            size_t bufferLength = tensor->Size();
            std::vector<int32_t> dimensions;
            for (auto size : tensor->Desc()->sizes) {
                // convert from uint32_t to int32_t.
                dimensions.push_back(static_cast<int32_t>(size));
            }
            Ref<ResultBase> result = AcquireRef(new Result(outputBuffer, bufferLength, dimensions));
            results->Set(outputName.c_str(), result.Detach());
            if (outputs != nullptr) {
                const Output* output = outputs->GetRecords().at(outputName);
                if (output->size >= bufferLength) {
                    memcpy(output->buffer, outputBuffer, bufferLength);
                }
            }
            delete tensor;
        }
        callback(MLComputeGraphStatus_Success, reinterpret_cast<MLNamedResults>(results.Detach()),
                 nullptr, userdata);
        return;
    }

}}  // namespace webnn_native::dml
