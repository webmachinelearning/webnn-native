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

#include "webnn_native/xnnpack/GraphXNN.h"

#include <math.h>
#include <numeric>

#include "common/Assert.h"
#include "common/Log.h"
#include "webnn_native/ErrorData.h"
#include "webnn_native/NamedInputs.h"
#include "webnn_native/NamedOutputs.h"
#include "webnn_native/NamedResults.h"
#include "webnn_native/Operand.h"
#include "webnn_native/Result.h"
#include "webnn_native/xnnpack/ContextXNN.h"

#define FAILED(status) (((xnn_status)(status)) != xnn_status_success)

#define XNNPACK_MAX_VALUE_ID 10000

const char* xnn_status2str(xnn_status v) {
    if (v == xnn_status_success)
        return "success";
    if (v == xnn_status_uninitialized)
        return "uninitialized";
    if (v == xnn_status_invalid_parameter)
        return "invalid_parameter";
    if (v == xnn_status_invalid_state)
        return "invalid_state";
    if (v == xnn_status_unsupported_parameter)
        return "unsupported_parameter";
    if (v == xnn_status_unsupported_hardware)
        return "unsupported_hardware";
    if (v == xnn_status_out_of_memory)
        return "out_of_memory";
    return "unknown status";
}

#define COMPLAIN_XNN_ERROR_AND_RETURN_XNN_ERROR(what, status)                             \
    do {                                                                                  \
        dawn::ErrorLog() << what << " returns XNNPACK error: " << xnn_status2str(status); \
        return status;                                                                    \
    } while (0)

#define XNN_TRY(f)                                           \
    do {                                                     \
        xnn_status s_ = f;                                   \
        if (s_ != xnn_status_success)                        \
            COMPLAIN_XNN_ERROR_AND_RETURN_XNN_ERROR(#f, s_); \
    } while (0)

#define COMPLAIN_XNN_ERROR_AND_RETURN_DAWN_ERROR(what, status)                              \
    do {                                                                                    \
        std::string message = std::string(what) + std::string(" returns XNNPACK error: ") + \
                              std::string(xnn_status2str(s_));                              \
        return DAWN_INTERNAL_ERROR(message.c_str());                                        \
    } while (0)

#if defined(DAWN_TRY)
#    undef DAWN_TRY
#endif

#define DAWN_TRY(f)                                           \
    do {                                                      \
        xnn_status s_ = f;                                    \
        if (s_ != xnn_status_success)                         \
            COMPLAIN_XNN_ERROR_AND_RETURN_DAWN_ERROR(#f, s_); \
    } while (0)

#define COMPLAIN_XNN_ERROR_AND_CALLBACK(what, status)                                       \
    do {                                                                                    \
        std::string message = std::string(what) + std::string(" returns XNNPACK error: ") + \
                              std::string(xnn_status2str(s_));                              \
        if (callback) {                                                                     \
            callback(MLComputeGraphStatus_Error, nullptr, message.c_str(), userdata);       \
            return MLComputeGraphStatus_Error;                                              \
        } else {                                                                            \
            dawn::ErrorLog() << message;                                                    \
            return MLComputeGraphStatus_Error;                                              \
        }                                                                                   \
    } while (0)

#define COMPLAIN_AND_CALLBACK(what)                                                   \
    do {                                                                              \
        std::string message = std::string(what);                                      \
        if (callback) {                                                               \
            callback(MLComputeGraphStatus_Error, nullptr, message.c_str(), userdata); \
            return MLComputeGraphStatus_Error;                                        \
        } else {                                                                      \
            dawn::ErrorLog() << message;                                              \
            return MLComputeGraphStatus_Error;                                        \
        }                                                                             \
    } while (0)

#define CALLBACK_TRY(f)                              \
    do {                                             \
        xnn_status s_ = f;                           \
        if (s_ != xnn_status_success)                \
            COMPLAIN_XNN_ERROR_AND_CALLBACK(#f, s_); \
    } while (0)

namespace webnn_native { namespace xnnpack {

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
        xnn_status GetXnnDataType(ml::OperandType operandType, xnn_datatype& xnnDataType) {
            if (operandType == ml::OperandType::Float32) {
                xnnDataType = xnn_datatype_fp32;
            } else {
                return xnn_status_invalid_parameter;
            }
            return xnn_status_success;
        }

        size_t SizeOfXnnDataType(xnn_datatype dataType) {
            if (dataType == xnn_datatype_fp32) {
                return sizeof(float);
            } else if (dataType == xnn_datatype_fp16) {
                return sizeof(uint16_t);
            } else if (dataType == xnn_datatype_qint32) {
                return sizeof(int32_t);
            } else if (dataType == xnn_datatype_qint8) {
                return sizeof(int8_t);
            }
            return 0;
        }

        xnn_status BroadcastDimensions(const std::vector<size_t>& aDims,
                                       const std::vector<size_t>& bDims,
                                       std::vector<size_t>& cDims) {
            cDims.resize(std::max(aDims.size(), bDims.size()));
            for (size_t i = 0; i < cDims.size(); ++i) {
                size_t aDim = i < aDims.size() ? aDims[aDims.size() - i - 1] : 1;
                size_t bDim = i < bDims.size() ? bDims[bDims.size() - i - 1] : 1;
                size_t cIndex = cDims.size() - i - 1;
                if (aDim == 1 && bDim != 1) {
                    cDims[cIndex] = bDim;
                } else if (aDim != 1 && bDim == 1) {
                    cDims[cIndex] = aDim;
                } else if (aDim == bDim) {
                    cDims[cIndex] = aDim;
                } else {
                    return xnn_status_invalid_parameter;
                }
            }
            return xnn_status_success;
        }

        size_t GetEffectiveFilterSize(size_t filterSize, size_t dilation) {
            if (dilation <= 1) {
                return filterSize;
            }
            return filterSize + (filterSize - 1) * (dilation - 1);
        }

        size_t ComputeConv2DOutputSize(size_t input,
                                       size_t filter,
                                       size_t padBegin,
                                       size_t padEnd,
                                       size_t stride,
                                       size_t dilation) {
            size_t effectiveFilter = GetEffectiveFilterSize(filter, dilation);
            return (input - effectiveFilter + padBegin + padEnd) / stride + 1;
        }
    }  // anonymous namespace

    Graph::Graph(Context* context) : GraphBase(context), mXnnOperator(nullptr) {
    }

    Graph::~Graph() {
        if (mXnnOperator) {
            if (FAILED(xnn_delete_operator(mXnnOperator))) {
                dawn::ErrorLog() << "xnn_delete_operator failed.";
            }
        }
    }

    MaybeError Graph::AddConstant(const op::Constant* constant) {
        std::shared_ptr<OperandInfo> info = std::make_shared<OperandInfo>(OperandType::CONSTANT);
        const OperandDescriptor* desc = constant->GetOperandDescriptor();
        DAWN_TRY(GetXnnDataType(desc->type, info->dataType));
        info->dims.assign(desc->dimensions, desc->dimensions + desc->dimensionsCount);
        info->buffer.reset(new char[constant->GetSize()]);
        if (info->buffer.get() == nullptr) {
            return DAWN_OUT_OF_MEMORY_ERROR("");
        }
        memcpy(info->buffer.get(), constant->GetValue(), constant->GetSize());
        mConstants.push_back(info);
        mOperandInfoMap.insert(std::make_pair(constant, info));
        return {};
    }

    MaybeError Graph::AddInput(const op::Input* input) {
        std::shared_ptr<OperandInfo> info = std::make_shared<OperandInfo>(OperandType::INPUT);
        const OperandDescriptor* desc = input->GetOperandDescriptor();
        DAWN_TRY(GetXnnDataType(desc->type, info->dataType));
        info->dims.assign(desc->dimensions, desc->dimensions + desc->dimensionsCount);
        info->name = input->GetName();
        mOperandInfoMap.insert(std::make_pair(input, info));
        return {};
    }

    MaybeError Graph::AddOutput(const std::string& name, const OperandBase* op) {
        std::shared_ptr<OperandInfo>& info = mOperandInfoMap.at(op);
        if (info->opType == OperandType::INPUT || info->opType == OperandType::CONSTANT) {
            return DAWN_INTERNAL_ERROR("There is no operator to be created.");
        }
        info->name = std::move(name);
        return {};
    }

    MaybeError Graph::AddBinary(const op::Binary* binary) {
        std::shared_ptr<OperandInfo> info = std::make_shared<OperandInfo>(OperandType::BINARY);
        mOperandInfoMap.insert(std::make_pair(binary, info));
        mOperandsToBuild.push_back(binary);
        return {};
    }

    MaybeError Graph::AddClamp(const op::Clamp* clamp) {
        std::shared_ptr<OperandInfo> info = std::make_shared<OperandInfo>(OperandType::CLAMP);
        mOperandInfoMap.insert(std::make_pair(clamp, info));
        mOperandsToBuild.push_back(clamp);
        return {};
    }

    MaybeError Graph::AddConv2d(const op::Conv2d* conv2d) {
        std::shared_ptr<OperandInfo> info = std::make_shared<OperandInfo>(OperandType::CONV2D);
        mOperandInfoMap.insert(std::make_pair(conv2d, info));
        mOperandsToBuild.push_back(conv2d);
        return {};
    }

    MaybeError Graph::AddPool2d(const op::Pool2d* pool2d) {
        std::shared_ptr<OperandInfo> info = std::make_shared<OperandInfo>(OperandType::POOL2D);
        mOperandInfoMap.insert(std::make_pair(pool2d, info));
        mOperandsToBuild.push_back(pool2d);
        return {};
    }

    MaybeError Graph::AddUnary(const op::Unary* unary) {
        std::shared_ptr<OperandInfo> info = std::make_shared<OperandInfo>(OperandType::UNARY);
        mOperandInfoMap.insert(std::make_pair(unary, info));
        mOperandsToBuild.push_back(unary);
        return {};
    }

    MaybeError Graph::Finish() {
        if (mOperandsToBuild.size() == 0) {
            return DAWN_INTERNAL_ERROR("No operators to build.");
        }
        const OperandBase* op = mOperandsToBuild[0];
        DAWN_ASSERT(mOperandInfoMap.find(op) != mOperandInfoMap.end());
        std::shared_ptr<OperandInfo>& info = mOperandInfoMap.at(op);
        if (mOperandsToBuild.size() == 1) {
            if (info->opType == OperandType::UNARY) {
                DAWN_TRY(CreateXnnOp(reinterpret_cast<const op::Unary*>(op)));
            } else if (info->opType == OperandType::CLAMP) {
                DAWN_TRY(CreateXnnOp(reinterpret_cast<const op::Clamp*>(op)));
            } else if (info->opType == OperandType::BINARY) {
                DAWN_TRY(CreateXnnOp(reinterpret_cast<const op::Binary*>(op)));
            } else if (info->opType == OperandType::CONV2D) {
                DAWN_TRY(CreateXnnOp(reinterpret_cast<const op::Conv2d*>(op)));
            } else if (info->opType == OperandType::POOL2D) {
                DAWN_TRY(CreateXnnOp(reinterpret_cast<const op::Pool2d*>(op)));
            } else {
                return DAWN_UNIMPLEMENTED_ERROR("");
            }
        } else if (info->opType == OperandType::CONV2D) {
            // Try to fuse add and clamp into conv2d
            const op::Conv2d* conv2d = reinterpret_cast<const op::Conv2d*>(op);
            if (mOperandsToBuild.size() > 3) {
                return DAWN_INTERNAL_ERROR("Cannot fuse conv2d subgraph with more than 3 ops.");
            }
            const op::Binary* add = nullptr;
            const op::Clamp* clamp = nullptr;
            for (auto& operand : mOperandsToBuild) {
                DAWN_ASSERT(mOperandInfoMap.find(operand) != mOperandInfoMap.end());
                std::shared_ptr<OperandInfo>& operandInfo = mOperandInfoMap.at(operand);
                if (operandInfo->opType == OperandType::BINARY &&
                    reinterpret_cast<const op::Binary*>(operand)->GetType() ==
                        op::BinaryOpType::kAdd) {
                    add = reinterpret_cast<const op::Binary*>(operand);
                } else if (operandInfo->opType == OperandType::CLAMP) {
                    clamp = reinterpret_cast<const op::Clamp*>(operand);
                }
            }
            if ((mOperandsToBuild.size() == 2 && !add && !clamp) ||
                (mOperandsToBuild.size() == 3 && (!add || !clamp))) {
                return DAWN_INTERNAL_ERROR("Failed to fuse conv2d subgraph.");
            }
            DAWN_TRY(CreateXnnOp(conv2d, add, clamp));
        }
        return {};
    }

    xnn_status Graph::CreateXnnOp(const op::Unary* unary) {
        DAWN_ASSERT(unary->Inputs().size() == 1);
        const OperandBase* inputOperand = unary->Inputs()[0].Get();
        DAWN_ASSERT(mOperandInfoMap.find(inputOperand) != mOperandInfoMap.end());
        const std::shared_ptr<OperandInfo>& inputInfo = mOperandInfoMap.at(inputOperand);
        mInputs.push_back(inputInfo);
        if (inputInfo->opType == OperandType::INPUT) {
            mExternalInputs.insert(std::make_pair(inputInfo->name, mInputs.size() - 1));
        }
        if (unary->GetType() == op::UnaryOpType::kRelu) {
            XNN_TRY(xnn_create_clamp_nc_f32(1, 1, 1, 0, +std::numeric_limits<float>::infinity(), 0,
                                            &mXnnOperator));
            mXnnOperatorType = XnnOpType::clamp_nc_f32;
        } else {
            return xnn_status_unsupported_parameter;
        }
        std::shared_ptr<OperandInfo>& outputInfo = mOperandInfoMap.at(unary);
        outputInfo->dataType = inputInfo->dataType;
        outputInfo->dims = inputInfo->dims;
        mOutputs.push_back(outputInfo);
        mExternalOutputs.insert(std::make_pair(outputInfo->name, mOutputs.size() - 1));
        return xnn_status_success;
    }

    xnn_status Graph::CreateXnnOp(const op::Clamp* clamp) {
        const OperandBase* inputOperand = clamp->Inputs()[0].Get();
        DAWN_ASSERT(mOperandInfoMap.find(inputOperand) != mOperandInfoMap.end());
        const std::shared_ptr<OperandInfo>& inputInfo = mOperandInfoMap.at(inputOperand);
        mInputs.push_back(inputInfo);
        if (inputInfo->opType == OperandType::INPUT) {
            mExternalInputs.insert(std::make_pair(inputInfo->name, mInputs.size() - 1));
        }
        const ClampOptions* options = clamp->GetOptions();
        float minValue = -std::numeric_limits<float>::infinity();
        if (options->minValue != nullptr) {
            const std::shared_ptr<OperandInfo>& minInfo = mOperandInfoMap.at(options->minValue);
            if (minInfo->opType != OperandType::CONSTANT) {
                dawn::ErrorLog() << "XNNPACK only supports clamp by value.";
                return xnn_status_invalid_parameter;
            }
            minValue = (reinterpret_cast<float*>(minInfo->buffer.get()))[0];
        }
        float maxValue = +std::numeric_limits<float>::infinity();
        if (options->maxValue != nullptr) {
            const std::shared_ptr<OperandInfo>& maxInfo = mOperandInfoMap.at(options->maxValue);
            if (maxInfo->opType != OperandType::CONSTANT) {
                dawn::ErrorLog() << "XNNPACK only supports clamp by value.";
                return xnn_status_invalid_parameter;
            }
            maxValue = (reinterpret_cast<float*>(maxInfo->buffer.get()))[0];
        }
        XNN_TRY(xnn_create_clamp_nc_f32(1, 1, 1, minValue, maxValue, 0, &mXnnOperator));
        mXnnOperatorType = XnnOpType::clamp_nc_f32;
        std::shared_ptr<OperandInfo>& outputInfo = mOperandInfoMap.at(clamp);
        outputInfo->dataType = inputInfo->dataType;
        outputInfo->dims = inputInfo->dims;
        mOutputs.push_back(outputInfo);
        mExternalOutputs.insert(std::make_pair(outputInfo->name, mOutputs.size() - 1));
        return xnn_status_success;
    }

    xnn_status Graph::CreateXnnOp(const op::Binary* binary) {
        DAWN_ASSERT(binary->Inputs().size() == 2);
        const OperandBase* input0Operand = binary->Inputs()[0].Get();
        DAWN_ASSERT(mOperandInfoMap.find(input0Operand) != mOperandInfoMap.end());
        const std::shared_ptr<OperandInfo>& input0Info = mOperandInfoMap.at(input0Operand);
        mInputs.push_back(input0Info);
        if (input0Info->opType == OperandType::INPUT) {
            mExternalInputs.insert(std::make_pair(input0Info->name, mInputs.size() - 1));
        }
        const OperandBase* input1Operand = binary->Inputs()[1].Get();
        DAWN_ASSERT(mOperandInfoMap.find(input1Operand) != mOperandInfoMap.end());
        const std::shared_ptr<OperandInfo>& input1Info = mOperandInfoMap.at(input1Operand);
        mInputs.push_back(input1Info);
        if (input1Info->opType == OperandType::INPUT) {
            mExternalInputs.insert(std::make_pair(input1Info->name, mInputs.size() - 1));
        }
        const float outputMin = -std::numeric_limits<float>::infinity();
        const float outputMax = +std::numeric_limits<float>::infinity();
        if (binary->GetType() == op::BinaryOpType::kAdd) {
            XNN_TRY(xnn_create_add_nd_f32(outputMin, outputMax, 0, &mXnnOperator));
            mXnnOperatorType = XnnOpType::add_nd_f32;
        } else if (binary->GetType() == op::BinaryOpType::kMul) {
            XNN_TRY(xnn_create_multiply_nd_f32(outputMin, outputMax, 0, &mXnnOperator));
            mXnnOperatorType = XnnOpType::multiply_nd_f32;
        } else if (binary->GetType() == op::BinaryOpType::kSub) {
            XNN_TRY(xnn_create_subtract_nd_f32(outputMin, outputMax, 0, &mXnnOperator));
            mXnnOperatorType = XnnOpType::subtract_nd_f32;
        } else {
            return xnn_status_unsupported_parameter;
        }
        std::shared_ptr<OperandInfo>& outputInfo = mOperandInfoMap.at(binary);
        outputInfo->dataType = input0Info->dataType;
        XNN_TRY(BroadcastDimensions(input0Info->dims, input1Info->dims, outputInfo->dims));
        mOutputs.push_back(outputInfo);
        mExternalOutputs.insert(std::make_pair(outputInfo->name, mOutputs.size() - 1));
        return xnn_status_success;
    }

    xnn_status Graph::CreateXnnOp(const op::Pool2d* pool2d) {
        DAWN_ASSERT(pool2d->Inputs().size() == 1);
        const OperandBase* inputOperand = pool2d->Inputs()[0].Get();
        DAWN_ASSERT(mOperandInfoMap.find(inputOperand) != mOperandInfoMap.end());
        const std::shared_ptr<OperandInfo>& inputInfo = mOperandInfoMap.at(inputOperand);
        mInputs.push_back(inputInfo);
        if (inputInfo->opType == OperandType::INPUT) {
            mExternalInputs.insert(std::make_pair(inputInfo->name, mInputs.size() - 1));
        }
        const Pool2dOptions* options = pool2d->GetOptions();
        if (options->layout != ml::InputOperandLayout::Nhwc) {
            dawn::ErrorLog() << "XNNPACK only supports input layout nhwc.";
            return xnn_status_invalid_parameter;
        }
        uint32_t strideHeight = options->strides[0];
        uint32_t strideWidth = options->strides[1];
        uint32_t dilationHeight = options->dilations[0];
        uint32_t dilationWidth = options->dilations[1];
        // nhwc
        size_t inputHeight = inputInfo->dims[1];
        size_t inputWidth = inputInfo->dims[2];
        size_t channels = inputInfo->dims[3];
        uint32_t filterHeight, filterWidth;
        if (options->windowDimensions != nullptr) {
            filterHeight = options->windowDimensions[0];
            filterWidth = options->windowDimensions[1];
        } else {
            filterHeight = inputHeight;
            filterWidth = inputWidth;
        }

        size_t outputHeight, outputWidth;
        uint32_t padTop, padBottom, padLeft, padRight;
        if (options->autoPad == ml::AutoPad::Explicit) {
            // WebNN padding: [beginning_height, ending_height, beginning_width, ending_width]
            padTop = options->padding[0];
            padBottom = options->padding[1];
            padLeft = options->padding[2];
            padRight = options->padding[3];
            outputHeight = ComputeConv2DOutputSize(inputHeight, filterHeight, padTop, padBottom,
                                                   strideHeight, dilationHeight);
            outputWidth = ComputeConv2DOutputSize(inputWidth, filterWidth, padLeft, padRight,
                                                  strideWidth, dilationWidth);
        } else {
            outputHeight = ceil(inputHeight / strideHeight);
            outputWidth = ceil(inputWidth / strideWidth);
            size_t padAlongHeight =
                std::max(size_t(0), (outputHeight - 1) * strideHeight + filterHeight - inputHeight);
            size_t padAlongWidth =
                std::max(size_t(0), (outputWidth - 1) * strideWidth + filterWidth - inputWidth);
            if (options->autoPad == ml::AutoPad::SameUpper) {
                padTop = floor(padAlongHeight / 2);
                padBottom = padAlongHeight - padTop;
                padLeft = floor(padAlongWidth / 2);
                padRight = padAlongWidth - padLeft;
            } else {
                padBottom = floor(padAlongHeight / 2);
                padTop = padAlongHeight - padBottom;
                padRight = floor(padAlongWidth / 2);
                padLeft = padAlongWidth - padRight;
            }
        }

        float outputMin = -std::numeric_limits<float>::infinity();
        float outputMax = +std::numeric_limits<float>::infinity();
        const uint32_t flags = 0;
        if (pool2d->GetType() == op::Pool2dType::kAveragePool2d) {
            if (dilationHeight != 1 || dilationWidth != 1) {
                dawn::ErrorLog() << "XNNPACK does not support dilation for averagePool2d.";
                return xnn_status_invalid_parameter;
            }
            XNN_TRY(xnn_create_average_pooling2d_nhwc_f32(
                padTop, padRight, padBottom, padLeft, filterHeight, filterWidth, strideHeight,
                strideWidth, channels, channels, channels, outputMin, outputMax, flags,
                &mXnnOperator));
            mXnnOperatorType = XnnOpType::average_pooling2d_nhwc_f32;
        } else if (pool2d->GetType() == op::Pool2dType::kMaxPool2d) {
            XNN_TRY(xnn_create_max_pooling2d_nhwc_f32(
                padTop, padRight, padBottom, padLeft, filterHeight, filterWidth, strideHeight,
                strideWidth, dilationHeight, dilationWidth, channels, channels, channels, outputMin,
                outputMax, flags, &mXnnOperator));
            mXnnOperatorType = XnnOpType::max_pooling2d_nhwc_f32;
        } else {
            dawn::ErrorLog() << "XNNPACK does not support l2Pool2d.";
            return xnn_status_invalid_parameter;
        }
        const std::shared_ptr<OperandInfo> outputInfo = mOperandInfoMap.at(pool2d);
        outputInfo->dataType = inputInfo->dataType;
        size_t batchSize = inputInfo->dims[0];
        // nchw
        outputInfo->dims = {batchSize, outputHeight, outputWidth, channels};
        mOutputs.push_back(outputInfo);
        mExternalOutputs.insert(std::make_pair(outputInfo->name, mOutputs.size() - 1));
        return xnn_status_success;
    }

    xnn_status Graph::CreateXnnOp(const op::Conv2d* conv2d,
                                  const op::Binary* add,
                                  const op::Clamp* clamp) {
        DAWN_ASSERT(conv2d->Inputs().size() == 2);
        const OperandBase* inputOperand = conv2d->Inputs()[0].Get();
        DAWN_ASSERT(mOperandInfoMap.find(inputOperand) != mOperandInfoMap.end());
        const std::shared_ptr<OperandInfo>& inputInfo = mOperandInfoMap.at(inputOperand);
        mInputs.push_back(inputInfo);
        if (inputInfo->opType == OperandType::INPUT) {
            mExternalInputs.insert(std::make_pair(inputInfo->name, mInputs.size() - 1));
        }
        const OperandBase* filterOperand = conv2d->Inputs()[1].Get();
        DAWN_ASSERT(mOperandInfoMap.find(filterOperand) != mOperandInfoMap.end());
        const std::shared_ptr<OperandInfo>& filterInfo = mOperandInfoMap.at(filterOperand);
        if (filterInfo->opType != OperandType::CONSTANT) {
            dawn::ErrorLog() << "filter is not a constant.";
            return xnn_status_invalid_parameter;
        }
        const float* filter = reinterpret_cast<const float*>(filterInfo->buffer.get());

        const Conv2dOptions* options = conv2d->GetOptions();
        uint32_t groups = options->groups;
        uint32_t strideHeight = options->strides[0];
        uint32_t strideWidth = options->strides[1];
        uint32_t dilationHeight = options->dilations[0];
        uint32_t dilationWidth = options->dilations[1];
        size_t inputHeight, inputWidth;
        uint32_t filterHeight, filterWidth;
        size_t inputChannels, outputChannels;
        if (options->inputLayout == ml::InputOperandLayout::Nhwc) {
            inputHeight = inputInfo->dims[1];
            inputWidth = inputInfo->dims[2];
            inputChannels = inputInfo->dims[3];
            if (groups != 1 && groups == inputChannels) {
                // For depthwiseConv2d, xnn pack expects the weights layout hwio
                //   [filter_height, filter_width, input_channels, channel_multiplier]
                if (options->filterLayout != ml::FilterOperandLayout::Hwio) {
                    dawn::ErrorLog()
                        << "XNNPACK only supports filter layout hwio for depthwise conv2d.";
                    return xnn_status_invalid_parameter;
                }
                if (filterInfo->dims[2] != 1) {
                    dawn::ErrorLog() << "The filter layout is invalid.";
                    return xnn_status_invalid_parameter;
                }
                filterHeight = filterInfo->dims[0];
                filterWidth = filterInfo->dims[1];
                outputChannels = filterInfo->dims[3];
            } else {
                // For regular conv2d, xnn pack expects weights layed out like:
                //   [output_channels, filter_height, filter_width, input_channels]
                if (options->filterLayout != ml::FilterOperandLayout::Ohwi) {
                    dawn::ErrorLog() << "XNNPACK only supports filter layout ohwi for conv2d.";
                    return xnn_status_invalid_parameter;
                }
                if (filterInfo->dims[3] != inputChannels) {
                    dawn::ErrorLog() << "The filter layout is invalid.";
                    return xnn_status_invalid_parameter;
                }
                outputChannels = filterInfo->dims[0];
                filterHeight = filterInfo->dims[1];
                filterWidth = filterInfo->dims[2];
            }
        } else {
            dawn::ErrorLog() << "XNNPACK only supports input layout nhwc.";
            return xnn_status_invalid_parameter;
        }
        const size_t inputChannelStride = inputChannels;
        const size_t outputChannelStride = outputChannels;
        size_t groupInputChannels;
        size_t groupOutputChannels;
        uint32_t flags = 0;
        if (groups == 1) {
            groupInputChannels = inputChannels;
            groupOutputChannels = outputChannels;
        } else if (groups == inputChannels) {
            groupInputChannels = 1;
            groupOutputChannels = outputChannels / groups;
            flags |= XNN_FLAG_DEPTHWISE_CONVOLUTION;
        } else {
            // FIXME(nhu): implement the grouped conv2d.
            dawn::ErrorLog() << "Grouped conv2d is unimplemented.";
            return xnn_status_unsupported_parameter;
        }

        size_t outputHeight, outputWidth;
        uint32_t padTop, padBottom, padLeft, padRight;
        if (options->autoPad == ml::AutoPad::Explicit) {
            // WebNN padding: [beginning_height, ending_height, beginning_width, ending_width]
            padTop = options->padding[0];
            padBottom = options->padding[1];
            padLeft = options->padding[2];
            padRight = options->padding[3];
            outputHeight = ComputeConv2DOutputSize(inputHeight, filterHeight, padTop, padBottom,
                                                   strideHeight, dilationHeight);
            outputWidth = ComputeConv2DOutputSize(inputWidth, filterWidth, padLeft, padRight,
                                                  strideWidth, dilationWidth);
        } else {
            outputHeight = ceil(inputHeight / strideHeight);
            outputWidth = ceil(inputWidth / strideWidth);
            size_t padAlongHeight =
                std::max(size_t(0), (outputHeight - 1) * strideHeight + filterHeight - inputHeight);
            size_t padAlongWidth =
                std::max(size_t(0), (outputWidth - 1) * strideWidth + filterWidth - inputWidth);
            if (options->autoPad == ml::AutoPad::SameUpper) {
                padTop = floor(padAlongHeight / 2);
                padBottom = padAlongHeight - padTop;
                padLeft = floor(padAlongWidth / 2);
                padRight = padAlongWidth - padLeft;
            } else {
                padBottom = floor(padAlongHeight / 2);
                padTop = padAlongHeight - padBottom;
                padRight = floor(padAlongWidth / 2);
                padLeft = padAlongWidth - padRight;
            }
        }

        const float* bias = nullptr;
        if (add) {
            DAWN_ASSERT(add->Inputs().size() == 2);
            OperandBase* biasOperand = nullptr;
            if (conv2d == add->Inputs()[0].Get()) {
                biasOperand = add->Inputs()[1].Get();
            } else if (conv2d == add->Inputs()[1].Get()) {
                biasOperand = add->Inputs()[0].Get();
            } else {
                dawn::ErrorLog() << "The add is not fusable.";
                return xnn_status_invalid_parameter;
            }
            DAWN_ASSERT(mOperandInfoMap.find(biasOperand) != mOperandInfoMap.end());
            const std::shared_ptr<OperandInfo>& biasInfo = mOperandInfoMap.at(biasOperand);
            if (biasInfo->opType != OperandType::CONSTANT) {
                dawn::ErrorLog() << "bias is not a constant.";
                return xnn_status_invalid_parameter;
            }
            if (biasInfo->dims.size() != 1 && biasInfo->dims[0] != outputChannels) {
                dawn::ErrorLog() << "bias dimensions is invalid.";
                return xnn_status_invalid_parameter;
            }
            bias = reinterpret_cast<const float*>(biasInfo->buffer.get());
        }

        float outputMin = -std::numeric_limits<float>::infinity();
        float outputMax = +std::numeric_limits<float>::infinity();
        if (clamp) {
            if (add) {
                if (add != clamp->Inputs()[0].Get()) {
                    dawn::ErrorLog() << "The clamp is not fusable.";
                    return xnn_status_invalid_parameter;
                }
            } else {
                if (conv2d != clamp->Inputs()[0].Get()) {
                    dawn::ErrorLog() << "The clamp is not fusable.";
                    return xnn_status_invalid_parameter;
                }
            }
            const ClampOptions* options = clamp->GetOptions();
            if (options->minValue != nullptr) {
                const std::shared_ptr<OperandInfo>& minInfo = mOperandInfoMap.at(options->minValue);
                if (minInfo->opType != OperandType::CONSTANT) {
                    dawn::ErrorLog() << "XNNPACK only supports clamp by value.";
                    return xnn_status_invalid_parameter;
                }
                outputMin = (reinterpret_cast<float*>(minInfo->buffer.get()))[0];
            }
            if (options->maxValue != nullptr) {
                const std::shared_ptr<OperandInfo>& maxInfo = mOperandInfoMap.at(options->maxValue);
                if (maxInfo->opType != OperandType::CONSTANT) {
                    dawn::ErrorLog() << "XNNPACK only supports clamp by value.";
                    return xnn_status_invalid_parameter;
                }
                outputMax = (reinterpret_cast<float*>(maxInfo->buffer.get()))[0];
            }
        }

        XNN_TRY(xnn_create_convolution2d_nhwc_f32(
            padTop, padRight, padBottom, padLeft, filterHeight, filterWidth, strideHeight,
            strideWidth, dilationHeight, dilationWidth, groups, groupInputChannels,
            groupOutputChannels, inputChannelStride, outputChannelStride, filter, bias, outputMin,
            outputMax, flags, &mXnnOperator));
        mXnnOperatorType = XnnOpType::convolution2d_nhwc_f32;
        std::shared_ptr<OperandInfo> outputInfo;
        if (clamp) {
            outputInfo = mOperandInfoMap.at(clamp);
        } else if (add) {
            outputInfo = mOperandInfoMap.at(add);
        } else {
            outputInfo = mOperandInfoMap.at(conv2d);
        }
        outputInfo->dataType = inputInfo->dataType;
        size_t batchSize = inputInfo->dims[0];
        outputInfo->dims = {batchSize, outputHeight, outputWidth, outputChannels};
        mOutputs.push_back(outputInfo);
        mExternalOutputs.insert(std::make_pair(outputInfo->name, mOutputs.size() - 1));
        return xnn_status_success;
    }

    size_t Graph::SizeOfOperandInfo(const std::shared_ptr<OperandInfo>& info) {
        return std::accumulate(info->dims.begin(), info->dims.end(), 1, std::multiplies<size_t>()) *
               SizeOfXnnDataType(info->dataType);
    }

    pthreadpool_t Graph::GetThreadpool() {
        return reinterpret_cast<Context*>(GetContext())->GetThreadpool();
    }

    void Graph::CompileImpl(BuildGraphCallbackDelegate delegate) {
        delegate(MLBuildGraphStatus_Success, this);
        return;
    }

    MLBuildGraphStatus Graph::CompileSyncImpl() {
        return MLBuildGraphStatus_Success;
    }

    MLComputeGraphStatus Graph::ComputeSyncImpl(NamedInputsBase* inputs,
                                                NamedOutputsBase* outputs) {
        return this->GenericComputeImpl(inputs, outputs);
    }

    void Graph::ComputeImpl(NamedInputsBase* inputs,
                            MLComputeGraphCallback callback,
                            void* userdata,
                            NamedOutputsBase* outputs) {
        this->GenericComputeImpl(inputs, outputs, callback, userdata);
    }

    MLComputeGraphStatus Graph::GenericComputeImpl(NamedInputsBase* inputs,
                                                   NamedOutputsBase* outputs,
                                                   MLComputeGraphCallback callback,
                                                   void* userdata) {
        std::vector<const void*> inputBuffers(mInputs.size(), nullptr);
        for (size_t i = 0; i < mInputs.size(); ++i) {
            if (mInputs[i]->opType == OperandType::CONSTANT) {
                inputBuffers[i] = mInputs[i]->buffer.get();
            }
        }
        for (auto& input : inputs->GetRecords()) {
            if (mExternalInputs.find(input.first) == mExternalInputs.end()) {
                COMPLAIN_AND_CALLBACK("Invalid parameters.");
            }
            size_t index = mExternalInputs.at(input.first);
            inputBuffers[index] = input.second->buffer;
        }

        std::vector<std::string> outputNames;
        if (outputs != nullptr) {
            for (auto& output : outputs->GetRecords()) {
                outputNames.push_back(output.first);
            }
        } else {
            for (auto& output : mExternalOutputs) {
                outputNames.push_back(output.first);
            }
        }

        std::vector<void*> outputBuffers(mOutputs.size(), nullptr);
        Ref<NamedResultsBase> results = AcquireRef(new NamedResultsBase());
        for (size_t i = 0; i < outputNames.size(); ++i) {
            std::string outputName = outputNames[i];
            size_t outputIndex = mExternalOutputs.at(outputName);
            const std::shared_ptr<OperandInfo>& outputInfo = mOutputs[outputIndex];
            std::vector<int32_t> dimensions(outputInfo->dims.begin(), outputInfo->dims.end());
            size_t bufferLength = SizeOfOperandInfo(outputInfo);
            Ref<ResultBase> result;
            if (outputs != nullptr) {
                if (outputs->GetRecords().find(outputName) != outputs->GetRecords().end()) {
                    const Output* output = outputs->GetRecords().at(outputName);
                    if (output->buffer != nullptr) {
                        DAWN_ASSERT(output->size >= bufferLength);
                        outputBuffers[outputIndex] = output->buffer;
                        result =
                            AcquireRef(new ResultBase(output->buffer, bufferLength, dimensions));
                    }
                }
            }
            if (outputBuffers[outputIndex] == nullptr) {
                void* outputBuffer = malloc(bufferLength);
                outputBuffers[outputIndex] = outputBuffer;
                result = AcquireRef(new Result(outputBuffer, bufferLength, dimensions));
            }
            results->Set(outputName.c_str(), result.Detach());
        }

        if (mXnnOperatorType == XnnOpType::convolution2d_nhwc_f32 ||
            mXnnOperatorType == XnnOpType::average_pooling2d_nhwc_f32 ||
            mXnnOperatorType == XnnOpType::max_pooling2d_nhwc_f32) {
            std::vector<size_t> inputDims = mInputs[0]->dims;
            if (!inputBuffers[0] || !outputBuffers[0]) {
                COMPLAIN_AND_CALLBACK("Invalid parameters.");
            }
            const float* input = reinterpret_cast<const float*>(inputBuffers[0]);
            float* output = reinterpret_cast<float*>(outputBuffers[0]);
            size_t batchSize = inputDims[0];
            size_t inputHeight = inputDims[1];
            size_t inputWidth = inputDims[2];
            if (mXnnOperatorType == XnnOpType::convolution2d_nhwc_f32) {
                CALLBACK_TRY(xnn_setup_convolution2d_nhwc_f32(mXnnOperator, batchSize, inputHeight,
                                                              inputWidth, input, output,
                                                              GetThreadpool()));
            } else if (mXnnOperatorType == XnnOpType::average_pooling2d_nhwc_f32) {
                CALLBACK_TRY(xnn_setup_average_pooling2d_nhwc_f32(mXnnOperator, batchSize,
                                                                  inputHeight, inputWidth, input,
                                                                  output, GetThreadpool()));
            } else if (mXnnOperatorType == XnnOpType::max_pooling2d_nhwc_f32) {
                CALLBACK_TRY(xnn_setup_max_pooling2d_nhwc_f32(mXnnOperator, batchSize, inputHeight,
                                                              inputWidth, input, output,
                                                              GetThreadpool()));
            }
        } else if (mXnnOperatorType == XnnOpType::clamp_nc_f32) {
            const std::shared_ptr<OperandInfo>& outputInfo = mOutputs[0];
            size_t batchSize = std::accumulate(outputInfo->dims.begin(), outputInfo->dims.end(), 1,
                                               std::multiplies<size_t>());
            if (!inputBuffers[0] || !outputBuffers[0]) {
                COMPLAIN_AND_CALLBACK("Invalid parameters.");
            }
            const float* input = reinterpret_cast<const float*>(inputBuffers[0]);
            float* output = reinterpret_cast<float*>(outputBuffers[0]);
            CALLBACK_TRY(
                xnn_setup_clamp_nc_f32(mXnnOperator, batchSize, input, output, GetThreadpool()));
        } else if (mXnnOperatorType == XnnOpType::add_nd_f32 ||
                   mXnnOperatorType == XnnOpType::multiply_nd_f32 ||
                   mXnnOperatorType == XnnOpType::subtract_nd_f32) {
            std::vector<size_t> dims0 = mInputs[0]->dims;
            std::vector<size_t> dims1 = mInputs[1]->dims;
            if (!inputBuffers[0] || !inputBuffers[1] || !outputBuffers[0]) {
                COMPLAIN_AND_CALLBACK("Invalid parameters.");
            }
            const float* input0 = reinterpret_cast<const float*>(inputBuffers[0]);
            const float* input1 = reinterpret_cast<const float*>(inputBuffers[1]);
            float* output = reinterpret_cast<float*>(outputBuffers[0]);
            if (mXnnOperatorType == XnnOpType::add_nd_f32) {
                CALLBACK_TRY(xnn_setup_add_nd_f32(mXnnOperator, dims0.size(), dims0.data(),
                                                  dims1.size(), dims1.data(), input0, input1,
                                                  output, GetThreadpool()));
            } else if (mXnnOperatorType == XnnOpType::multiply_nd_f32) {
                CALLBACK_TRY(xnn_setup_multiply_nd_f32(mXnnOperator, dims0.size(), dims0.data(),
                                                       dims1.size(), dims1.data(), input0, input1,
                                                       output, GetThreadpool()));
            } else if (mXnnOperatorType == XnnOpType::subtract_nd_f32) {
                CALLBACK_TRY(xnn_setup_subtract_nd_f32(mXnnOperator, dims0.size(), dims0.data(),
                                                       dims1.size(), dims1.data(), input0, input1,
                                                       output, GetThreadpool()));
            }
        } else {
            COMPLAIN_AND_CALLBACK("The operator is not supported.");
        }

        CALLBACK_TRY(xnn_run_operator(mXnnOperator, GetThreadpool()));

        if (callback) {
            callback(MLComputeGraphStatus_Success,
                     reinterpret_cast<MLNamedResults>(results.Detach()), nullptr, userdata);
        }

        return MLComputeGraphStatus_Success;
    }

}}  // namespace webnn_native::xnnpack
