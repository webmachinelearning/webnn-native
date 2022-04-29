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
#include "webnn_native/Operand.h"
#include "webnn_native/xnnpack/ContextXNN.h"

#define FAILED(status) (((xnn_status)(status)) != xnn_status_success)

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

namespace webnn_native::xnnpack {

    namespace {
        xnn_status GetXnnDataType(wnn::OperandType operandType, xnn_datatype& xnnDataType) {
            if (operandType == wnn::OperandType::Float32) {
                xnnDataType = xnn_datatype_fp32;
            } else {
                return xnn_status_invalid_parameter;
            }
            return xnn_status_success;
        }
    }  // anonymous namespace

    Graph::Graph(Context* context)
        : GraphBase(context),
          mExternalId(0),
          mRuntime(nullptr),
          mNamedInputs(nullptr),
          mNamedOutputs(nullptr) {
    }

    Graph::~Graph() {
    }

    MaybeError Graph::AddInput(const op::Input* input) {
        mOperators.push_back({OperatorType::Input, input});
        uint32_t inputId = mExternalId++;
        mInputs.insert(std::make_pair(input->PrimaryOutput(), inputId));
        xnn_external_value externalValue = {inputId, nullptr};
        mExternals.insert(std::make_pair(input->GetName(), externalValue));
        return {};
    }

    MaybeError Graph::AddOutput(std::string_view name, const OperandBase* op) {
        uint32_t outputId = mExternalId++;
        mOutputs.insert(std::make_pair(op, outputId));
        xnn_external_value externalValue = {outputId, nullptr};
        mExternals.insert(std::make_pair(name, externalValue));
        return {};
    }

#define GRAPH_ADD_OP(OpType)                              \
    MaybeError Graph::Add##OpType(const op::OpType* op) { \
        mOperators.push_back({OperatorType::OpType, op}); \
        return {};                                        \
    }

    GRAPH_ADD_OP(Binary)
    GRAPH_ADD_OP(Clamp)
    GRAPH_ADD_OP(Concat)
    GRAPH_ADD_OP(Conv2d)
    GRAPH_ADD_OP(Constant)
    GRAPH_ADD_OP(Gemm)
    GRAPH_ADD_OP(Pad)
    GRAPH_ADD_OP(Pool2d)
    GRAPH_ADD_OP(Reshape)
    GRAPH_ADD_OP(Split)
    GRAPH_ADD_OP(Squeeze)
    GRAPH_ADD_OP(Unary)

    xnn_status Graph::DefineXnnTensorValue(xnn_subgraph_t subgraph,
                                           const OperandBase* operand,
                                           uint32_t* id,
                                           const void* data) {
        xnn_datatype datatype = xnn_datatype_invalid;
        if (GetXnnDataType(operand->Type(), datatype) != xnn_status_success) {
            // Ignore the unsupproted data type, it may be used for attributes, such as padding
            return xnn_status_success;
        }
        std::vector<size_t> dims;
        for (auto& d : operand->Shape()) {
            dims.push_back(static_cast<size_t>(d));
        }
        uint32_t flags = 0;
        uint32_t externalId;
        if (mInputs.find(operand) != mInputs.end()) {
            externalId = mInputs.at(operand);
            flags |= XNN_VALUE_FLAG_EXTERNAL_INPUT;
        } else if (mOutputs.find(operand) != mOutputs.end()) {
            externalId = mOutputs.at(operand);
            flags |= XNN_VALUE_FLAG_EXTERNAL_OUTPUT;
        } else {
            externalId = XNN_INVALID_VALUE_ID;
        }
        XNN_TRY(xnn_define_tensor_value(subgraph, datatype, dims.size(), dims.data(), data,
                                        externalId, flags, id));
        mOperands.insert(std::make_pair(operand, *id));
        return xnn_status_success;
    }

    xnn_status Graph::DefineXnnNode(xnn_subgraph_t subgraph, const op::Constant* constant) {
        std::unique_ptr<char> buffer(new char[constant->GetByteLength()]);
        if (buffer.get() == nullptr) {
            return xnn_status_out_of_memory;
        }
        memcpy(buffer.get(), constant->GetBuffer(), constant->GetByteLength());
        uint32_t id;
        XNN_TRY(DefineXnnTensorValue(subgraph, constant->PrimaryOutput(), &id, buffer.get()));
        mOperands.insert(std::make_pair(constant->PrimaryOutput(), id));
        mBuffers.push_back(std::move(buffer));
        return xnn_status_success;
    }

    xnn_status Graph::DefineXnnNode(xnn_subgraph_t subgraph, const op::Input* input) {
        DAWN_ASSERT(mInputs.find(input->PrimaryOutput()) != mInputs.end());
        uint32_t id;
        XNN_TRY(DefineXnnTensorValue(subgraph, input->PrimaryOutput(), &id));
        mOperands.insert(std::make_pair(input->PrimaryOutput(), id));
        return xnn_status_success;
    }

    xnn_status Graph::DefineXnnNode(xnn_subgraph_t subgraph, const op::Binary* binary) {
        DAWN_ASSERT(binary->Inputs().size() == 2);
        const OperandBase* input0Operand = binary->Inputs()[0].Get();
        DAWN_ASSERT(mOperands.find(input0Operand) != mOperands.end());
        uint32_t input0Id = mOperands.at(input0Operand);
        const OperandBase* input1Operand = binary->Inputs()[1].Get();
        DAWN_ASSERT(mOperands.find(input1Operand) != mOperands.end());
        uint32_t input1Id = mOperands.at(input1Operand);
        auto outputOperand = binary->PrimaryOutput();
        uint32_t outputId;
        XNN_TRY(DefineXnnTensorValue(subgraph, outputOperand, &outputId));
        const float outputMin = -std::numeric_limits<float>::infinity();
        const float outputMax = +std::numeric_limits<float>::infinity();
        switch (binary->GetType()) {
            case op::BinaryOpType::kAdd:
                XNN_TRY(xnn_define_add2(subgraph, outputMin, outputMax, input0Id, input1Id,
                                        outputId, 0));
                break;
            case op::BinaryOpType::kDiv:
                XNN_TRY(xnn_define_divide(subgraph, outputMin, outputMax, input0Id, input1Id,
                                          outputId, 0));
                break;
            case op::BinaryOpType::kMax:
                XNN_TRY(xnn_define_maximum2(subgraph, input0Id, input1Id, outputId, 0));
                break;
            case op::BinaryOpType::kMin:
                XNN_TRY(xnn_define_minimum2(subgraph, input0Id, input1Id, outputId, 0));
                break;
            case op::BinaryOpType::kMul:
                XNN_TRY(xnn_define_multiply2(subgraph, outputMin, outputMax, input0Id, input1Id,
                                             outputId, 0));
                break;
            case op::BinaryOpType::kSub:
                XNN_TRY(xnn_define_subtract(subgraph, outputMin, outputMax, input0Id, input1Id,
                                            outputId, 0));
                break;
            case op::BinaryOpType::kMatMul:
                if (input1Operand->Shape().size() != 2) {
                    dawn::ErrorLog() << "XNNPACK backend only support 2D operand b of matmul.";
                    return xnn_status_invalid_parameter;
                }
                XNN_TRY(xnn_define_fully_connected(subgraph, outputMin, outputMax, input0Id,
                                                   input1Id, XNN_INVALID_VALUE_ID, outputId,
                                                   XNN_FLAG_TRANSPOSE_WEIGHTS));
                break;
            default:
                dawn::ErrorLog() << "XNNPACK backend doesn't support unary op "
                                 << static_cast<int>(binary->GetType());
                return xnn_status_unsupported_parameter;
        }
        return xnn_status_success;
    }

    xnn_status Graph::DefineXnnNode(xnn_subgraph_t subgraph, const op::Clamp* clamp) {
        DAWN_ASSERT(clamp->Inputs().size() == 1);
        auto inputOperand = clamp->Inputs()[0].Get();
        DAWN_ASSERT(mOperands.find(inputOperand) != mOperands.end());
        uint32_t inputId = mOperands.at(inputOperand);
        auto outputOperand = clamp->PrimaryOutput();
        uint32_t outputId;
        XNN_TRY(DefineXnnTensorValue(subgraph, outputOperand, &outputId));
        XNN_TRY(xnn_define_clamp(subgraph, clamp->GetMinValue(), clamp->GetMaxValue(), inputId,
                                 outputId, 0));
        return xnn_status_success;
    }

    xnn_status Graph::DefineXnnNode(xnn_subgraph_t subgraph, const op::Concat* concat) {
        auto inputOperands = concat->Inputs();
        DAWN_ASSERT(inputOperands.size() >= 1);
        if (inputOperands.size() > 4) {
            dawn::ErrorLog() << "XNNPACK backend doesn't support concat inputs size "
                             << inputOperands.size();
            return xnn_status_invalid_parameter;
        }
        std::vector<uint32_t> inputIds(inputOperands.size());
        for (size_t i = 0; i < inputOperands.size(); ++i) {
            DAWN_ASSERT(mOperands.find(inputOperands[i].Get()) != mOperands.end());
            inputIds[i] = mOperands.at(inputOperands[i].Get());
        }
        auto outputOperand = concat->PrimaryOutput();
        uint32_t outputId;
        XNN_TRY(DefineXnnTensorValue(subgraph, outputOperand, &outputId));
        size_t axis = concat->GetAxis();
        switch (concat->Inputs().size()) {
            case 2:
                XNN_TRY(
                    xnn_define_concatenate2(subgraph, axis, inputIds[0], inputIds[1], outputId, 0));
                break;
            case 3:
                XNN_TRY(xnn_define_concatenate3(subgraph, axis, inputIds[0], inputIds[1],
                                                inputIds[2], outputId, 0));
                break;
            case 4:
                XNN_TRY(xnn_define_concatenate4(subgraph, axis, inputIds[0], inputIds[1],
                                                inputIds[2], inputIds[3], outputId, 0));
                break;
            default:
                dawn::ErrorLog() << "XNNPACK backend doesn't support concat inputs size "
                                 << inputOperands.size();
                return xnn_status_invalid_parameter;
        }
        return xnn_status_success;
    }

    xnn_status Graph::DefineXnnNode(xnn_subgraph_t subgraph, const op::Conv2d* conv2d) {
        auto inputOperands = conv2d->Inputs();
        DAWN_ASSERT(inputOperands.size() == 2 || inputOperands.size() == 3);
        auto inputOperand = inputOperands[0].Get();
        DAWN_ASSERT(mOperands.find(inputOperand) != mOperands.end());
        uint32_t inputId = mOperands.at(inputOperand);
        auto filterOperand = inputOperands[1].Get();
        DAWN_ASSERT(mOperands.find(filterOperand) != mOperands.end());
        uint32_t filterId = mOperands.at(filterOperand);
        uint32_t biasId = XNN_INVALID_VALUE_ID;
        if (inputOperands.size() == 3) {
            DAWN_ASSERT(mOperands.find(inputOperands[2].Get()) != mOperands.end());
            biasId = mOperands.at(inputOperands[2].Get());
        }
        auto outputOperand = conv2d->PrimaryOutput();

        const Conv2dOptions* options = conv2d->GetOptions();
        uint32_t groups = options->groups;
        uint32_t strideHeight = options->strides[0];
        uint32_t strideWidth = options->strides[1];
        uint32_t dilationHeight = options->dilations[0];
        uint32_t dilationWidth = options->dilations[1];
        size_t inputHeight, inputWidth;
        uint32_t filterHeight, filterWidth;
        size_t inputChannels, outputChannels;
        bool depthwise = false;
        if (options->inputLayout == wnn::InputOperandLayout::Nhwc) {
            inputHeight = inputOperand->Shape()[1];
            inputWidth = inputOperand->Shape()[2];
            inputChannels = inputOperand->Shape()[3];
            depthwise = (groups == inputChannels);
            if (!depthwise) {
                // For regular conv2d, xnn pack expects weights layed out like (ohwi):
                //   [groups * group_output_channels, kernel_height, kernel_width,
                //   group_input_channels]
                if (options->filterLayout != wnn::Conv2dFilterOperandLayout::Ohwi) {
                    dawn::ErrorLog()
                        << "XNNPACK backend only supports filter layout ohwi for conv2d.";
                    return xnn_status_invalid_parameter;
                }
            } else {
                // For depthwise conv2d, xnn pack expects weights layed out like (ihwo):
                //   [1, kernel_height, kernel_width, input_channels * depth_multiplier]
                if (options->filterLayout != wnn::Conv2dFilterOperandLayout::Ihwo) {
                    dawn::ErrorLog()
                        << "XNNPACK backend only supports filter layout ihwo for depthwise conv2d.";
                    return xnn_status_invalid_parameter;
                }
            }
            filterHeight = filterOperand->Shape()[1];
            filterWidth = filterOperand->Shape()[2];
            outputChannels = outputOperand->Shape()[3];
        } else {
            dawn::ErrorLog() << "XNNPACK backend only supports input layout nhwc.";
            return xnn_status_invalid_parameter;
        }
        size_t groupInputChannels = inputChannels / groups;
        size_t groupOutputChannels = outputChannels / groups;

        size_t outputHeight, outputWidth;
        uint32_t padTop, padBottom, padLeft, padRight;
        if (options->autoPad == wnn::AutoPad::Explicit) {
            // WebNN padding: [beginning_height, ending_height, beginning_width, ending_width]
            padTop = options->padding[0];
            padBottom = options->padding[1];
            padLeft = options->padding[2];
            padRight = options->padding[3];
        } else {
            outputHeight = ceil(inputHeight / strideHeight);
            outputWidth = ceil(inputWidth / strideWidth);
            size_t padAlongHeight =
                std::max(size_t(0), (outputHeight - 1) * strideHeight + filterHeight - inputHeight);
            size_t padAlongWidth =
                std::max(size_t(0), (outputWidth - 1) * strideWidth + filterWidth - inputWidth);
            if (options->autoPad == wnn::AutoPad::SameUpper) {
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
        if (options->activation) {
            switch (options->activation->GetFusionType()) {
                case FusionType::Clamp: {
                    auto clamp = reinterpret_cast<const op::FusionClamp*>(options->activation);
                    outputMin = clamp->GetMinValue();
                    outputMax = clamp->GetMaxValue();
                    break;
                }
                case FusionType::Relu:
                    outputMin = 0.0f;
                    outputMax = std::numeric_limits<float>::infinity();
                    break;
                default:
                    dawn::ErrorLog() << "XNNPACK backend doesn't support fused operator "
                                     << static_cast<int>(options->activation->GetFusionType());
                    return xnn_status_invalid_parameter;
            }
        }
        uint32_t outputId;
        XNN_TRY(DefineXnnTensorValue(subgraph, outputOperand, &outputId));
        if (depthwise) {
            XNN_TRY(xnn_define_depthwise_convolution_2d(
                subgraph, padTop, padRight, padBottom, padLeft, filterHeight, filterWidth,
                strideHeight, strideWidth, dilationHeight, dilationWidth, 1, inputChannels,
                outputMin, outputMax, inputId, filterId, biasId, outputId, 0));
        } else {
            XNN_TRY(xnn_define_convolution_2d(subgraph, padTop, padRight, padBottom, padLeft,
                                              filterHeight, filterWidth, strideHeight, strideWidth,
                                              dilationHeight, dilationWidth, groups,
                                              groupInputChannels, groupOutputChannels, outputMin,
                                              outputMax, inputId, filterId, biasId, outputId, 0));
        }
        return xnn_status_success;
    }

    xnn_status Graph::DefineXnnNode(xnn_subgraph_t subgraph, const op::Gemm* gemm) {
        auto inputs = gemm->Inputs();
        DAWN_ASSERT(inputs.size() == 2 || inputs.size() == 3);
        DAWN_ASSERT(mOperands.find(inputs[0].Get()) != mOperands.end());
        uint32_t inputId = mOperands.at(inputs[0].Get());
        DAWN_ASSERT(mOperands.find(inputs[1].Get()) != mOperands.end());
        uint32_t filterId = mOperands.at(inputs[1].Get());
        uint32_t biasId = XNN_INVALID_VALUE_ID;
        if (inputs.size() == 3) {
            DAWN_ASSERT(mOperands.find(inputs[2].Get()) != mOperands.end());
            biasId = mOperands.at(inputs[2].Get());
        }
        const GemmOptions* options = gemm->GetOptions();
        if (fabs(options->alpha - 1.0f) > std::numeric_limits<float>::epsilon()) {
            dawn::ErrorLog() << "XNNPACK backend doesn't support alpha " << options->alpha;
            return xnn_status_invalid_parameter;
        }
        if (fabs(options->beta - 1.0f) > std::numeric_limits<float>::epsilon()) {
            dawn::ErrorLog() << "XNNPACK backend doesn't support beta " << options->beta;
            return xnn_status_invalid_parameter;
        }
        if (options->aTranspose) {
            dawn::ErrorLog() << "XNNPACK backend doesn't support aTranspose.";
            return xnn_status_invalid_parameter;
        }
        uint32_t flags = 0;
        if (!options->bTranspose) {
            flags = XNN_FLAG_TRANSPOSE_WEIGHTS;
        }
        auto outputOperand = gemm->PrimaryOutput();
        uint32_t outputId;
        XNN_TRY(DefineXnnTensorValue(subgraph, outputOperand, &outputId));
        const float outputMin = -std::numeric_limits<float>::infinity();
        const float outputMax = +std::numeric_limits<float>::infinity();
        XNN_TRY(xnn_define_fully_connected(subgraph, outputMin, outputMax, inputId, filterId,
                                           biasId, outputId, flags));
        return xnn_status_success;
    }

    xnn_status Graph::DefineXnnNode(xnn_subgraph_t subgraph, const op::Pad* pad) {
        auto inputOperands = pad->Inputs();
        DAWN_ASSERT(inputOperands.size() == 2);
        auto inputOperand = inputOperands[0].Get();
        size_t inputRank = inputOperand->Shape().size();
        DAWN_ASSERT(mOperands.find(inputOperand) != mOperands.end());
        uint32_t inputId = mOperands.at(inputOperand);
        const op::Constant* paddingConstant =
            reinterpret_cast<const op::Constant*>(inputOperands[1]->Operator());
        const PadOptions* options = pad->GetOptions();
        if (options->mode != wnn::PaddingMode::Constant) {
            dawn::ErrorLog() << "XNNPACK backend doesn't support padding mode "
                             << static_cast<int>(options->mode);
            return xnn_status_invalid_parameter;
        }
        float paddingValue = options->value;
        std::vector<size_t> startPaddingVector;
        std::vector<size_t> endPaddingVector;
        const uint32_t* paddingData = static_cast<const uint32_t*>(paddingConstant->GetBuffer());
        for (size_t i = 0; i < inputRank; ++i) {
            startPaddingVector.push_back(paddingData[2 * i]);
            endPaddingVector.push_back(paddingData[2 * i + 1]);
        }
        auto outputOperand = pad->PrimaryOutput();
        uint32_t outputId;
        XNN_TRY(DefineXnnTensorValue(subgraph, outputOperand, &outputId));
        XNN_TRY(xnn_define_static_constant_pad(subgraph, startPaddingVector.data(),
                                               endPaddingVector.data(), paddingValue, inputId,
                                               outputId, 0));
        return xnn_status_success;
    }

    xnn_status Graph::DefineXnnNode(xnn_subgraph_t subgraph, const op::Pool2d* pool2d) {
        DAWN_ASSERT(pool2d->Inputs().size() == 1);
        auto inputOperand = pool2d->Inputs()[0].Get();
        DAWN_ASSERT(mOperands.find(inputOperand) != mOperands.end());
        uint32_t inputId = mOperands.at(inputOperand);
        const Pool2dOptions* options = pool2d->GetOptions();
        if (options->layout != wnn::InputOperandLayout::Nhwc) {
            dawn::ErrorLog() << "XNNPACK only supports input layout nhwc.";
            return xnn_status_invalid_parameter;
        }
        uint32_t strideHeight = options->strides[0];
        uint32_t strideWidth = options->strides[1];
        uint32_t dilationHeight = options->dilations[0];
        uint32_t dilationWidth = options->dilations[1];
        // nhwc
        size_t inputHeight = inputOperand->Shape()[1];
        size_t inputWidth = inputOperand->Shape()[2];
        uint32_t filterHeight, filterWidth;
        bool global = false;
        if (options->windowDimensions != nullptr) {
            filterHeight = options->windowDimensions[0];
            filterWidth = options->windowDimensions[1];
        } else {
            filterHeight = inputHeight;
            filterWidth = inputWidth;
            global = true;
        }

        size_t outputHeight, outputWidth;
        uint32_t padTop, padBottom, padLeft, padRight;
        if (options->autoPad == wnn::AutoPad::Explicit) {
            // WebNN padding: [beginning_height, ending_height, beginning_width, ending_width]
            padTop = options->padding[0];
            padBottom = options->padding[1];
            padLeft = options->padding[2];
            padRight = options->padding[3];
        } else {
            outputHeight = ceil(inputHeight / strideHeight);
            outputWidth = ceil(inputWidth / strideWidth);
            size_t padAlongHeight =
                std::max(size_t(0), (outputHeight - 1) * strideHeight + filterHeight - inputHeight);
            size_t padAlongWidth =
                std::max(size_t(0), (outputWidth - 1) * strideWidth + filterWidth - inputWidth);
            if (options->autoPad == wnn::AutoPad::SameUpper) {
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

        auto outputOperand = pool2d->PrimaryOutput();
        uint32_t outputId;
        XNN_TRY(DefineXnnTensorValue(subgraph, outputOperand, &outputId));
        float outputMin = -std::numeric_limits<float>::infinity();
        float outputMax = +std::numeric_limits<float>::infinity();
        const uint32_t flags = 0;
        if (pool2d->GetType() == op::Pool2dType::kAveragePool2d) {
            if (dilationHeight != 1 || dilationWidth != 1) {
                dawn::ErrorLog() << "XNNPACK does not support dilation for averagePool2d.";
                return xnn_status_invalid_parameter;
            }
            if (global) {
                XNN_TRY(xnn_define_global_average_pooling_2d(subgraph, outputMin, outputMax,
                                                             inputId, outputId, flags));
            } else {
                XNN_TRY(xnn_define_average_pooling_2d(
                    subgraph, padTop, padRight, padBottom, padLeft, filterHeight, filterWidth,
                    strideHeight, strideWidth, outputMin, outputMax, inputId, outputId, flags));
            }
        } else if (pool2d->GetType() == op::Pool2dType::kMaxPool2d) {
            XNN_TRY(xnn_define_max_pooling_2d(subgraph, padTop, padRight, padBottom, padLeft,
                                              filterHeight, filterWidth, strideHeight, strideWidth,
                                              dilationHeight, dilationWidth, outputMin, outputMax,
                                              inputId, outputId, flags));
        } else {
            dawn::ErrorLog() << "XNNPACK does not support l2Pool2d.";
            return xnn_status_invalid_parameter;
        }
        return xnn_status_success;
    }

    xnn_status Graph::DefineXnnNode(xnn_subgraph_t subgraph, const op::Reshape* reshape) {
        DAWN_ASSERT(reshape->Inputs().size() == 1);
        auto inputOperand = reshape->Inputs()[0].Get();
        DAWN_ASSERT(mOperands.find(inputOperand) != mOperands.end());
        uint32_t inputId = mOperands.at(inputOperand);
        auto outputOperand = reshape->PrimaryOutput();
        std::vector<size_t> newSizes;
        for (auto& d : outputOperand->Shape()) {
            newSizes.push_back(static_cast<size_t>(d));
        }
        if (newSizes.size() > XNN_MAX_TENSOR_DIMS) {
            dawn::ErrorLog() << "XNNPACK backend doesn't new shape rank " << newSizes.size();
            return xnn_status_invalid_parameter;
        }
        uint32_t outputId;
        XNN_TRY(DefineXnnTensorValue(subgraph, outputOperand, &outputId));
        XNN_TRY(xnn_define_static_reshape(subgraph, newSizes.size(), newSizes.data(), inputId,
                                          outputId, 0));
        return xnn_status_success;
    }

    xnn_status Graph::DefineXnnNode(xnn_subgraph_t subgraph, const op::Split* split) {
        DAWN_ASSERT(split->Inputs().size() == 1);
        auto inputOperand = split->Inputs()[0].Get();
        DAWN_ASSERT(mOperands.find(inputOperand) != mOperands.end());
        uint32_t inputId = mOperands.at(inputOperand);
        if (split->GetSplits().size() != 1) {
            dawn::ErrorLog() << "XNNPACK backend only supports even split.";
            return xnn_status_invalid_parameter;
        }
        int32_t axis = split->GetAxis();
        size_t outputSize = split->Outputs().size();
        if (outputSize > 4) {
            dawn::ErrorLog() << "XNNPACK backend doesn't support even split more than 4.";
            return xnn_status_invalid_parameter;
        }
        std::vector<uint32_t> outputIds(outputSize);
        for (size_t i = 0; i < outputSize; ++i) {
            uint32_t outputId;
            auto outputOperand = split->Outputs()[i].Get();
            XNN_TRY(DefineXnnTensorValue(subgraph, outputOperand, &outputId));
            outputIds[i] = outputId;
        }
        switch (outputSize) {
            case 2:
                XNN_TRY(
                    xnn_define_even_split2(subgraph, axis, inputId, outputIds[0], outputIds[1], 0));
                break;
            case 3:
                XNN_TRY(xnn_define_even_split3(subgraph, axis, inputId, outputIds[0], outputIds[1],
                                               outputIds[2], 0));
                break;
            case 4:
                XNN_TRY(xnn_define_even_split4(subgraph, axis, inputId, outputIds[0], outputIds[1],
                                               outputIds[2], outputIds[3], 0));
                break;
            default:
                dawn::ErrorLog() << "XNNPACK backend doesn't support even split more than 4.";
                return xnn_status_invalid_parameter;
        }
        return xnn_status_success;
    }

    xnn_status Graph::DefineXnnNode(xnn_subgraph_t subgraph, const op::Squeeze* squeeze) {
        DAWN_ASSERT(squeeze->Inputs().size() == 1);
        auto inputOperand = squeeze->Inputs()[0].Get();
        DAWN_ASSERT(mOperands.find(inputOperand) != mOperands.end());
        uint32_t inputId = mOperands.at(inputOperand);
        auto outputOperand = squeeze->PrimaryOutput();
        std::vector<size_t> newSizes;
        for (auto& d : outputOperand->Shape()) {
            newSizes.push_back(static_cast<size_t>(d));
        }
        if (newSizes.size() > XNN_MAX_TENSOR_DIMS) {
            dawn::ErrorLog() << "XNNPACK backend doesn't new size rank " << newSizes.size();
            return xnn_status_invalid_parameter;
        }
        uint32_t outputId;
        XNN_TRY(DefineXnnTensorValue(subgraph, outputOperand, &outputId));
        XNN_TRY(xnn_define_static_reshape(subgraph, newSizes.size(), newSizes.data(), inputId,
                                          outputId, 0));
        return xnn_status_success;
    }

    xnn_status Graph::DefineXnnNode(xnn_subgraph_t subgraph, const op::Unary* unary) {
        DAWN_ASSERT(unary->Inputs().size() == 1);
        auto inputOperand = unary->Inputs()[0].Get();
        DAWN_ASSERT(mOperands.find(inputOperand) != mOperands.end());
        uint32_t inputId = mOperands.at(inputOperand);
        auto outputOperand = unary->PrimaryOutput();
        uint32_t outputId;
        XNN_TRY(DefineXnnTensorValue(subgraph, outputOperand, &outputId));
        switch (unary->GetType()) {
            case op::UnaryOpType::kAbs:
                XNN_TRY(xnn_define_abs(subgraph, inputId, outputId, 0));
                break;
            case op::UnaryOpType::kCeil:
                XNN_TRY(xnn_define_ceiling(subgraph, inputId, outputId, 0));
                break;
            case op::UnaryOpType::kFloor:
                XNN_TRY(xnn_define_floor(subgraph, inputId, outputId, 0));
                break;
            case op::UnaryOpType::kHardSwish:
                XNN_TRY(xnn_define_hardswish(subgraph, inputId, outputId, 0));
                break;
            case op::UnaryOpType::kLeakyRelu:
                XNN_TRY(xnn_define_leaky_relu(
                    subgraph, reinterpret_cast<const op::LeakyRelu*>(unary)->GetAlpha(), inputId,
                    outputId, 0));
                break;
            case op::UnaryOpType::kNeg:
                XNN_TRY(xnn_define_negate(subgraph, inputId, outputId, 0));
                break;
            case op::UnaryOpType::kRelu:
                XNN_TRY(xnn_define_clamp(subgraph, 0.0f, std::numeric_limits<float>::infinity(),
                                         inputId, outputId, 0));
                break;
            case op::UnaryOpType::kSigmoid:
                XNN_TRY(xnn_define_sigmoid(subgraph, inputId, outputId, 0));
                break;
            case op::UnaryOpType::kSoftmax:
                XNN_TRY(xnn_define_softmax(subgraph, inputId, outputId, 0));
                break;
            default:
                dawn::ErrorLog() << "XNNPACK backend doesn't support unary op "
                                 << static_cast<int>(unary->GetType());
                return xnn_status_unsupported_parameter;
        }
        return xnn_status_success;
    }

#define HANDLE_OP(OpType)                                                                \
    case OperatorType::OpType: {                                                         \
        DAWN_TRY(DefineXnnNode(subgraph, reinterpret_cast<const op::OpType*>(info.op))); \
        break;                                                                           \
    }

    MaybeError Graph::Finish() {
        xnn_subgraph_t subgraph;
        if (FAILED(xnn_create_subgraph(mExternals.size(), 0, &subgraph))) {
            return DAWN_INTERNAL_ERROR("xnn_create_subgraph failed.");
        }
        for (auto const& info : mOperators) {
            switch (info.type) {
                HANDLE_OP(Binary)
                HANDLE_OP(Clamp)
                HANDLE_OP(Constant)
                HANDLE_OP(Concat)
                HANDLE_OP(Conv2d)
                HANDLE_OP(Gemm)
                HANDLE_OP(Input)
                HANDLE_OP(Pad)
                HANDLE_OP(Pool2d)
                HANDLE_OP(Reshape)
                HANDLE_OP(Split)
                HANDLE_OP(Squeeze)
                HANDLE_OP(Unary)
                default: {
                    return DAWN_UNIMPLEMENTED_ERROR("");
                }
            }
        }
        uint32_t flags = XNN_FLAG_YIELD_WORKERS;
        DAWN_TRY(xnn_create_runtime_v2(subgraph, GetThreadpool(), flags, &mRuntime));
        DAWN_TRY(xnn_delete_subgraph(subgraph));
        return {};
    }

    pthreadpool_t Graph::GetThreadpool() {
        return reinterpret_cast<Context*>(GetContext())->GetThreadpool();
    }

    MaybeError Graph::CompileImpl() {
        return {};
    }

    MaybeError Graph::ComputeImpl(NamedInputsBase* inputs, NamedOutputsBase* outputs) {
        if (mNamedInputs != inputs || mNamedOutputs != outputs) {
            bool anyPointersChanged = false;
            for (auto& input : inputs->GetRecords()) {
                if (mExternals.find(input.first) == mExternals.end()) {
                    return DAWN_VALIDATION_ERROR("Invalid input.");
                }
                void* data = static_cast<int8_t*>(input.second.resource.arrayBufferView.buffer) +
                             input.second.resource.arrayBufferView.byteOffset;
                if (mExternals[input.first].data != data) {
                    mExternals[input.first].data = data;
                    anyPointersChanged = true;
                }
            }
            mNamedInputs = inputs;

            for (auto& output : outputs->GetRecords()) {
                if (mExternals.find(output.first) == mExternals.end()) {
                    return DAWN_VALIDATION_ERROR("Invalid output.");
                }
                void* data = static_cast<int8_t*>(output.second.arrayBufferView.buffer) +
                             output.second.arrayBufferView.byteOffset;
                if (mExternals[output.first].data != data) {
                    mExternals[output.first].data = data;
                    anyPointersChanged = true;
                }
            }
            mNamedOutputs = outputs;

            if (anyPointersChanged) {
                std::vector<xnn_external_value> externalValues;
                for (auto& iterator : mExternals) {
                    externalValues.push_back(iterator.second);
                }
                DAWN_TRY(xnn_setup_runtime(mRuntime, externalValues.size(), externalValues.data()));
            }
        }

        DAWN_TRY(xnn_invoke_runtime(mRuntime));

        return {};
    }

}  // namespace webnn_native::xnnpack
