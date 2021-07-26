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

#include "webnn_native/openvino/GraphIE.h"

#include <vector>

#include "common/Assert.h"
#include "common/Log.h"
#include "webnn_native/ErrorData.h"
#include "webnn_native/NamedInputs.h"
#include "webnn_native/NamedOperands.h"
#include "webnn_native/NamedOutputs.h"
#include "webnn_native/openvino/ErrorIE.h"

#define IE(sym) sym

namespace webnn_native { namespace ie {

    namespace {
        std::string GetOutputId(const std::map<std::string, std::string>& outputNameMap,
                                const std::string& name) {
            for (auto& outputName : outputNameMap) {
                if (outputName.second == name) {
                    return outputName.first;
                }
            }

            return "";
        }

        ie_operand_descriptor ConvertTo(OperandDescriptor const* desc) {
            ie_operand_descriptor ieDesc;
            ieDesc.dimensions = desc->dimensions;
            ieDesc.dimensionsCount = desc->dimensionsCount;
            switch (desc->type) {
                case ml::OperandType::Float32:
                    ieDesc.type = ie_operand_type::Float32;
                    break;
                case ml::OperandType::Int32:
                    ieDesc.type = ie_operand_type::Int32;
                    break;
                case ml::OperandType::Float16:
                    ieDesc.type = ie_operand_type::Float16;
                    break;
                case ml::OperandType::Uint32:
                    ieDesc.type = ie_operand_type::Uint32;
                    break;
                default:
                    UNREACHABLE();
            }
            return ieDesc;
        }

        ie_conv2d_options Conv2dOptionsForIE(Conv2dOptions const* options) {
            ie_conv2d_options ieOptions;
            ieOptions.padding = options->padding;
            ieOptions.strides = options->strides;
            ieOptions.dilations = options->dilations;
            ieOptions.groups = options->groups;
            ieOptions.inputLayout = static_cast<ie_input_operand_layout>(options->inputLayout);
            ieOptions.filterLayout = static_cast<ie_filter_operand_layout>(options->filterLayout);
            ieOptions.autoPad = static_cast<ie_auto_pad>(options->autoPad);
            return ieOptions;
        }

        ie_transpose_options TransposeOptionsForIE(TransposeOptions const* options) {
            if (options == nullptr)
                return {};
            ie_transpose_options ieOptions;
            ieOptions.permutation = options->permutation;
            ieOptions.permutationCount = options->permutationCount;
            return ieOptions;
        }

        ie_gemm_options GemmOptionsForIE(GemmOptions const* options) {
            if (options == nullptr) {
                return {};
            }
            ie_gemm_options ieOptions;
            ieOptions.alpha = options->alpha;
            ieOptions.beta = options->beta;
            ieOptions.aTranspose = options->aTranspose;
            ieOptions.bTranspose = options->bTranspose;
            return ieOptions;
        }

        ie_pool2d_options Pool2dOptionsForIE(Pool2dOptions const* options) {
            ie_pool2d_options ieOptions;
            ieOptions.windowDimensions = options->windowDimensions;
            ieOptions.padding = options->padding;
            ieOptions.strides = options->strides;
            ieOptions.dilations = options->dilations;
            ieOptions.autoPad = static_cast<ie_auto_pad>(options->autoPad);
            ieOptions.layout = static_cast<ie_input_operand_layout>(options->layout);
            return ieOptions;
        }

        ie_resample_options ResampleOptionsForIE(ResampleOptions const* options) {
            if (options == nullptr) {
                return {};
            }
            ie_resample_options ieOptions;
            ieOptions.mode = static_cast<ie_interpolation_mode>(options->mode);
            ieOptions.scalesCount = options->scalesCount;
            ieOptions.scales = options->scales;
            ieOptions.sizesCount = options->sizesCount;
            ieOptions.sizes = options->sizes;
            return ieOptions;
        }
    }  // namespace

    Graph::Graph(Context* context) : GraphBase(context), mIeCompilation(nullptr) {
        // Create model.
        IEStatusCode code = IE(ie_create_model)(&mIeModel);
        if (code != IEStatusCode::OK) {
            dawn::ErrorLog() << "Failing to load ienn_c_api.dll.";
            return;
        }
    }

    Graph::~Graph() {
        if (mIeModel) {
            IE(ie_model_free)(mIeModel);
        }
        if (mIeCompilation) {
            IE(ie_compilation_free)(mIeCompilation);
        }
    }

    MaybeError Graph::AddConstant(const op::Constant* constant) {
        ie_operand_descriptor ieDesc = ConvertTo(constant->GetOperandDescriptor());
        ie_operand_t* ieOperand;
        IEStatusCode code = IE(ie_model_add_constant)(mIeModel, &ieDesc, constant->GetBuffer(),
                                                      constant->GetByteLength(), &ieOperand);
        DAWN_TRY(CheckStatusCode(code, "IE add constant"));

        mOperandIdMap[constant] = std::string(ieOperand->name);
        mConstantSet.insert(constant);
        return {};
    }

    MaybeError Graph::AddInput(const op::Input* input) {
        ie_operand_descriptor ieDesc = ConvertTo(input->GetOperandDescriptor());
        ie_operand_t* ieOperand;
        IEStatusCode code = IE(ie_model_add_input)(mIeModel, &ieDesc, &ieOperand);
        DAWN_TRY(CheckStatusCode(code, "IE add input"));
        mOperandIdMap[input] = std::string(ieOperand->name);
        mInputIdMap[input->GetName()] = std::string(ieOperand->name);
        return {};
    }

    MaybeError Graph::AddOutput(const std::string& name, const OperandBase* output) {
        ie_operand_t ieOperand;
        ieOperand.name = const_cast<char*>(mOperandIdMap[output].c_str());
        IEStatusCode code = IE(ie_model_add_output)(mIeModel, &ieOperand);
        DAWN_TRY(CheckStatusCode(code, "IE add output"));

        mOutputNameMap[ieOperand.name] = name;
        return {};
    }

    MaybeError Graph::AddBatchNorm(const op::BatchNorm* batchNorm) {
        auto inputs = batchNorm->Inputs();
        ie_operand_t input, mean, variance;
        input.name = const_cast<char*>(mOperandIdMap[inputs[0].Get()].c_str());
        mean.name = const_cast<char*>(mOperandIdMap[inputs[1].Get()].c_str());
        variance.name = const_cast<char*>(mOperandIdMap[inputs[2].Get()].c_str());
        ie_batch_norm_options_t ieOptions;
        auto options = batchNorm->GetOptions();
        if (options->scale != nullptr) {
            ieOptions.scale = {const_cast<char*>(mOperandIdMap[inputs[3].Get()].c_str())};
        }
        if (options->bias != nullptr) {
            size_t biasIndex = options->scale != nullptr ? 4 : 3;
            ieOptions.bias = {const_cast<char*>(mOperandIdMap[inputs[biasIndex].Get()].c_str())};
        }
        ieOptions.axis = options->axis;
        ieOptions.epsilon = options->epsilon;
        ie_operand_t* ieOperand;
        IEStatusCode code =
            IE(ie_model_add_batch_norm)(mIeModel, &input, &mean, &variance, &ieOptions, &ieOperand);
        DAWN_TRY(CheckStatusCode(code, "IE add batchNorm"));

        mOperandIdMap[batchNorm] = std::string(ieOperand->name);
        return {};
    }

    MaybeError Graph::AddBinary(const op::Binary* binary) {
        auto inputs = binary->Inputs();
        ie_operand_t primary;
        primary.name = const_cast<char*>(mOperandIdMap[inputs[0].Get()].c_str());
        ie_operand_t secondary;
        secondary.name = const_cast<char*>(mOperandIdMap[inputs[1].Get()].c_str());
        ie_operand_t* ieOperand = nullptr;
        IEStatusCode code = NOT_FOUND;
        if (binary->GetType() == op::BinaryOpType::kMatMul) {
            code = IE(ie_model_add_mat_mul)(mIeModel, &primary, &secondary, &ieOperand);
        } else {
            code = IE(ie_model_add_binary)(mIeModel, static_cast<ie_binary_type>(binary->GetType()),
                                           &primary, &secondary, &ieOperand);
        }
        DAWN_TRY(CheckStatusCode(code, "IE add binary"));

        mOperandIdMap[binary] = std::string(ieOperand->name);
        return {};
    }

    MaybeError Graph::AddClamp(const op::Clamp* clamp) {
        auto inputs = clamp->Inputs();
        ie_operand_t input;
        input.name = const_cast<char*>(mOperandIdMap[inputs[0].Get()].c_str());
        ie_clamp_options_t ieOptions;
        auto options = clamp->GetOptions();
        if (options->minValue != nullptr) {
            if (mConstantSet.find(inputs[1].Get()) != mConstantSet.end()) {
                op::Constant* minConstant = reinterpret_cast<op::Constant*>(inputs[1].Get());
                ieOptions.minValue = static_cast<const float*>(minConstant->GetBuffer());
                ieOptions.minDimensions = minConstant->GetOperandDescriptor()->dimensions;
                ieOptions.minDimensionsCount = minConstant->GetOperandDescriptor()->dimensionsCount;
            } else {
                return DAWN_INTERNAL_ERROR("The min of clamp options is not a constant");
            }
        }
        if (options->maxValue != nullptr) {
            size_t maxIndex = options->minValue != nullptr ? 2 : 1;
            if (mConstantSet.find(inputs[maxIndex].Get()) != mConstantSet.end()) {
                op::Constant* maxConstant = reinterpret_cast<op::Constant*>(inputs[maxIndex].Get());
                ieOptions.maxValue = static_cast<const float*>(maxConstant->GetBuffer());
                ieOptions.maxDimensions = maxConstant->GetOperandDescriptor()->dimensions;
                ieOptions.maxDimensionsCount = maxConstant->GetOperandDescriptor()->dimensionsCount;
            } else {
                return DAWN_INTERNAL_ERROR("The max of clamp options is not a constant");
            }
        }
        ie_operand_t* ieOperand;
        IEStatusCode code = IE(ie_model_add_clamp)(mIeModel, &input, &ieOptions, &ieOperand);
        DAWN_TRY(CheckStatusCode(code, "IE add clamp"));

        mOperandIdMap[clamp] = std::string(ieOperand->name);
        return {};
    }

    MaybeError Graph::AddConv2d(const op::Conv2d* conv2d) {
        auto inputs = conv2d->Inputs();
        ie_operand_t input;
        input.name = const_cast<char*>(mOperandIdMap[inputs[0].Get()].c_str());
        ie_operand_t filter;
        filter.name = const_cast<char*>(mOperandIdMap[inputs[1].Get()].c_str());
        ie_operand_t* ieOperand;
        ie_conv2d_options_t ieOptions = Conv2dOptionsForIE(conv2d->GetOptions());
        IEStatusCode code =
            IE(ie_model_add_conv2d)(mIeModel, &input, &filter, &ieOptions, &ieOperand);
        DAWN_TRY(CheckStatusCode(code, "IE add conv2d"));

        mOperandIdMap[conv2d] = std::string(ieOperand->name);
        return {};
    }

    MaybeError Graph::AddPad(const op::Pad* pad) {
        auto inputs = pad->Inputs();
        ie_operand_t input;
        input.name = const_cast<char*>(mOperandIdMap[inputs[0].Get()].c_str());
        ie_operand_t padding;
        padding.name = const_cast<char*>(mOperandIdMap[inputs[1].Get()].c_str());
        ie_pad_options_t ieOptions;
        auto options = pad->GetOptions();
        ieOptions.padValue = options->value;
        ieOptions.mode = static_cast<ie_padding_mode>(options->mode);
        if (mConstantSet.find(inputs[1].Get()) != mConstantSet.end()) {
            op::Constant* padding = reinterpret_cast<op::Constant*>(inputs[1].Get());
            int32_t const* paddingDimensions = padding->GetOperandDescriptor()->dimensions;
            uint32_t inputRank = inputs[0]->Rank();
            uint32_t padCount = padding->GetByteLength() / sizeof(int32_t);
            if (paddingDimensions[1] != 2 ||
                paddingDimensions[0] != static_cast<int32_t>(inputRank)) {
                return DAWN_INTERNAL_ERROR(
                    "The padding should has shape [n, 2], where n is the rank of the input tensor");
            }
            ieOptions.padCount = padCount;
            ieOptions.padding = static_cast<const int32_t*>(padding->GetBuffer());
        } else {
            return DAWN_INTERNAL_ERROR("The padding is not a constant");
        }
        ie_operand_t* ieOperand;
        IEStatusCode code = IE(ie_model_add_pad)(mIeModel, &input, &ieOptions, &ieOperand);
        DAWN_TRY(CheckStatusCode(code, "IE add pad"));

        mOperandIdMap[pad] = std::string(ieOperand->name);
        return {};
    }

    MaybeError Graph::AddPool2d(const op::Pool2d* pool2d) {
        auto inputs = pool2d->Inputs();
        ie_operand_t input;
        input.name = const_cast<char*>(mOperandIdMap[inputs[0].Get()].c_str());
        ie_operand_t* ieOperand;
        ie_pool2d_options_t ieOptions = Pool2dOptionsForIE(pool2d->GetOptions());
        IEStatusCode code = IE(ie_model_add_pool2d)(
            mIeModel, static_cast<ie_pool_type>(pool2d->GetType()), &input, &ieOptions, &ieOperand);
        DAWN_TRY(CheckStatusCode(code, "IE add pool2d"));

        mOperandIdMap[pool2d] = std::string(ieOperand->name);
        return {};
    }

    MaybeError Graph::AddUnary(const op::Unary* unary) {
        auto inputs = unary->Inputs();
        ie_operand_t input;
        input.name = const_cast<char*>(mOperandIdMap[inputs[0].Get()].c_str());
        ie_operand_t* ieOperand = nullptr;
        IEStatusCode code = NOT_FOUND;
        if (unary->GetType() == op::UnaryOpType::kRelu) {
            code = IE(ie_model_add_relu)(mIeModel, &input, &ieOperand);
        } else if (unary->GetType() == op::UnaryOpType::kLeakyRelu) {
            const op::LeakyRelu* leakyRelu = reinterpret_cast<const op::LeakyRelu*>(unary);
            ie_leaky_relu_options_t ieOptions = {leakyRelu->GetAlpha()};
            code = IE(ie_model_add_leaky_relu)(mIeModel, &input, &ieOptions, &ieOperand);
        } else if (unary->GetType() == op::UnaryOpType::kSoftmax) {
            code = IE(ie_model_add_softmax)(mIeModel, &input, &ieOperand);
        } else if (unary->GetType() == op::UnaryOpType::kSigmoid) {
            code = IE(ie_model_add_sigmoid)(mIeModel, &input, &ieOperand);
        } else if (unary->GetType() == op::UnaryOpType::kTanh) {
            code = IE(ie_model_add_tanh)(mIeModel, &input, &ieOperand);
        }
        DAWN_TRY(CheckStatusCode(code, "IE add unary"));

        mOperandIdMap[unary] = std::string(ieOperand->name);
        return {};
    }

    MaybeError Graph::AddReduceMean(const op::ReduceMean* reduceMean) {
        auto inputs = reduceMean->Inputs();
        ie_operand_t input;
        input.name = const_cast<char*>(mOperandIdMap[inputs[0].Get()].c_str());
        ie_operand_t* ieOperand;
        ie_reduce_mean_options_t ieOptions;
        auto options = reduceMean->GetOptions();
        ieOptions.keepDimensions = options->keepDimensions;
        ieOptions.axesCount = options->axesCount;
        ieOptions.axes = options->axes;
        IEStatusCode code = IE(ie_model_add_reduce_mean)(mIeModel, &input, &ieOptions, &ieOperand);
        DAWN_TRY(CheckStatusCode(code, "IE add reduceMean"));

        mOperandIdMap[reduceMean] = std::string(ieOperand->name);
        return {};
    }

    MaybeError Graph::AddResample(const op::Resample* resample) {
        auto inputs = resample->Inputs();
        ie_operand_t input;
        input.name = const_cast<char*>(mOperandIdMap[inputs[0].Get()].c_str());
        ie_operand_t* ieOperand;
        ie_resample_options_t ieOptions = ResampleOptionsForIE(resample->GetOptions());
        IEStatusCode code = IE(ie_model_add_resample)(mIeModel, &input, &ieOptions, &ieOperand);

        DAWN_TRY(CheckStatusCode(code, "IE add resample"));

        mOperandIdMap[resample] = std::string(ieOperand->name);
        return {};
    }

    MaybeError Graph::AddReshape(const op::Reshape* reshape) {
        auto inputs = reshape->Inputs();
        ie_operand_t input;
        input.name = const_cast<char*>(mOperandIdMap[inputs[0].Get()].c_str());
        ie_operand_t* ieOperand;
        IEStatusCode code = IE(ie_model_add_reshape)(mIeModel, &input, reshape->GetNewShape(),
                                                     reshape->GetNewShapeCount(), &ieOperand);
        DAWN_TRY(CheckStatusCode(code, "IE add reshape"));

        mOperandIdMap[reshape] = std::string(ieOperand->name);
        return {};
    }

    MaybeError Graph::AddTranspose(const op::Transpose* transpose) {
        auto inputs = transpose->Inputs();
        ie_operand_t input;
        input.name = const_cast<char*>(mOperandIdMap[inputs[0].Get()].c_str());
        ie_operand_t* ieOperand;
        ie_transpose_options_t ieOptions = TransposeOptionsForIE(transpose->GetOptions());
        IEStatusCode code = IE(ie_model_add_transpose)(mIeModel, &input, &ieOptions, &ieOperand);
        DAWN_TRY(CheckStatusCode(code, "IE add transpose"));

        mOperandIdMap[transpose] = std::string(ieOperand->name);
        return {};
    }

    MaybeError Graph::AddConcat(const op::Concat* concat) {
        auto inputs = concat->Inputs();
        std::vector<ie_operand_t> ieInputs;
        ieInputs.reserve(inputs.size());
        for (auto& input : inputs) {
            ie_operand_t ieInput;
            ieInput.name = const_cast<char*>(mOperandIdMap[input.Get()].c_str());
            ieInputs.push_back(ieInput);
        }
        ie_operand_t* ieOperand;
        IEStatusCode code = IE(ie_model_add_concat)(mIeModel, ieInputs.data(), ieInputs.size(),
                                                    concat->GetAxis(), &ieOperand);
        DAWN_TRY(CheckStatusCode(code, "IE add Concat"));

        mOperandIdMap[concat] = std::string(ieOperand->name);
        return {};
    }

    MaybeError Graph::AddGemm(const op::Gemm* gemm) {
        auto inputs = gemm->Inputs();
        std::vector<ie_operand_t> ieInputs;
        ieInputs.reserve(inputs.size());
        for (auto& input : inputs) {
            ie_operand_t ieInput;
            ieInput.name = const_cast<char*>(mOperandIdMap[input.Get()].c_str());
            ieInputs.push_back(ieInput);
        }
        ie_operand_t* ieOperand;
        ie_gemm_options_t ieOptions = GemmOptionsForIE(gemm->GetOptions());
        IEStatusCode code = IE(ie_model_add_gemm)(mIeModel, ieInputs.data(), ieInputs.size(),
                                                  &ieOptions, &ieOperand);
        DAWN_TRY(CheckStatusCode(code, "IE add gemm"));

        mOperandIdMap[gemm] = std::string(ieOperand->name);
        return {};
    }

    MaybeError Graph::Finish() {
        IEStatusCode code = IE(ie_model_finish)(mIeModel);
        DAWN_TRY(CheckStatusCode(code, "IE finish creating model"));

        return {};
    }

    MaybeError Graph::CompileImpl() {
        ml::DevicePreference devicePreference = GetContext()->GetContextOptions().devicePreference;
        const char* deviceName = devicePreference == ml::DevicePreference::Cpu ||
                                         devicePreference == ml::DevicePreference::Default
                                     ? "CPU"
                                     : "GPU";
        IEStatusCode code = IE(ie_create_compilation)(mIeModel, &mIeCompilation, deviceName);
        DAWN_TRY(CheckStatusCode(code, "IE finish compiling model"));

        return {};
    }

    MLComputeGraphStatus Graph::ComputeImpl(NamedInputsBase* inputs, NamedOutputsBase* outputs) {
        auto namedInputs = inputs->GetRecords();
        for (auto& input : mInputIdMap) {
            // All the inputs must be set.
            if (namedInputs.find(input.first) == namedInputs.end()) {
                dawn::ErrorLog() << "The input isn't set";
                return MLComputeGraphStatus_Error;
            }
            ie_operand_t ieOperand;
            ieOperand.name = const_cast<char*>(input.second.c_str());
            auto& resource = namedInputs[input.first]->resource;
            IEStatusCode code = IE(ie_compilation_set_input)(
                mIeCompilation, &ieOperand,
                static_cast<int8_t*>(resource.buffer) + resource.byteOffset, resource.byteLength);
            if (code != IEStatusCode::OK) {
                dawn::ErrorLog() << "IE Failed to set input";
                return MLComputeGraphStatus_Error;
            }
        }

        // Compute the compiled model.
        IEStatusCode code = IE(ie_compilation_compute)(mIeCompilation);
        if (code != IEStatusCode::OK) {
            dawn::ErrorLog() << "IE Failed to compute model";
            return MLComputeGraphStatus_Error;
        }

        // Get Data from nGraph with output.
        for (auto namedOutput : outputs->GetRecords()) {
            const ArrayBufferView* output = namedOutput.second;
            DAWN_ASSERT(output->buffer != nullptr && output->byteLength != 0);
            // Get output id with friendly name.
            std::string outputId = GetOutputId(mOutputNameMap, namedOutput.first);
            if (outputId.empty()) {
                dawn::ErrorLog() << "The output id is empty";
                return MLComputeGraphStatus_Error;
            }
            // pre-allocated outputs.
            ie_operand_t ieOperand;
            ieOperand.name = const_cast<char*>(outputId.c_str());
            IEStatusCode code = IE(ie_compilation_get_output)(
                mIeCompilation, &ieOperand,
                static_cast<int8_t*>(output->buffer) + output->byteOffset, output->byteLength);
            if (code != IEStatusCode::OK) {
                dawn::ErrorLog() << "IE Failed to get output buffer";
                return MLComputeGraphStatus_Error;
            }
        }

        return MLComputeGraphStatus_Success;
    }

}}  // namespace webnn_native::ie
