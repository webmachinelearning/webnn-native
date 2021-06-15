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
#include "webnn_native/NamedResults.h"
#include "webnn_native/Result.h"
#include "webnn_native/openvino/ErrorIE.h"
#include "webnn_native/openvino/ienn_symbol_table/ienn_symbol_table.h"

#define COMPUTE_ERROR_CALLBACK(code, messages)                                             \
    {                                                                                      \
        MaybeError maybeError = CheckStatusCode(code, messages);                           \
        if (maybeError.IsError()) {                                                        \
            std::unique_ptr<ErrorData> error = maybeError.AcquireError();                  \
            if (callback) {                                                                \
                callback(MLComputeGraphStatus_Error, nullptr, error->GetMessage().c_str(), \
                         userdata);                                                        \
                return MLComputeGraphStatus_Error;                                         \
            } else {                                                                       \
                dawn::ErrorLog() << error->GetMessage();                                   \
                return MLComputeGraphStatus_Error;                                         \
            }                                                                              \
        }                                                                                  \
    }                                                                                      \
    for (;;)                                                                               \
    break

namespace webnn_native { namespace ie {
    class Result : public ResultBase {
      public:
        using ResultBase::Reference;
        ~Result() {
            ie_compilation_free_buffer(&mBuffer);
        }
    };

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

        IEStatusCode SetResult(const ie_compilation_t* compilation,
                               const std::string& outputId,
                               const std::string& outputName,
                               Ref<NamedResultsBase>& results) {
            void* outputBuffer;
            size_t bufferLength;
            IEStatusCode code = IE(ie_compilation_get_buffer)(compilation, outputId.data(),
                                                              &outputBuffer, &bufferLength);
            if (code != IEStatusCode::OK) {
                return code;
            }

            ie_dimensions_t ieDimensions;
            code = IE(ie_compilation_get_dimensions)(compilation, outputId.data(), &ieDimensions);
            if (code != IEStatusCode::OK) {
                return code;
            }
            std::vector<int32_t> dimensions(ieDimensions.dims,
                                            ieDimensions.dims + ieDimensions.ranks);
            code = IE(ie_compilation_free_dimensions)(&ieDimensions);
            Ref<ResultBase> result =
                AcquireRef(new Result::ResultBase(outputBuffer, bufferLength, dimensions));
            results->Set(outputName.c_str(), result.Detach());
            return IEStatusCode::OK;
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
        IEStatusCode code = IE(ie_model_add_constant)(mIeModel, &ieDesc, constant->GetValue(),
                                                      constant->GetSize(), &ieOperand);
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
                ieOptions.minValue = static_cast<const float*>(minConstant->GetValue());
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
                ieOptions.maxValue = static_cast<const float*>(maxConstant->GetValue());
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
        }
        DAWN_TRY(CheckStatusCode(code, "IE add unary"));

        mOperandIdMap[unary] = std::string(ieOperand->name);
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

    void Graph::CompileImpl(BuildGraphCallbackDelegate delegate) {
        // TODO(junwei): We may leverage https://dawn-review.googlesource.com/c/dawn/+/36360 to
        // implement async compilation as standle-alone component.
        // Create compilation for IE backend.
        IEStatusCode code = IE(ie_create_compilation)(mIeModel, &mIeCompilation);
        delegate(code == IEStatusCode::OK ? MLBuildGraphStatus_Success : MLBuildGraphStatus_Error,
                 this);
    }

    void Graph::ComputeImpl(NamedInputsBase* inputs,
                            MLComputeGraphCallback callback,
                            void* userdata,
                            NamedOutputsBase* outputs) {
        this->GenericComputeImpl(inputs, outputs, callback, userdata);
    }

    MLBuildGraphStatus Graph::CompileSyncImpl() {
        IEStatusCode code = IE(ie_create_compilation)(mIeModel, &mIeCompilation);
        return code == IEStatusCode::OK ? MLBuildGraphStatus_Success : MLBuildGraphStatus_Error;
    }

    MLComputeGraphStatus Graph::ComputeSyncImpl(NamedInputsBase* inputs,
                                                NamedOutputsBase* outputs) {
        return this->GenericComputeImpl(inputs, outputs);
    }

    MLComputeGraphStatus Graph::GenericComputeImpl(NamedInputsBase* inputs,
                                                   NamedOutputsBase* outputs,
                                                   MLComputeGraphCallback callback,
                                                   void* userdata) {
        auto namedInputs = inputs->GetRecords();
        for (auto& input : mInputIdMap) {
            // All the inputs must be set.
            if (namedInputs.find(input.first) == namedInputs.end()) {
                COMPUTE_ERROR_CALLBACK(IEStatusCode::GENERAL_ERROR, "The input isn't set");
            }
            ie_operand_t ieOperand;
            ieOperand.name = const_cast<char*>(input.second.c_str());
            IEStatusCode code = IE(ie_compilation_set_input)(mIeCompilation, &ieOperand,
                                                             namedInputs[input.first]->buffer,
                                                             namedInputs[input.first]->size);
            COMPUTE_ERROR_CALLBACK(code, "IE set input");
        }

        // Compute the compiled model.
        IEStatusCode code = IE(ie_compilation_compute)(mIeCompilation);
        COMPUTE_ERROR_CALLBACK(code, "IE compute model");
        // Get Data from nGraph with output.
        Ref<NamedResultsBase> results = AcquireRef(new NamedResultsBase());
        if (outputs != nullptr) {
            for (auto namedOutput : outputs->GetRecords()) {
                const Output* output = namedOutput.second;
                // Get output id with friendly name.
                std::string outputId = GetOutputId(mOutputNameMap, namedOutput.first);
                if (outputId.empty()) {
                    COMPUTE_ERROR_CALLBACK(IEStatusCode::GENERAL_ERROR, "Get output id");
                }
                // pre-allocated outputs.
                if (output->buffer != nullptr && output->size != 0) {
                    ie_operand_t ieOperand;
                    ieOperand.name = const_cast<char*>(outputId.c_str());
                    IEStatusCode code = IE(ie_compilation_get_output)(mIeCompilation, &ieOperand,
                                                                      output->buffer, output->size);
                    COMPUTE_ERROR_CALLBACK(code, "IE get output");
                } else {
                    // specified outputs.
                    code = SetResult(mIeCompilation, outputId, namedOutput.first, results);
                    COMPUTE_ERROR_CALLBACK(code, "IE get result");
                }
            }
        } else {
            for (auto& outputName : mOutputNameMap) {
                code = SetResult(mIeCompilation, outputName.first, outputName.second, results);
                COMPUTE_ERROR_CALLBACK(code, "IE get result");
            }
        }
        if (callback) {
            callback(MLComputeGraphStatus_Success,
                     reinterpret_cast<MLNamedResults>(results.Detach()), nullptr, userdata);
        }
        return MLComputeGraphStatus_Success;
    }

}}  // namespace webnn_native::ie
