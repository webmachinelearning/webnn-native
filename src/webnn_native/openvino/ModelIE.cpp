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

#include "webnn_native/openvino/ModelIE.h"

#include <vector>

#include "common/Assert.h"
#include "common/Log.h"
#include "webnn_native/ErrorData.h"
#include "webnn_native/NamedOperands.h"
#include "webnn_native/openvino/CompilationIE.h"
#include "webnn_native/openvino/ErrorIE.h"
#include "webnn_native/openvino/ienn_symbol_table/ienn_symbol_table.h"

namespace webnn_native { namespace ie {

    namespace {
        ie_operand_descriptor ConvertTo(OperandDescriptor const* desc) {
            ie_operand_descriptor ieDesc;
            ieDesc.dimensions = desc->dimensions;
            ieDesc.dimensionsCount = desc->dimensionsCount;
            switch (desc->type) {
                case webnn::OperandType::Float32:
                    ieDesc.type = ie_operand_type::Float32;
                    break;
                case webnn::OperandType::Int32:
                    ieDesc.type = ie_operand_type::Int32;
                    break;
                case webnn::OperandType::Float16:
                    ieDesc.type = ie_operand_type::Float16;
                    break;
                case webnn::OperandType::Uint32:
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
            ieOptions.layout = static_cast<ie_operand_layout>(options->layout);
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

        ie_pool2d_options Pool2dOptionsForIE(Pool2dOptions const* options) {
            ie_pool2d_options ieOptions;
            ieOptions.windowDimensions = options->windowDimensions;
            ieOptions.padding = options->padding;
            ieOptions.strides = options->strides;
            ieOptions.dilations = options->dilations;
            ieOptions.layout = static_cast<ie_operand_layout>(options->layout);
            return ieOptions;
        }

    }  // namespace

    Model::Model(ModelBuilder* modelBuilder) : ModelBase(modelBuilder) {
        // Create model.
        IEStatusCode code = IE(ie_create_model)(&mIeModel);
        if (code != IEStatusCode::OK) {
            dawn::ErrorLog() << "Failing to load ienn_c_api.dll.";
            return;
        }
    }

    Model::~Model() {
        IE(ie_model_free)(mIeModel);
    }

    MaybeError Model::AddConstant(const op::Constant* constant) {
        ie_operand_descriptor ieDesc = ConvertTo(constant->GetOperandDescriptor());
        ie_operand_t* ieOperand;
        IEStatusCode code = IE(ie_model_add_constant)(mIeModel, &ieDesc, constant->GetValue(),
                                                      constant->GetSize(), &ieOperand);
        DAWN_TRY(CheckStatusCode(code, "IE add constant"));

        mOperandIdMap[constant] = std::string(ieOperand->name);
        return {};
    }

    MaybeError Model::AddInput(const op::Input* input) {
        ie_operand_descriptor ieDesc = ConvertTo(input->GetOperandDescriptor());
        ie_operand_t* ieOperand;
        IEStatusCode code = IE(ie_model_add_input)(mIeModel, &ieDesc, &ieOperand);
        DAWN_TRY(CheckStatusCode(code, "IE add input"));

        mOperandIdMap[input] = std::string(ieOperand->name);
        mInputIdMap[input->GetName()] = std::string(ieOperand->name);
        return {};
    }

    MaybeError Model::AddOutput(const std::string& name, const OperandBase* output) {
        ie_operand_t ieOperand;
        ieOperand.name = const_cast<char*>(mOperandIdMap[output].c_str());
        IEStatusCode code = IE(ie_model_add_output)(mIeModel, &ieOperand);
        DAWN_TRY(CheckStatusCode(code, "IE add output"));

        mOutputNameMap[ieOperand.name] = name;
        return {};
    }

    MaybeError Model::AddBinary(const op::Binary* binary) {
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

    MaybeError Model::AddConv2d(const op::Conv2d* conv2d) {
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

    MaybeError Model::AddPool2d(const op::Pool2d* pool2d) {
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

    MaybeError Model::AddUnary(const op::Unary* unary) {
        auto inputs = unary->Inputs();
        ie_operand_t input;
        input.name = const_cast<char*>(mOperandIdMap[inputs[0].Get()].c_str());
        ie_operand_t* ieOperand = nullptr;
        IEStatusCode code = NOT_FOUND;
        if (unary->GetType() == op::UnaryOpType::kRelu) {
            code = IE(ie_model_add_relu)(mIeModel, &input, &ieOperand);
        } else if (unary->GetType() == op::UnaryOpType::kSoftmax) {
            code = IE(ie_model_add_softmax)(mIeModel, &input, &ieOperand);
        }
        DAWN_TRY(CheckStatusCode(code, "IE add unary"));

        mOperandIdMap[unary] = std::string(ieOperand->name);
        return {};
    }

    MaybeError Model::AddReshape(const op::Reshape* reshape) {
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

    MaybeError Model::AddTranspose(const op::Transpose* transpose) {
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

    MaybeError Model::Finish() {
        IEStatusCode code = IE(ie_model_finish)(mIeModel);
        DAWN_TRY(CheckStatusCode(code, "IE finish creating model"));
        return {};
    }

    void Model::CompileImpl(WebnnCompileCallback callback,
                            void* userdata,
                            CompilationOptions const* options) {
        Compilation* compilation = new Compilation(this);
        compilation->Compile(callback, userdata, options);
    }

    ie_model_t* Model::GetInferenceEngineModel() {
        return mIeModel;
    }

    size_t Model::GetOutputsNumber() {
        size_t outputNumber = 0;
        IEStatusCode code = IE(ie_model_get_outputs_number)(mIeModel, &outputNumber);
        if (code != IEStatusCode::OK) {
            dawn::ErrorLog() << "Failing to get output number for IE.";
        }
        return outputNumber;
    }

    std::string Model::GetOutputId(size_t index) {
        char* outputName;
        IEStatusCode code = IE(ie_model_get_output_name)(mIeModel, index, &outputName);
        if (code != IEStatusCode::OK) {
            dawn::ErrorLog() << "Failing to get output name for IE.";
            return std::string();
        }
        std::string name(outputName);
        // The name has been kept in outputs object, so it can be free.
        IE(ie_model_free_name)(&outputName);

        return name;
    }

}}  // namespace webnn_native::ie
