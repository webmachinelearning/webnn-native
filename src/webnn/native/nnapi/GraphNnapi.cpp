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

#include "webnn/native/nnapi/GraphNnapi.h"

#include <errno.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "NnapiUtils.h"
#include "common/Assert.h"
#include "common/Log.h"
#include "nnapi_implementation.h"
#include "webnn/native/ErrorData.h"
#include "webnn/native/NamedInputs.h"
#include "webnn/native/NamedOperands.h"
#include "webnn/native/NamedOutputs.h"
#include "webnn/native/Utils.h"

namespace webnn::native::nnapi {

    Graph::Graph(Context* context) : GraphBase(context), mOperandCount(0) {
        mNnapiMgr = std::make_shared<NnapiManager>();
    }

    Graph::~Graph() {
    }

    MaybeError Graph::AddConstant(const op::Constant* constant) {
        auto desc = constant->GetOperandDescriptor();
        void* buffer = const_cast<void*>(constant->GetBuffer());
        auto node = CreateOperand("const", desc, buffer);
        DAWN_TRY(CheckForNullNode(node, "Failed to create Const operand"));
        mGraphNodeMap[constant->PrimaryOutput()] = node->opIndex;
        return {};
    }

    MaybeError Graph::AddInput(const op::Input* input) {
        auto desc = input->GetOperandDescriptor();
        auto node = CreateIOOperand(input->GetName(), desc, true);
        DAWN_TRY(CheckForNullNode(node, "Failed to create Input operand"));
        mGraphNodeMap[input->PrimaryOutput()] = node->opIndex;
        return {};
    }

    MaybeError Graph::AddOutput(const std::string_view name, const OperandBase* output) {
        uint32_t index = mGraphNodeMap[output];
        auto node = mIndexNodeMap[index];
        auto outputNode = CreateIOOperand(name.data(), node, false);
        DAWN_TRY(CheckForNullNode(outputNode, "Failed to create Input operand"));
        return {};
    }

    MaybeError Graph::AddInstanceNorm(const op::InstanceNorm* instanceNorm) {
        DAWN_TRY(CheckStatusCode(ANEURALNETWORKS_OP_FAILED, "nnapi Instance norm"));
        return {};
    }

    MaybeError Graph::AddBatchNorm(const op::BatchNorm* batchNorm) {
        const BatchNormOptions* options = batchNorm->GetOptions();

        // Get Inputs
        auto inputOpIndex = mGraphNodeMap[batchNorm->Inputs()[0].Get()];
        auto inputNodeInfo = mIndexNodeMap[inputOpIndex];

        auto meanOpIndex = mGraphNodeMap[batchNorm->Inputs()[1].Get()];
        auto meanNodeInfo = mIndexNodeMap[meanOpIndex];

        auto varianceOpIndex = mGraphNodeMap[batchNorm->Inputs()[2].Get()];
        auto varianceNodeInfo = mIndexNodeMap[varianceOpIndex];

        float epsilon = options->epsilon;
        std::vector<float> epsVec(1, epsilon);
        std::vector<uint32_t> _dims({1});
        auto epsilonNode = CreateOperand("epsilon", wnn::OperandType::Float32, _dims, &epsVec[0]);
        DAWN_TRY(CheckForNullNode(epsilonNode, "Failed to create NNAPI operand"));

        uint32_t scaleOpIndex = 99999;
        std::shared_ptr<NodeInfo> scaleNodeInfo;
        uint32_t biasOpIndex = 99999;
        std::shared_ptr<NodeInfo> biasNodeInfo;

        if (batchNorm->Inputs().size() == 5) {
            scaleOpIndex = mGraphNodeMap[batchNorm->Inputs()[3].Get()];
            scaleNodeInfo = mIndexNodeMap[scaleOpIndex];

            biasOpIndex = mGraphNodeMap[batchNorm->Inputs()[4].Get()];
            biasNodeInfo = mIndexNodeMap[biasOpIndex];
        } else if (batchNorm->Inputs().size() == 4) {
            if (options->bias) {
                biasOpIndex = mGraphNodeMap[batchNorm->Inputs()[3].Get()];
                biasNodeInfo = mIndexNodeMap[biasOpIndex];
            } else {
                scaleOpIndex = mGraphNodeMap[batchNorm->Inputs()[3].Get()];
                scaleNodeInfo = mIndexNodeMap[scaleOpIndex];
            }
        } else {
            dawn::ErrorLog() << "No Scale and bias";
        }

        // Define Output
        auto outputDims = batchNorm->PrimaryOutput()->Shape();
        auto outputNode = CreateOperand("", inputNodeInfo->type, outputDims, nullptr);
        DAWN_TRY(CheckForNullNode(outputNode, "Failed to create NNAPI operand"));

        // Reshape to [1,C,1,1] or [1,1,1,C]
        auto axis = options->axis;
        auto dims = inputNodeInfo->dimensions.size();
        auto noChannels = inputNodeInfo->dimensions[axis];
        memInt32Vec.emplace_back(new int(dims));
        int32_t* shapeVec = memInt32Vec.back().get();
        for (size_t i = 0; i < 4; i++) {
            shapeVec[i] = static_cast<uint32_t>(1);
        }
        shapeVec[axis] = noChannels;
        auto reshapeNodeDims = std::vector<uint32_t>({static_cast<uint32_t>(dims)});
        auto shapeNode =
            CreateOperand("reshape", wnn::OperandType::Int32, reshapeNodeDims, &shapeVec[0]);
        DAWN_TRY(CheckForNullNode(shapeNode, "Failed to create NNAPI operand"));

        std::vector<int32_t> reshapeDims(dims, 1);
        reshapeDims[axis] = noChannels;
        std::vector<uint32_t> inputListReshapeMean = {meanOpIndex, shapeNode->opIndex};
        auto meanReshapeNode = CreateOperand("", meanNodeInfo->type, reshapeDims, nullptr);
        DAWN_TRY(CheckForNullNode(meanReshapeNode, "Failed to create NNAPI operand"));
        DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_RESHAPE, inputListReshapeMean.size(),
                                         inputListReshapeMean.data(), 1,
                                         &meanReshapeNode->opIndex));
        std::vector<uint32_t> inputListReshapeVar = {varianceOpIndex, shapeNode->opIndex};
        auto varReshapeNode = CreateOperand("", varianceNodeInfo->type, reshapeDims, nullptr);
        DAWN_TRY(CheckForNullNode(varReshapeNode, "Failed to create NNAPI operand"));
        DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_RESHAPE, inputListReshapeVar.size(),
                                         inputListReshapeVar.data(), 1, &varReshapeNode->opIndex));

        // Sub x-xbar
        int32_t fuseCode = ANEURALNETWORKS_FUSED_NONE;
        uint32_t input2OpIndex = 0;
        DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &fuseCode, input2OpIndex));
        auto subNode =
            CreateOperand("", inputNodeInfo->type, batchNorm->PrimaryOutput()->Shape(), nullptr);
        DAWN_TRY(CheckForNullNode(subNode, "Failed to create NNAPI operand"));
        std::vector<uint32_t> inputListSub = {inputOpIndex, meanReshapeNode->opIndex,
                                              input2OpIndex};
        DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_SUB, inputListSub.size(),
                                         inputListSub.data(), 1, &subNode->opIndex));
        // Add var + eps
        auto addNode = CreateOperand("", varReshapeNode->type, reshapeDims, nullptr);
        DAWN_TRY(CheckForNullNode(addNode, "Failed to create NNAPI operand"));
        std::vector<uint32_t> inputListAdd = {varReshapeNode->opIndex, epsilonNode->opIndex,
                                              input2OpIndex};
        DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_ADD, inputListAdd.size(),
                                         inputListAdd.data(), 1, &addNode->opIndex));
        // sqrt of (var+eps)
        auto powNode = CreateOperand("", varReshapeNode->type, reshapeDims, nullptr);
        DAWN_TRY(CheckForNullNode(powNode, "Failed to create NNAPI operand"));
        float expValue = 0.5;
        std::vector<float> expValVec(1, expValue);
        auto expValNode = CreateOperand("expVal", wnn::OperandType::Float32, _dims, &expValVec[0]);
        DAWN_TRY(CheckForNullNode(expValNode, "Failed to create NNAPI operand"));

        std::vector<uint32_t> inputListPow = {addNode->opIndex, expValNode->opIndex};
        DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_POW, inputListPow.size(),
                                         inputListPow.data(), 1, &powNode->opIndex));
        // Div subnode/pownode
        std::vector<uint32_t> inputListDiv = {subNode->opIndex, powNode->opIndex, input2OpIndex};
        DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_DIV, inputListDiv.size(),
                                         inputListDiv.data(), 1, &outputNode->opIndex));
        uint32_t outputOpIndex = outputNode->opIndex;

        // multiply scale
        if (options->scale) {
            // Reshape
            auto scaleReshapeNode = CreateOperand("", scaleNodeInfo->type, reshapeDims, nullptr);
            DAWN_TRY(CheckForNullNode(scaleReshapeNode, "Failed to create NNAPI operand"));
            std::vector<uint32_t> inputListReshapeScale = {scaleOpIndex, shapeNode->opIndex};
            DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_RESHAPE, inputListReshapeScale.size(),
                                             inputListReshapeScale.data(), 1,
                                             &scaleReshapeNode->opIndex));

            // NodeInfo mulNode;
            auto mulNode = CreateOperand("", inputNodeInfo->type,
                                         batchNorm->PrimaryOutput()->Shape(), nullptr);
            DAWN_TRY(CheckForNullNode(mulNode, "Failed to create NNAPI operand"));
            std::vector<uint32_t> inputListMul = {scaleReshapeNode->opIndex, outputOpIndex,
                                                  input2OpIndex};
            DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_MUL, inputListMul.size(),
                                             inputListMul.data(), 1, &mulNode->opIndex));
            outputOpIndex = mulNode->opIndex;
        }
        // Add shift(bias)
        if (options->bias) {
            auto biasReshapeNode = CreateOperand("", biasNodeInfo->type, reshapeDims, nullptr);
            DAWN_TRY(CheckForNullNode(biasReshapeNode, "Failed to create NNAPI operand"));
            std::vector<uint32_t> inputListReshapeBias = {biasOpIndex, shapeNode->opIndex};
            DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_RESHAPE, inputListReshapeBias.size(),
                                             inputListReshapeBias.data(), 1,
                                             &biasReshapeNode->opIndex));

            auto addBiasNode = CreateOperand("", inputNodeInfo->type,
                                             batchNorm->PrimaryOutput()->Shape(), nullptr);
            DAWN_TRY(CheckForNullNode(addBiasNode, "Failed to create NNAPI operand"));
            std::vector<uint32_t> inputList = {outputOpIndex, biasReshapeNode->opIndex,
                                               input2OpIndex};
            DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_ADD, inputList.size(),
                                             inputList.data(), 1, &addBiasNode->opIndex));
            outputOpIndex = addBiasNode->opIndex;
        }
        if (options->activation != nullptr) {
            std::shared_ptr<NodeInfo> finalOutputNode;
            finalOutputNode = mIndexNodeMap[outputOpIndex];

            if (options->activation->GetFusionType() == FusionType::Relu) {
                auto activationNode =
                    CreateOperand("", outputNode->type, outputNode->dimensions, nullptr);
                DAWN_TRY(AddReluImpl(finalOutputNode, activationNode));
                outputOpIndex = activationNode->opIndex;
            } else if (options->activation->GetFusionType() == FusionType::Clamp) {
                auto activationNode =
                    CreateOperand("", outputNode->type, outputNode->dimensions, nullptr);
                auto clamp = reinterpret_cast<const op::FusionClamp*>(options->activation);
                DAWN_TRY(AddClampImpl(finalOutputNode, activationNode, clamp->GetMinValue(),
                                      clamp->GetMaxValue()));
                outputOpIndex = activationNode->opIndex;
            } else if (options->activation->GetFusionType() == FusionType::LeakyRelu) {
                auto activationNode =
                    CreateOperand("", outputNode->type, outputNode->dimensions, nullptr);
                auto leakyRelu = reinterpret_cast<const op::FusionLeakyRelu*>(options->activation);
                DAWN_TRY(AddLeakyReluImpl(finalOutputNode, activationNode, leakyRelu->GetAlpha()));
                outputOpIndex = activationNode->opIndex;
            } else if (options->activation->GetFusionType() == FusionType::Sigmoid) {
                auto activationNode =
                    CreateOperand("", outputNode->type, outputNode->dimensions, nullptr);
                DAWN_TRY(AddSigmoidImpl(finalOutputNode, activationNode));
                outputOpIndex = activationNode->opIndex;
            } else {
                dawn::ErrorLog() << "Unsupported activation";
            }
        }
        mGraphNodeMap[batchNorm->PrimaryOutput()] = outputOpIndex;

        return {};
    }

    MaybeError Graph::AddExpandDimsImpl(const std::shared_ptr<NodeInfo>& node,
                                        int32_t dim_index,
                                        uint32_t& index) {
        uint32_t dimSize = node->dimensions.size() + 1, scalarOpIndex;
        std::vector<uint32_t> dims(dimSize);
        auto outNode = CreateOperand("", node->type, dims);
        DAWN_TRY(CheckForNullNode(outNode, "Failed to create NNAPI operand"));
        DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &dim_index, scalarOpIndex));
        uint32_t inputList[2] = {node->opIndex, scalarOpIndex};
        DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_EXPAND_DIMS, 2, inputList, 1,
                                         &outNode->opIndex));
        index = outNode->opIndex;
        return {};
    }

    MaybeError Graph::AddMatMulImpl(const std::shared_ptr<NodeInfo>& input0NodeInfo,
                                    const std::shared_ptr<NodeInfo>& input1NodeInfo,
                                    std::vector<int32_t> outputDims,
                                    uint32_t& outputIndex) {
        int32_t fuseCode = ANEURALNETWORKS_FUSED_NONE;
        uint32_t input0OpIndex = input0NodeInfo->opIndex;
        uint32_t input1OpIndex = input1NodeInfo->opIndex, fcActivationIndex = 0;
        int32_t input0Rank = input0NodeInfo->dimensions.size();
        int32_t input1Rank = input1NodeInfo->dimensions.size();

        if ((input1Rank <= 0) || (input1Rank > 2)) {
            return DAWN_VALIDATION_ERROR("Second Operand is supported only upto rank 2");
        }

        DAWN_TRY(
            mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &fuseCode, fcActivationIndex));
        if (input0Rank == 1)
            DAWN_TRY(AddExpandDimsImpl(input0NodeInfo, 0, input0OpIndex));

        uint32_t biasLen = 1;
        if (input1Rank == 1) {
            DAWN_TRY(AddExpandDimsImpl(input1NodeInfo, 0, input1OpIndex));
        } else {
            biasLen = input1NodeInfo->dimensions[1];
            memInt32Vec.emplace_back(new int(2));
            int32_t* permute = memInt32Vec.back().get();
            permute[0] = 1;
            permute[1] = 0;
            DAWN_TRY(AddTransposeImpl(input1NodeInfo, permute, 2, input1OpIndex));
        }

        std::vector<float> biasMem(biasLen, 0);
        std::vector<int32_t> fcDims(2);
        fcDims[0] = 1;
        for (size_t i = 0; i < outputDims.size(); i++) {
            if (i == (outputDims.size() - 1))
                fcDims[1] = static_cast<uint32_t>(outputDims[i]);
            else
                fcDims[0] = fcDims[0] * outputDims[i];
        }

        auto fcOutputNode = CreateOperand("", input0NodeInfo->type, fcDims);
        DAWN_TRY(CheckForNullNode(fcOutputNode, "Failed to create NNAPI operand"));
        auto biasDimensions = std::vector<uint32_t>({static_cast<uint32_t>(biasLen)});
        auto biasNode = CreateOperand("bias", input0NodeInfo->type, biasDimensions, &biasMem[0]);
        DAWN_TRY(CheckForNullNode(biasNode, "Failed to create NNAPI operand"));
        std::vector<uint32_t> inputList = {input0OpIndex, input1OpIndex, biasNode->opIndex,
                                           fcActivationIndex};
        DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_FULLY_CONNECTED, inputList.size(),
                                         inputList.data(), 1, &fcOutputNode->opIndex));
        outputIndex = fcOutputNode->opIndex;
        if (input0Rank > 2) {
            memInt32Vec.emplace_back(new int(outputDims.size()));
            int32_t* shapeVec = memInt32Vec.back().get();
            for (size_t i = 0; i < outputDims.size(); i++) {
                shapeVec[i] = static_cast<uint32_t>(outputDims[i]);
            }
            auto reshapeNodeDims =
                std::vector<uint32_t>({static_cast<uint32_t>(outputDims.size())});
            auto reshapeNode =
                CreateOperand("reshape", wnn::OperandType::Int32, reshapeNodeDims, &shapeVec[0]);
            DAWN_TRY(CheckForNullNode(reshapeNode, "Failed to create NNAPI operand"));
            auto outputNode = CreateOperand("", input0NodeInfo->type, outputDims, nullptr);
            DAWN_TRY(CheckForNullNode(outputNode, "Failed to create NNAPI operand"));
            uint32_t reshapeNodeInputs[2] = {fcOutputNode->opIndex, reshapeNode->opIndex};
            DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_RESHAPE, 2, reshapeNodeInputs, 1,
                                             &outputNode->opIndex));
            outputIndex = outputNode->opIndex;
        }
        return {};
    }

    MaybeError Graph::AddBinary(const op::Binary* binary) {
        auto input0OpIndex = mGraphNodeMap[binary->Inputs()[0].Get()];
        auto input0NodeInfo = mIndexNodeMap[input0OpIndex];
        auto input1OpIndex = mGraphNodeMap[binary->Inputs()[1].Get()];
        auto input1NodeInfo = mIndexNodeMap[input1OpIndex];
        // output
        auto outputDims = binary->PrimaryOutput()->Shape();

        if (binary->GetType() == op::BinaryOpType::kAdd) {
            int32_t fuseCode = ANEURALNETWORKS_FUSED_NONE;
            uint32_t input2OpIndex = 0;
            auto outputNode = CreateOperand("", input0NodeInfo->type, outputDims, nullptr);
            DAWN_TRY(CheckForNullNode(outputNode, "Failed to create NNAPI operand"));
            mGraphNodeMap[binary->PrimaryOutput()] = outputNode->opIndex;
            DAWN_TRY(
                mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &fuseCode, input2OpIndex));
            std::vector<uint32_t> inputList = {input0OpIndex, input1OpIndex, input2OpIndex};
            DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_ADD, inputList.size(),
                                             inputList.data(), 1, &outputNode->opIndex));
        } else if (binary->GetType() == op::BinaryOpType::kSub) {
            int32_t fuseCode = ANEURALNETWORKS_FUSED_NONE;
            uint32_t input2OpIndex = 0;
            auto outputNode = CreateOperand("", input0NodeInfo->type, outputDims, nullptr);
            DAWN_TRY(CheckForNullNode(outputNode, "Failed to create NNAPI operand"));
            mGraphNodeMap[binary->PrimaryOutput()] = outputNode->opIndex;
            DAWN_TRY(
                mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &fuseCode, input2OpIndex));
            std::vector<uint32_t> inputList = {input0OpIndex, input1OpIndex, input2OpIndex};
            DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_SUB, inputList.size(),
                                             inputList.data(), 1, &outputNode->opIndex));
        } else if (binary->GetType() == op::BinaryOpType::kMul) {
            int32_t fuseCode = ANEURALNETWORKS_FUSED_NONE;
            uint32_t input2OpIndex = 0;
            auto outputNode = CreateOperand("", input0NodeInfo->type, outputDims, nullptr);
            DAWN_TRY(CheckForNullNode(outputNode, "Failed to create NNAPI operand"));
            mGraphNodeMap[binary->PrimaryOutput()] = outputNode->opIndex;
            DAWN_TRY(
                mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &fuseCode, input2OpIndex));
            std::vector<uint32_t> inputList = {input0OpIndex, input1OpIndex, input2OpIndex};
            DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_MUL, inputList.size(),
                                             inputList.data(), 1, &outputNode->opIndex));
        } else if (binary->GetType() == op::BinaryOpType::kDiv) {
            int32_t fuseCode = ANEURALNETWORKS_FUSED_NONE;
            uint32_t input2OpIndex = 0;
            auto outputNode = CreateOperand("", input0NodeInfo->type, outputDims, nullptr);
            DAWN_TRY(CheckForNullNode(outputNode, "Failed to create NNAPI operand"));
            mGraphNodeMap[binary->PrimaryOutput()] = outputNode->opIndex;
            DAWN_TRY(
                mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &fuseCode, input2OpIndex));
            std::vector<uint32_t> inputList = {input0OpIndex, input1OpIndex, input2OpIndex};
            DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_DIV, inputList.size(),
                                             inputList.data(), 1, &outputNode->opIndex));
        } else if (binary->GetType() == op::BinaryOpType::kMax) {
            auto outputNode = CreateOperand("", input0NodeInfo->type, outputDims, nullptr);
            DAWN_TRY(CheckForNullNode(outputNode, "Failed to create NNAPI operand"));
            mGraphNodeMap[binary->PrimaryOutput()] = outputNode->opIndex;
            std::vector<uint32_t> inputList = {input0OpIndex, input1OpIndex};
            DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_MAXIMUM, inputList.size(),
                                             inputList.data(), 1, &outputNode->opIndex));
        } else if (binary->GetType() == op::BinaryOpType::kMin) {
            auto outputNode = CreateOperand("", input0NodeInfo->type, outputDims, nullptr);
            DAWN_TRY(CheckForNullNode(outputNode, "Failed to create NNAPI operand"));
            mGraphNodeMap[binary->PrimaryOutput()] = outputNode->opIndex;
            std::vector<uint32_t> inputList = {input0OpIndex, input1OpIndex};
            DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_MINIMUM, inputList.size(),
                                             inputList.data(), 1, &outputNode->opIndex));
        } else if (binary->GetType() == op::BinaryOpType::kPower) {
            auto outputNode = CreateOperand("", input0NodeInfo->type, outputDims, nullptr);
            DAWN_TRY(CheckForNullNode(outputNode, "Failed to create NNAPI operand"));
            mGraphNodeMap[binary->PrimaryOutput()] = outputNode->opIndex;
            std::vector<uint32_t> inputList = {input0OpIndex, input1OpIndex};
            DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_POW, inputList.size(),
                                             inputList.data(), 1, &outputNode->opIndex));
        } else if (binary->GetType() == op::BinaryOpType::kMatMul) {
            uint32_t outIndex;
            DAWN_TRY(AddMatMulImpl(input0NodeInfo, input1NodeInfo, outputDims, outIndex));
            mGraphNodeMap[binary->PrimaryOutput()] = outIndex;
        } else {
            DAWN_TRY(CheckStatusCode(ANEURALNETWORKS_OP_FAILED, "nnapi AddBinary"));
        }
        return {};
    }

    MaybeError Graph::AddClampImpl(const std::shared_ptr<NodeInfo>& inputNode,
                                   std::shared_ptr<NodeInfo> outputNode,
                                   float min,
                                   float max) {
        std::vector<float> minVec(inputNode->getDimsSize(), min);
        std::vector<float> maxVec(inputNode->getDimsSize(), max);
        auto outputNode0 = CreateOperand("", inputNode->type, inputNode->dimensions, nullptr);
        DAWN_TRY(CheckForNullNode(outputNode0, "Failed to create NNAPI operand"));
        auto minNode = CreateOperand("min", inputNode->type, inputNode->dimensions, &minVec[0]);
        DAWN_TRY(CheckForNullNode(minNode, "Failed to create NNAPI operand"));
        auto maxNode = CreateOperand("max", inputNode->type, inputNode->dimensions, &maxVec[0]);
        DAWN_TRY(CheckForNullNode(maxNode, "Failed to create NNAPI operand"));

        std::vector<uint32_t> inputList = {inputNode->opIndex, minNode->opIndex};
        DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_MAXIMUM, inputList.size(),
                                         inputList.data(), 1, &outputNode0->opIndex));
        inputList = {outputNode0->opIndex, maxNode->opIndex};
        DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_MINIMUM, inputList.size(),
                                         inputList.data(), 1, &outputNode->opIndex));
        return {};
    }

    MaybeError Graph::AddLeakyReluImpl(const std::shared_ptr<NodeInfo>& inputNode,
                                       std::shared_ptr<NodeInfo> outputNode,
                                       float alpha) {
        std::vector<float> alphaVec(1, alpha);
        std::vector<uint32_t> dims({1});
        auto alphaNode = CreateOperand("alpha", inputNode->type, dims, &alphaVec[0]);
        DAWN_TRY(CheckForNullNode(alphaNode, "Failed to create NNAPI operand"));
        std::vector<uint32_t> inputList = {inputNode->opIndex, alphaNode->opIndex};
        DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_PRELU, inputList.size(), inputList.data(),
                                         1, &outputNode->opIndex));
        return {};
    }

    MaybeError Graph::AddSigmoidImpl(const std::shared_ptr<NodeInfo>& inputNode,
                                     std::shared_ptr<NodeInfo> outputNode) {
        uint32_t inputList = inputNode->opIndex;
        DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_LOGISTIC, 1, &inputList, 1,
                                         &outputNode->opIndex));
        return {};
    }

    MaybeError Graph::AddReluImpl(const std::shared_ptr<NodeInfo>& inputNode,
                                  std::shared_ptr<NodeInfo> outputNode) {
        uint32_t inputList = inputNode->opIndex;
        DAWN_TRY(
            mNnapiMgr->AddOperation(ANEURALNETWORKS_RELU, 1, &inputList, 1, &outputNode->opIndex));
        return {};
    }

    MaybeError Graph::AddClamp(const op::Clamp* clamp) {
        auto inputOpIndex = mGraphNodeMap[clamp->Inputs()[0].Get()];
        auto inputNodeInfo = mIndexNodeMap[inputOpIndex];
        auto outputNode =
            CreateOperand("", inputNodeInfo->type, inputNodeInfo->dimensions, nullptr);
        DAWN_TRY(CheckForNullNode(outputNode, "Failed to create NNAPI operand"));
        mGraphNodeMap[clamp->PrimaryOutput()] = outputNode->opIndex;
        return AddClampImpl(inputNodeInfo, outputNode, clamp->GetMinValue(), clamp->GetMaxValue());
    }

    MaybeError Graph::AddSlice(const op::Slice* slice) {
        DAWN_TRY(CheckStatusCode(ANEURALNETWORKS_OP_FAILED, "nnapi AddSlice"));
        return {};
    }

    void getPermuteArray(wnn::Conv2dFilterOperandLayout srcLayout,
                         wnn::Conv2dFilterOperandLayout dstLayout,
                         int* perm) {
        std::map<char, int32_t> OhwiLayout = {
            {'o', 0},
            {'h', 1},
            {'w', 2},
            {'i', 3},
        };
        std::map<char, int32_t> HwioLayout = {
            {'o', 3},
            {'h', 0},
            {'w', 1},
            {'i', 2},
        };
        std::map<char, int32_t> IhwoLayout = {
            {'o', 3},
            {'h', 1},
            {'w', 2},
            {'i', 0},
        };
        std::map<char, int32_t> OihwLayout = {
            {'o', 0},
            {'h', 2},
            {'w', 3},
            {'i', 1},
        };

        auto getSrcLayoutIndex = [&](char c) {
            switch (srcLayout) {
                case wnn::Conv2dFilterOperandLayout::Oihw:
                    return OihwLayout[c];
                case wnn::Conv2dFilterOperandLayout::Hwio:
                    return HwioLayout[c];
                case wnn::Conv2dFilterOperandLayout::Ihwo:
                    return IhwoLayout[c];
                case wnn::Conv2dFilterOperandLayout::Ohwi:
                default:
                    return OhwiLayout[c];
            }
        };

        switch (dstLayout) {
            case wnn::Conv2dFilterOperandLayout::Oihw:
                perm[0] = getSrcLayoutIndex('o');
                perm[1] = getSrcLayoutIndex('i');
                perm[2] = getSrcLayoutIndex('h');
                perm[3] = getSrcLayoutIndex('w');
                break;
            case wnn::Conv2dFilterOperandLayout::Hwio:
                perm[0] = getSrcLayoutIndex('h');
                perm[1] = getSrcLayoutIndex('w');
                perm[2] = getSrcLayoutIndex('i');
                perm[3] = getSrcLayoutIndex('o');
                break;
            case wnn::Conv2dFilterOperandLayout::Ihwo:
                perm[0] = getSrcLayoutIndex('i');
                perm[1] = getSrcLayoutIndex('h');
                perm[2] = getSrcLayoutIndex('w');
                perm[3] = getSrcLayoutIndex('o');
                break;
            case wnn::Conv2dFilterOperandLayout::Ohwi:
                perm[0] = getSrcLayoutIndex('o');
                perm[1] = getSrcLayoutIndex('h');
                perm[2] = getSrcLayoutIndex('w');
                perm[3] = getSrcLayoutIndex('i');
                break;
            default:
                break;
        }
    }

    MaybeError Graph::AddPool2d(const op::Pool2d* pool2d) {
        const Pool2dOptions* options = pool2d->GetOptions();

        // input
        auto inputShape = pool2d->Inputs()[0]->Shape();
        bool nchw = options->layout == wnn::InputOperandLayout::Nchw;
        int32_t inputHeight = nchw ? inputShape[2] : inputShape[1];
        int32_t inputWidth = nchw ? inputShape[3] : inputShape[2];
        auto inputOpIndex = mGraphNodeMap[pool2d->Inputs()[0].Get()];
        auto inputNodeInfo = mIndexNodeMap[inputOpIndex];

        // output
        auto outputDims = pool2d->PrimaryOutput()->Shape();
        auto outputNode = CreateOperand("", inputNodeInfo->type, outputDims, nullptr);
        DAWN_TRY(CheckForNullNode(outputNode, "Failed to create NNAPI operand"));
        mGraphNodeMap[pool2d->PrimaryOutput()] = outputNode->opIndex;

        int32_t paddingLeft = options->padding ? options->padding[2] : 0;
        int32_t paddingRight = options->padding ? options->padding[3] : 0;
        int32_t paddingTop = options->padding ? options->padding[0] : 0;
        int32_t paddingBottom = options->padding ? options->padding[1] : 0;
        int32_t strideWidth = options->strides ? options->strides[1] : 0;
        int32_t strideHeight = options->strides ? options->strides[0] : 0;
        int32_t filterWidth =
            options->windowDimensions == nullptr ? inputWidth : options->windowDimensions[1];
        int32_t filterHeight =
            options->windowDimensions == nullptr ? inputHeight : options->windowDimensions[0];
        int32_t dilationWidth = options->dilations ? options->dilations[1] : 0;
        int32_t dilationHeight = options->dilations ? options->dilations[0] : 0;
        int8_t layout = (options->layout == wnn::InputOperandLayout::Nhwc) ? 0 : 1;
        uint32_t fuseOperation = 0;

        if (dilationWidth > 1 && dilationHeight > 1) {
            dawn::ErrorLog() << "AddPool2D: No Dilation support ";
            return DAWN_VALIDATION_ERROR("Dilation is not yet supported");
        }

        uint32_t paddingLeftWOp, paddingRightWOp, paddingTopHOp, paddingBottomHOp, strideWidthOp,
            strideHOp;
        uint32_t fuseOp, layoutOp, filterWOp, filterHOp;
        if (options->autoPad == wnn::AutoPad::Explicit) {
            DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &paddingLeft,
                                                    paddingLeftWOp));
            DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &paddingRight,
                                                    paddingRightWOp));
            DAWN_TRY(
                mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &paddingTop, paddingTopHOp));
            DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &paddingBottom,
                                                    paddingBottomHOp));
            DAWN_TRY(
                mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &strideWidth, strideWidthOp));
            DAWN_TRY(
                mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &strideHeight, strideHOp));
            DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &fuseOperation, fuseOp));
            DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_BOOL, &layout, layoutOp));
            DAWN_TRY(
                mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &filterWidth, filterWOp));
            DAWN_TRY(
                mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &filterHeight, filterHOp));

            std::vector<uint32_t> inputList = {inputOpIndex,  paddingLeftWOp,   paddingRightWOp,
                                               paddingTopHOp, paddingBottomHOp, strideWidthOp,
                                               strideHOp,     filterWOp,        filterHOp,
                                               fuseOp,        layoutOp};

            switch (pool2d->GetType()) {
                case op::Pool2dType::kAveragePool2d:
                    DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_AVERAGE_POOL_2D,
                                                     inputList.size(), inputList.data(), 1,
                                                     &outputNode->opIndex));
                    break;
                case op::Pool2dType::kL2Pool2d:
                    DAWN_TRY(CheckStatusCode(ANEURALNETWORKS_OP_FAILED,
                                             "nnapi L2Pool2d unsupported operation"));
                    break;
                case op::Pool2dType::kMaxPool2d:
                    DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_MAX_POOL_2D, inputList.size(),
                                                     inputList.data(), 1, &outputNode->opIndex));
                    break;
                default:
                    DAWN_TRY(CheckStatusCode(ANEURALNETWORKS_OP_FAILED,
                                             "nnapi Pool2D unsupported operation"));
                    break;
            }

        } else {
            int32_t height = (options->layout == wnn::InputOperandLayout::Nchw)
                                 ? inputNodeInfo->dimensions[2]
                                 : inputNodeInfo->dimensions[1];
            int32_t width = (options->layout == wnn::InputOperandLayout::Nchw)
                                ? inputNodeInfo->dimensions[3]
                                : inputNodeInfo->dimensions[2];

            utils::ComputeImplicitPaddingForAutoPad<int32_t>(
                options->autoPad, options->dilations[0], height, filterHeight, options->strides[0],
                paddingTop, paddingBottom);
            utils::ComputeImplicitPaddingForAutoPad<int32_t>(
                options->autoPad, options->dilations[1], width, filterWidth, options->strides[1],
                paddingLeft, paddingRight);

            DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &paddingLeft,
                                                    paddingLeftWOp));
            DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &paddingRight,
                                                    paddingRightWOp));
            DAWN_TRY(
                mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &paddingTop, paddingTopHOp));
            DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &paddingBottom,
                                                    paddingBottomHOp));
            DAWN_TRY(
                mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &strideWidth, strideWidthOp));
            DAWN_TRY(
                mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &strideHeight, strideHOp));
            DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &fuseOperation, fuseOp));
            DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_BOOL, &layout, layoutOp));
            DAWN_TRY(
                mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &filterWidth, filterWOp));
            DAWN_TRY(
                mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &filterHeight, filterHOp));

            std::vector<uint32_t> inputList = {inputOpIndex,  paddingLeftWOp,   paddingRightWOp,
                                               paddingTopHOp, paddingBottomHOp, strideWidthOp,
                                               strideHOp,     filterWOp,        filterHOp,
                                               fuseOp,        layoutOp};

            switch (pool2d->GetType()) {
                case op::Pool2dType::kAveragePool2d:
                    DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_AVERAGE_POOL_2D,
                                                     inputList.size(), inputList.data(), 1,
                                                     &outputNode->opIndex));
                    break;
                case op::Pool2dType::kL2Pool2d:
                    DAWN_TRY(CheckStatusCode(ANEURALNETWORKS_OP_FAILED,
                                             "nnapi L2Pool2d unsupported operation"));
                    break;
                case op::Pool2dType::kMaxPool2d:
                    DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_MAX_POOL_2D, inputList.size(),
                                                     inputList.data(), 1, &outputNode->opIndex));
                    break;
                default:
                    DAWN_TRY(CheckStatusCode(ANEURALNETWORKS_OP_FAILED,
                                             "nnapi Pool2D unsupported operation"));
                    break;
            }
        }
        return {};
    }

    MaybeError Graph::AddTransposeImpl(const std::shared_ptr<NodeInfo>& node,
                                       int32_t* permute,
                                       uint32_t permuteSize,
                                       uint32_t& outputIndex) {
        if (!permute)
            DAWN_ASSERT(permute != nullptr);

        auto permNode = CreateOperand("", wnn::OperandType::Int32,
                                      std::vector<uint32_t>({permuteSize}), nullptr);
        DAWN_TRY(CheckForNullNode(permNode, "Failed to create NNAPI operand"));
        DAWN_TRY(
            mNnapiMgr->SetVecOperand(permNode->opIndex, permute, sizeof(int32_t) * permuteSize));

        std::vector<uint32_t> outDims(node->dimensions.size());
        for (size_t i = 0; i < node->dimensions.size(); i++) {
            outDims[i] = node->dimensions[permute[i]];
        }

        auto outputNode = CreateOperand("", node->type, outDims, nullptr);
        DAWN_TRY(CheckForNullNode(outputNode, "Failed to create NNAPI operand"));
        uint32_t inputList[2] = {node->opIndex, permNode->opIndex};
        DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_TRANSPOSE, 2, inputList, 1,
                                         &outputNode->opIndex));
        outputIndex = outputNode->opIndex;
        return {};
    }

    MaybeError Graph::AddConv2d(const op::Conv2d* conv2d) {
        auto options = conv2d->GetOptions();

        auto getOutputChannels = [&](std::vector<uint32_t>& filterDims) {
            switch (options->filterLayout) {
                case wnn::Conv2dFilterOperandLayout::Hwio:
                case wnn::Conv2dFilterOperandLayout::Ihwo:
                    return filterDims[3];
                case wnn::Conv2dFilterOperandLayout::Oihw:
                case wnn::Conv2dFilterOperandLayout::Ohwi:
                default:
                    return filterDims[0];
            }
        };

        auto getFilterHeight = [&](std::vector<uint32_t>& filterDims) {
            switch (options->filterLayout) {
                case wnn::Conv2dFilterOperandLayout::Hwio:
                    return filterDims[0];
                case wnn::Conv2dFilterOperandLayout::Ihwo:
                case wnn::Conv2dFilterOperandLayout::Ohwi:
                    return filterDims[1];
                case wnn::Conv2dFilterOperandLayout::Oihw:
                default:
                    return filterDims[2];
            }
        };

        auto getFilterWidth = [&](std::vector<uint32_t>& filterDims) {
            switch (options->filterLayout) {
                case wnn::Conv2dFilterOperandLayout::Hwio:
                    return filterDims[1];
                case wnn::Conv2dFilterOperandLayout::Ihwo:
                case wnn::Conv2dFilterOperandLayout::Ohwi:
                    return filterDims[2];
                case wnn::Conv2dFilterOperandLayout::Oihw:
                default:
                    return filterDims[3];
            }
        };

        auto getFilterInChannels = [&](std::vector<uint32_t>& filterDims) {
            switch (options->filterLayout) {
                case wnn::Conv2dFilterOperandLayout::Hwio:
                    return filterDims[2];
                case wnn::Conv2dFilterOperandLayout::Ihwo:
                    return filterDims[0];
                case wnn::Conv2dFilterOperandLayout::Oihw:
                    return filterDims[1];
                case wnn::Conv2dFilterOperandLayout::Ohwi:
                default:
                    return filterDims[3];
            }
        };

        auto inputOpIndex = mGraphNodeMap[conv2d->Inputs()[0].Get()];
        auto inputNodeInfo = mIndexNodeMap[inputOpIndex];
        auto outputDims = conv2d->PrimaryOutput()->Shape();
        auto outputNode = CreateOperand("", inputNodeInfo->type, outputDims, nullptr);
        DAWN_TRY(CheckForNullNode(outputNode, "Failed to create NNAPI operand"));
        auto filterOpIndex = mGraphNodeMap[conv2d->Inputs()[1].Get()];
        auto filterNodeInfo = mIndexNodeMap[filterOpIndex];
        uint32_t biasOpIndex = 0;
        if (options->bias == nullptr) {
            std::vector<float> biasMem(getOutputChannels(filterNodeInfo->dimensions), 0);
            auto biasNode = CreateOperand("bias", inputNodeInfo->type,
                                          std::vector<uint32_t>({static_cast<uint32_t>(
                                              getOutputChannels(filterNodeInfo->dimensions))}),
                                          &biasMem[0]);
            DAWN_TRY(CheckForNullNode(biasNode, "Failed to create NNAPI operand"));
            biasOpIndex = biasNode->opIndex;
        } else {
            biasOpIndex = mGraphNodeMap[conv2d->Inputs()[2].Get()];
        }

        bool isDepthwiseConv2d = false, isGroupConvolution = false;
        {
            if (options->groups > 1) {
                int32_t inputChannels = 0;
                if (options->inputLayout == wnn::InputOperandLayout::Nchw)
                    inputChannels = inputNodeInfo->dimensions[1];
                else if (options->inputLayout == wnn::InputOperandLayout::Nhwc)
                    inputChannels = inputNodeInfo->dimensions[3];

                if (options->groups == inputChannels) {
                    int32_t filterChannels = 0;
                    switch (options->filterLayout) {
                        case wnn::Conv2dFilterOperandLayout::Oihw:
                        case wnn::Conv2dFilterOperandLayout::Ohwi:
                            filterChannels = static_cast<int32_t>(filterNodeInfo->dimensions[0]);
                            break;
                        case wnn::Conv2dFilterOperandLayout::Hwio:
                        case wnn::Conv2dFilterOperandLayout::Ihwo:
                            filterChannels = static_cast<int32_t>(filterNodeInfo->dimensions[3]);
                            break;
                        default:
                            break;
                    }

                    if (filterChannels == options->groups) {
                        if (getFilterInChannels(filterNodeInfo->dimensions) == 1) {
                            isDepthwiseConv2d = true;
                        } else {
                            isGroupConvolution = true;
                        }
                    }
                }
            }
        }

        int32_t paddingLeft = options->padding ? options->padding[2] : 0;
        int32_t paddingRight = options->padding ? options->padding[3] : 0;
        int32_t paddingTop = options->padding ? options->padding[0] : 0;
        int32_t paddingBottom = options->padding ? options->padding[1] : 0;
        int32_t strideWidth = options->strides ? options->strides[1] : 0;
        int32_t strideHeight = options->strides ? options->strides[0] : 0;
        int32_t dilationsWidth = options->dilations ? options->dilations[1] : 0;
        int32_t dilationsHeight = options->dilations ? options->dilations[0] : 0;
        int8_t layout = (options->inputLayout == wnn::InputOperandLayout::Nhwc) ? 0 : 1;
        int32_t groups = options->groups, fuseOperation = 0;
        uint32_t paddingLeftOp, paddingRightOp, paddingTopOp, paddingBottomOp, strideWeightOp,
            strideHeightOp;
        uint32_t fuseOp = 0, layoutOp = 0, dilationsWidthOp = 0, dilationsHeightOp = 0,
                 groupsOp = 0;

        if (options->autoPad != wnn::AutoPad::Explicit) {
            int32_t height = (options->inputLayout == wnn::InputOperandLayout::Nchw)
                                 ? inputNodeInfo->dimensions[2]
                                 : inputNodeInfo->dimensions[1];
            int32_t width = (options->inputLayout == wnn::InputOperandLayout::Nchw)
                                ? inputNodeInfo->dimensions[3]
                                : inputNodeInfo->dimensions[2];

            utils::ComputeImplicitPaddingForAutoPad<int32_t>(
                options->autoPad, options->dilations[0], height,
                getFilterHeight(filterNodeInfo->dimensions), options->strides[0], paddingTop,
                paddingBottom);
            utils::ComputeImplicitPaddingForAutoPad<int32_t>(
                options->autoPad, options->dilations[1], width,
                getFilterWidth(filterNodeInfo->dimensions), options->strides[1], paddingLeft,
                paddingRight);
        }

        if (options->activation != nullptr &&
            (options->activation->GetFusionType() == FusionType::Relu)) {
            fuseOperation = 1;
        }

        DAWN_TRY(
            mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &paddingLeft, paddingLeftOp));
        DAWN_TRY(
            mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &paddingRight, paddingRightOp));
        DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &paddingTop, paddingTopOp));
        DAWN_TRY(
            mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &paddingBottom, paddingBottomOp));
        DAWN_TRY(
            mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &strideWidth, strideWeightOp));
        DAWN_TRY(
            mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &strideHeight, strideHeightOp));
        DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &fuseOperation, fuseOp));
        DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_BOOL, &layout, layoutOp));

        if (!isGroupConvolution) {
            DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &dilationsWidth,
                                                    dilationsWidthOp));
            DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &dilationsHeight,
                                                    dilationsHeightOp));
        }

        uint32_t transposeFilterIndex;
        if (isGroupConvolution) {
            memInt32Vec.emplace_back(new int(4));
            int32_t* permute = memInt32Vec.back().get();
            getPermuteArray(options->filterLayout, wnn::Conv2dFilterOperandLayout::Ohwi, permute);
            DAWN_TRY(AddTransposeImpl(filterNodeInfo, permute, 4, transposeFilterIndex));
            DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &groups, groupsOp));
            std::vector<uint32_t> inputList = {inputOpIndex,    transposeFilterIndex,
                                               biasOpIndex,     paddingLeftOp,
                                               paddingRightOp,  paddingTopOp,
                                               paddingBottomOp, strideWeightOp,
                                               strideHeightOp,  groupsOp,
                                               fuseOp,          layoutOp};

            DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_GROUPED_CONV_2D, inputList.size(),
                                             inputList.data(), 1, &outputNode->opIndex));
        } else if (isDepthwiseConv2d) {
            groups = 1;
            memInt32Vec.emplace_back(new int(4));
            int32_t* permute = memInt32Vec.back().get();
            getPermuteArray(options->filterLayout, wnn::Conv2dFilterOperandLayout::Ihwo, permute);
            DAWN_TRY(AddTransposeImpl(filterNodeInfo, permute, 4, transposeFilterIndex));
            DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &groups, groupsOp));
            std::vector<uint32_t> inputList = {inputOpIndex,     transposeFilterIndex,
                                               biasOpIndex,      paddingLeftOp,
                                               paddingRightOp,   paddingTopOp,
                                               paddingBottomOp,  strideWeightOp,
                                               strideHeightOp,   groupsOp,
                                               fuseOp,           layoutOp,
                                               dilationsWidthOp, dilationsHeightOp};

            DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_DEPTHWISE_CONV_2D, inputList.size(),
                                             inputList.data(), 1, &outputNode->opIndex));
        } else {
            memInt32Vec.emplace_back(new int(4));
            int32_t* permute = memInt32Vec.back().get();
            getPermuteArray(options->filterLayout, wnn::Conv2dFilterOperandLayout::Ohwi, permute);
            DAWN_TRY(AddTransposeImpl(filterNodeInfo, permute, 4, transposeFilterIndex));
            std::vector<uint32_t> inputList = {inputOpIndex,     transposeFilterIndex,
                                               biasOpIndex,      paddingLeftOp,
                                               paddingRightOp,   paddingTopOp,
                                               paddingBottomOp,  strideWeightOp,
                                               strideHeightOp,   fuseOp,
                                               layoutOp,         dilationsWidthOp,
                                               dilationsHeightOp};
            DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_CONV_2D, inputList.size(),
                                             inputList.data(), 1, &outputNode->opIndex));
        }

        if (options->activation != nullptr) {
            if (options->activation->GetFusionType() == FusionType::Clamp) {
                auto activationNode =
                    CreateOperand("", outputNode->type, outputNode->dimensions, nullptr);
                DAWN_TRY(CheckForNullNode(activationNode, "Failed to create NNAPI operand"));
                auto clamp = reinterpret_cast<const op::FusionClamp*>(options->activation);
                DAWN_TRY(AddClampImpl(outputNode, activationNode, clamp->GetMinValue(),
                                      clamp->GetMaxValue()));
                mGraphNodeMap[conv2d->PrimaryOutput()] = activationNode->opIndex;
            } else if (options->activation->GetFusionType() == FusionType::LeakyRelu) {
                auto activationNode =
                    CreateOperand("", outputNode->type, outputNode->dimensions, nullptr);
                DAWN_TRY(CheckForNullNode(activationNode, "Failed to create NNAPI operand"));
                auto leakyRelu = reinterpret_cast<const op::FusionLeakyRelu*>(options->activation);
                DAWN_TRY(AddLeakyReluImpl(outputNode, activationNode, leakyRelu->GetAlpha()));
                mGraphNodeMap[conv2d->PrimaryOutput()] = activationNode->opIndex;
            } else if (options->activation->GetFusionType() == FusionType::Sigmoid) {
                auto activationNode =
                    CreateOperand("", outputNode->type, outputNode->dimensions, nullptr);
                DAWN_TRY(CheckForNullNode(activationNode, "Failed to create NNAPI operand"));
                DAWN_TRY(AddSigmoidImpl(outputNode, activationNode));
                mGraphNodeMap[conv2d->PrimaryOutput()] = activationNode->opIndex;
            } else if (options->activation->GetFusionType() == FusionType::Relu) {
                auto activationNode =
                    CreateOperand("", outputNode->type, outputNode->dimensions, nullptr);
                DAWN_TRY(CheckForNullNode(activationNode, "Failed to create NNAPI operand"));
                DAWN_TRY(AddReluImpl(outputNode, activationNode));
                mGraphNodeMap[conv2d->PrimaryOutput()] = activationNode->opIndex;
            } else {
                mGraphNodeMap[conv2d->PrimaryOutput()] = outputNode->opIndex;
            }
        } else {
            mGraphNodeMap[conv2d->PrimaryOutput()] = outputNode->opIndex;
        }

        return {};
    }

    MaybeError Graph::AddPad(const op::Pad* pad) {
        DAWN_TRY(CheckStatusCode(ANEURALNETWORKS_OP_FAILED, "nnapi AddPad"));
        return {};
    }

    MaybeError Graph::AddSoftMax(const std::shared_ptr<NodeInfo>& input0Node,
                                 std::shared_ptr<NodeInfo> outputNode) {
        float beta = 1;
        uint32_t betaOp = 0;
        DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_FLOAT32, &beta, betaOp));
        uint32_t inputList[2] = {input0Node->opIndex, betaOp};
        DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_SOFTMAX, 2, inputList, 1,
                                         &(outputNode->opIndex)));
        return {};
    }

    MaybeError Graph::AddUnary(const op::Unary* unary) {
        auto inputOpIndex = mGraphNodeMap[unary->Inputs()[0].Get()];
        auto inputNodeInfo = mIndexNodeMap[inputOpIndex];
        auto outputNode =
            CreateOperand("", inputNodeInfo->type, inputNodeInfo->dimensions, nullptr);
        DAWN_TRY(CheckForNullNode(outputNode, "Failed to create NNAPI operand"));
        mGraphNodeMap[unary->PrimaryOutput()] = outputNode->opIndex;

        switch (unary->GetType()) {
            case op::UnaryOpType::kSigmoid:
                DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_LOGISTIC, 1, &inputOpIndex, 1,
                                                 &outputNode->opIndex));
                break;
            case op::UnaryOpType::kRelu:
                DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_RELU, 1, &inputOpIndex, 1,
                                                 &outputNode->opIndex));
                break;
            case op::UnaryOpType::kSoftmax:
                DAWN_TRY(AddSoftMax(inputNodeInfo, outputNode));
                break;
            default:
                DAWN_TRY(CheckStatusCode(ANEURALNETWORKS_OP_FAILED,
                                         "nnapi AddUnary unsupported operation"));
                break;
        }

        return {};
    }

    MaybeError Graph::AddReduce(const op::Reduce* reduce) {
        DAWN_TRY(CheckStatusCode(ANEURALNETWORKS_OP_FAILED, "nnapi Reduce"));
        return {};
    }

    MaybeError Graph::AddResample2d(const op::Resample2d* resample) {
        DAWN_TRY(CheckStatusCode(ANEURALNETWORKS_OP_FAILED, "nnapi Resample2d"));
        return {};
    }

    MaybeError Graph::AddReshape(const op::Reshape* reshape) {
        auto inputOpIndex = mGraphNodeMap[reshape->Inputs()[0].Get()];
        auto inputNodeInfo = mIndexNodeMap[inputOpIndex];
        std::vector<uint32_t> newShapeDims = {static_cast<uint32_t>(reshape->GetNewShape().size())};
        auto newShapeNode = CreateOperand("const", wnn::OperandType::Int32, newShapeDims,
                                          reshape->GetNewShape().data());
        DAWN_TRY(CheckForNullNode(newShapeNode, "Failed to create NNAPI operand"));
        auto dimensions = reshape->PrimaryOutput()->Shape();
        auto outputNode = CreateOperand("", inputNodeInfo->type, dimensions, nullptr);
        DAWN_TRY(CheckForNullNode(outputNode, "Failed to create NNAPI operand"));
        mGraphNodeMap[reshape->PrimaryOutput()] = outputNode->opIndex;
        uint32_t inputList[2] = {inputOpIndex, newShapeNode->opIndex};
        DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_RESHAPE, 2, inputList, 1,
                                         &outputNode->opIndex));
        return {};
    }

    MaybeError Graph::AddSplit(const op::Split* split) {
        DAWN_TRY(CheckStatusCode(ANEURALNETWORKS_OP_FAILED, "nnapi split"));
        return {};
    }

    MaybeError Graph::AddSqueeze(const op::Squeeze* squeeze) {
        DAWN_TRY(CheckStatusCode(ANEURALNETWORKS_OP_FAILED, "nnapi squeeze"));
        return {};
    }

    MaybeError Graph::AddTranspose(const op::Transpose* transpose) {
        auto input0OpIndex = mGraphNodeMap[transpose->Inputs()[0].Get()];
        auto input0NodeInfo = mIndexNodeMap[input0OpIndex];

        std::vector<int32_t> permutation = transpose->GetPermutation();
        memInt32Vec.emplace_back(new int(permutation.size()));
        int32_t* permute = memInt32Vec.back().get();

        for (size_t i = 0; i < permutation.size(); i++) {
            permute[i] = permutation[i];
        }

        uint32_t index;
        DAWN_TRY(AddTransposeImpl(input0NodeInfo, permute, permutation.size(), index));
        mGraphNodeMap[transpose->PrimaryOutput()] = index;
        return {};
    }

    MaybeError Graph::AddConcat(const op::Concat* concat) {
        DAWN_TRY(CheckStatusCode(ANEURALNETWORKS_OP_FAILED, "nnapi concat"));
        return {};
    }

    MaybeError Graph::AddGemm(const op::Gemm* gemm) {
        auto inputs = gemm->Inputs();
        const GemmOptions* options = gemm->GetOptions();

        // inputs
        auto inputAOpIndex = mGraphNodeMap[gemm->Inputs()[0].Get()];  // A
        auto inputANodeInfo = mIndexNodeMap[inputAOpIndex];
        auto inputBOpIndex = mGraphNodeMap[gemm->Inputs()[1].Get()];  // B
        auto inputBNodeInfo = mIndexNodeMap[inputBOpIndex];

        // output
        auto outputDims = gemm->PrimaryOutput()->Shape();

        if (options->aTranspose) {
            std::shared_ptr<NodeInfo> transposedNodeA;
            memInt32Vec.emplace_back(new int(2));
            int32_t* permute = memInt32Vec.back().get();
            permute[0] = 1;
            permute[1] = 0;
            DAWN_TRY(AddTransposeImpl(inputANodeInfo, permute, 2, inputAOpIndex));
            inputANodeInfo = mIndexNodeMap[inputAOpIndex];
        }
        if (options->bTranspose) {
            std::shared_ptr<NodeInfo> transposedNodeB;
            memInt32Vec.emplace_back(new int(2));
            int32_t* permute = memInt32Vec.back().get();
            permute[0] = 1;
            permute[1] = 0;
            DAWN_TRY(AddTransposeImpl(inputBNodeInfo, permute, 2, inputBOpIndex));
            inputBNodeInfo = mIndexNodeMap[inputBOpIndex];
        }

        // operation: gemm = alpha*A*B + beta*C
        // matMul = A*B
        uint32_t matMuloutIndex;
        DAWN_TRY(AddMatMulImpl(inputANodeInfo, inputBNodeInfo, outputDims, matMuloutIndex));

        uint32_t outputOpIndex = 99999;
        int32_t fuseCode = ANEURALNETWORKS_FUSED_NONE;
        uint32_t fuseCodeOpIndex = 0;
        DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &fuseCode, fuseCodeOpIndex));

        if (options->alpha == 1)
            outputOpIndex = matMuloutIndex;
        else {
            float alpha = options->alpha;
            std::vector<float> alphaVec(1, alpha);
            std::vector<uint32_t> dims({1});
            auto alphaNode = CreateOperand("alpha", wnn::OperandType::Float32, dims, &alphaVec[0]);
            DAWN_TRY(CheckForNullNode(alphaNode, "Failed to create NNAPI operand"));

            // mulNode0 = alpha*matMulNode
            auto mulNode0 = CreateOperand("", inputANodeInfo->type, outputDims, nullptr);
            DAWN_TRY(CheckForNullNode(mulNode0, "Failed to create NNAPI operand"));
            std::vector<uint32_t> inputList = {alphaNode->opIndex, matMuloutIndex, fuseCodeOpIndex};
            DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_MUL, inputList.size(),
                                             inputList.data(), 1, &mulNode0->opIndex));
            outputOpIndex = mulNode0->opIndex;
        }

        if (inputs.size() > 2) {  // Check for C
            auto inputCOpIndex = mGraphNodeMap[gemm->Inputs()[2].Get()];
            auto inputCNodeInfo = mIndexNodeMap[inputCOpIndex];

            auto outputNode = CreateOperand("", inputANodeInfo->type, outputDims, nullptr);
            DAWN_TRY(CheckForNullNode(outputNode, "Failed to create NNAPI operand"));

            if (options->beta == 1) {
                // output = mulNode0 + NodeC
                std::vector<uint32_t> inputList4 = {outputOpIndex, inputCNodeInfo->opIndex,
                                                    fuseCodeOpIndex};
                DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_ADD, inputList4.size(),
                                                 inputList4.data(), 1, &outputNode->opIndex));
            } else {
                // mulNode1 = beta*C
                float beta = options->beta;
                std::vector<float> betaVec(1, beta);
                std::vector<uint32_t> dims({1});
                auto betaNode = CreateOperand("beta", wnn::OperandType::Float32, dims, &betaVec[0]);
                DAWN_TRY(CheckForNullNode(betaNode, "Failed to create NNAPI operand"));

                auto mulNode1 =
                    CreateOperand("", inputCNodeInfo->type, gemm->Inputs()[2]->Shape(), nullptr);
                DAWN_TRY(CheckForNullNode(mulNode1, "Failed to create NNAPI operand"));

                std::vector<uint32_t> inputList2 = {betaNode->opIndex, inputCNodeInfo->opIndex,
                                                    fuseCodeOpIndex};
                DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_MUL, inputList2.size(),
                                                 inputList2.data(), 1, &mulNode1->opIndex));

                // output = mulNode0 + mulNode1
                std::vector<uint32_t> inputList3 = {outputOpIndex, mulNode1->opIndex,
                                                    fuseCodeOpIndex};
                DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_ADD, inputList3.size(),
                                                 inputList3.data(), 1, &outputNode->opIndex));
            }
            mGraphNodeMap[gemm->PrimaryOutput()] = outputNode->opIndex;
        } else {
            mGraphNodeMap[gemm->PrimaryOutput()] = outputOpIndex;
        }
        return {};
    }

    MaybeError Graph::Finish() {
        return {};
    }

    MaybeError Graph::CompileImpl() {
        return mNnapiMgr->Compile(mGraphInputs.size(), mGraphInputs.data(), mGraphOutputs.size(),
                                  mGraphOutputs.data());
    }

    MaybeError Graph::ComputeImpl(NamedInputsBase* inputs, NamedOutputsBase* outputs) {
        if (mNnapiMgr->InitExecutionContext() != NNAPIComputeGraphStatus_Success)
            return DAWN_INTERNAL_ERROR("failed to build graph!");

        int fd;
        ANeuralNetworksMemory* mem;
        auto namedInputs = inputs->GetRecords();
        for (auto& input : mInputNameMap) {
            // All the inputs must be set.
            if (namedInputs.find(input.first) == namedInputs.end()) {
                dawn::ErrorLog() << "The input isn't set";
                return DAWN_INTERNAL_ERROR("The input isn't set");
            }

            auto nodeInfo = input.second;
            size_t index = 0;
            for (; index < mGraphInputs.size(); index++) {
                if (mGraphInputs[index] == nodeInfo->opIndex)
                    break;
            }

            if (index == mGraphInputs.size()) {
                dawn::ErrorLog() << "Failed to find the input node in nodeinfo";
                return DAWN_INTERNAL_ERROR("Failed to find the input node in nodeinfo");
            }

            auto& resource = namedInputs[input.first].resource;
            auto& arrayBuffer = resource.arrayBufferView;
            mNnapiMgr->getFdNNMemory(nodeInfo->opIndex, fd, mem);
            void* inputTensorPtr = reinterpret_cast<void*>(
                mmap(nullptr, arrayBuffer.byteLength, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
            std::memcpy(inputTensorPtr,
                        static_cast<int8_t*>(arrayBuffer.buffer) + arrayBuffer.byteOffset,
                        arrayBuffer.byteLength);
            munmap(inputTensorPtr, arrayBuffer.byteLength);

            int32_t status =
                mNnapiMgr->SetInputMemory(index, nullptr, mem, 0, arrayBuffer.byteLength);
            if (status != ANEURALNETWORKS_NO_ERROR) {
                dawn::ErrorLog() << "Failed ANeuralNetworksExecution_setInputFromMemory";
                return DAWN_INTERNAL_ERROR("Failed ANeuralNetworksExecution_setInputFromMemory");
            }
        }

        auto namedOutputs = outputs->GetRecords();
        for (auto& output : mOutputNameMap) {
            auto nodeInfo = mOutputNameMap[output.first];
            // All the inputs must be set.
            if (namedOutputs.find(output.first) == namedOutputs.end()) {
                dawn::ErrorLog() << "The output isn't set";
                return DAWN_INTERNAL_ERROR("The output isn't set");
            }

            size_t index = 0;
            for (; index < mGraphOutputs.size(); index++) {
                if (mGraphOutputs[index] == nodeInfo->opIndex)
                    break;
            }

            if (index == mGraphOutputs.size()) {
                dawn::ErrorLog() << "Failed to find the output node in nodeinfo";
                return DAWN_INTERNAL_ERROR("Failed to find the output node in nodeinfo");
            }
            mNnapiMgr->getFdNNMemory(nodeInfo->opIndex, fd, mem);
            ArrayBufferView outputBuffer = namedOutputs[output.first].arrayBufferView;
            int32_t status =
                mNnapiMgr->SetOutputMemory(index, nullptr, mem, 0, outputBuffer.byteLength);
            if (status != ANEURALNETWORKS_NO_ERROR) {
                dawn::ErrorLog() << "Failed ANeuralNetworksExecution_setOutputFromMemory";
                return DAWN_INTERNAL_ERROR("Failed ANeuralNetworksExecution_setOutputFromMemory");
            }
        }

        if (mNnapiMgr->ComputeAndWait() != NNAPIComputeGraphStatus_Success) {
            return DAWN_INTERNAL_ERROR("failed to build graph!");
        }

        for (auto namedOutput : outputs->GetRecords()) {
            ArrayBufferView output = namedOutput.second.arrayBufferView;
            DAWN_ASSERT(output.buffer != nullptr && output.byteLength != 0);
            // Get output id with friendly name.
            auto nodeInfo = mOutputNameMap[namedOutput.first];
            mNnapiMgr->getFdNNMemory(nodeInfo->opIndex, fd, mem);
            float* outputTensorPtr = reinterpret_cast<float*>(
                mmap(nullptr, output.byteLength, PROT_READ, MAP_SHARED, fd, 0));
            if (outputTensorPtr == MAP_FAILED) {
                dawn::ErrorLog() << "Failed to mmap output buffer";
                return DAWN_INTERNAL_ERROR("Failed to mmap output buffer");
            }

            std::memcpy(static_cast<int8_t*>(output.buffer) + output.byteOffset, outputTensorPtr,
                        output.byteLength);

            munmap(outputTensorPtr, output.byteLength);
        }

        return {};
    }
}  // namespace webnn::native::nnapi
