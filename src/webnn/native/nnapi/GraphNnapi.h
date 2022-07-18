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

#ifndef WEBNN_NATIVE_NNAPI_MODEL_NN_H_
#define WEBNN_NATIVE_NNAPI_MODEL_NN_H_

#include <map>
#include <set>
#include <unordered_set>
#include <vector>

#include "webnn/native/Error.h"
#include "webnn/native/Graph.h"
#include "webnn/native/Operand.h"
#include "webnn/native/Operator.h"
#include "webnn/native/nnapi/ContextNnapi.h"
#include "webnn/native/ops/BatchNorm.h"
#include "webnn/native/ops/Binary.h"
#include "webnn/native/ops/Clamp.h"
#include "webnn/native/ops/Concat.h"
#include "webnn/native/ops/Constant.h"
#include "webnn/native/ops/Conv2d.h"
#include "webnn/native/ops/Gemm.h"
#include "webnn/native/ops/Input.h"
#include "webnn/native/ops/InstanceNorm.h"
#include "webnn/native/ops/LeakyRelu.h"
#include "webnn/native/ops/Pad.h"
#include "webnn/native/ops/Pool2d.h"
#include "webnn/native/ops/Reduce.h"
#include "webnn/native/ops/Resample2d.h"
#include "webnn/native/ops/Reshape.h"
#include "webnn/native/ops/Slice.h"
#include "webnn/native/ops/Split.h"
#include "webnn/native/ops/Squeeze.h"
#include "webnn/native/ops/Transpose.h"
#include "webnn/native/ops/Unary.h"

#include "NeuralNetworksTypes.h"
#include "NnapiManager.h"
#include "NnapiUtils.h"
#include "nnapi_implementation.h"
#include "webnn/native/nnapi/ErrorNnapi.h"

namespace webnn::native::nnapi {

    class Graph : public GraphBase {
      public:
        explicit Graph(Context* context);
        ~Graph() override;

        virtual MaybeError AddConstant(const op::Constant* constant) override;
        virtual MaybeError AddInput(const op::Input* input) override;
        virtual MaybeError AddOutput(const std::string_view name,
                                     const OperandBase* ouput) override;
        virtual MaybeError AddBatchNorm(const op::BatchNorm* batchNorm) override;
        virtual MaybeError AddBinary(const op::Binary* binary) override;
        virtual MaybeError AddClamp(const op::Clamp* clamp) override;
        virtual MaybeError AddConv2d(const op::Conv2d* conv2d) override;
        virtual MaybeError AddPad(const op::Pad* pad) override;
        virtual MaybeError AddPool2d(const op::Pool2d* pool2d) override;
        virtual MaybeError AddReduce(const op::Reduce* reduce) override;
        virtual MaybeError AddResample2d(const op::Resample2d* resample) override;
        virtual MaybeError AddReshape(const op::Reshape* reshape) override;
        virtual MaybeError AddSlice(const op::Slice* slice) override;
        virtual MaybeError AddSplit(const op::Split* split) override;
        virtual MaybeError AddSqueeze(const op::Squeeze* squeeze) override;
        virtual MaybeError AddTranspose(const op::Transpose* transpose) override;
        virtual MaybeError AddUnary(const op::Unary* unary) override;
        virtual MaybeError AddConcat(const op::Concat* concat) override;
        virtual MaybeError AddGemm(const op::Gemm* Gemm) override;
        virtual MaybeError AddInstanceNorm(const op::InstanceNorm* InstanceNorm) override;
        virtual MaybeError Finish() override;

        MaybeError AddSoftMax(const std::shared_ptr<NodeInfo>& input0Node,
                              std::shared_ptr<NodeInfo> outputNode);

        virtual MaybeError ComputeImpl(NamedInputsBase* inputs, NamedOutputsBase* outputs) override;

      private:
        uint32_t getOperandIdx() {
            return mOperandCount++;
        }

        MaybeError AddTransposeImpl(const std::shared_ptr<NodeInfo>& node,
                                    int32_t* permute,
                                    uint32_t permuteSize,
                                    uint32_t& outputIndex);
        MaybeError AddExpandDimsImpl(const std::shared_ptr<NodeInfo>& node,
                                     int32_t dim_index,
                                     uint32_t& index);
        MaybeError AddMatMulImpl(const std::shared_ptr<NodeInfo>& input0NodeInfo,
                                 const std::shared_ptr<NodeInfo>& input1NodeInfo,
                                 std::vector<int32_t> dims,
                                 uint32_t& outputIndex);
        MaybeError AddClampImpl(const std::shared_ptr<NodeInfo>& inputNode,
                                std::shared_ptr<NodeInfo> outputNode,
                                float min,
                                float max);
        MaybeError AddLeakyReluImpl(const std::shared_ptr<NodeInfo>& inputNode,
                                    std::shared_ptr<NodeInfo> outputNode,
                                    float alpha);
        MaybeError AddSigmoidImpl(const std::shared_ptr<NodeInfo>& inputNode,
                                  std::shared_ptr<NodeInfo> outputNode);
        MaybeError AddReluImpl(const std::shared_ptr<NodeInfo>& inputNode,
                               std::shared_ptr<NodeInfo> outputNode);

        template <class T>
        std::shared_ptr<NodeInfo> CreateOperand(std::string name,
                                                wnn::OperandType type,
                                                std::vector<T> dims,
                                                const void* buffer = nullptr) {
            std::shared_ptr<NodeInfo> node = std::make_shared<NodeInfo>();
            node->type = type;
            for (size_t i = 0; i < dims.size(); i++) {
                node->dimensions.push_back(static_cast<uint32_t>(dims[i]));
            }

            MaybeError error;
            if (buffer) {
                error = mNnapiMgr->CreateOperandAndSetMemory(name, node, buffer);
            } else {
                error = mNnapiMgr->CreateOperand(node);
            }

            if (error.IsError()) {
                return std::make_shared<NodeInfo>();
            }

            mIndexNodeMap[node->opIndex] = node;
            return node;
        }

        std::shared_ptr<NodeInfo> CreateOperand(std::string name,
                                                const OperandDescriptor* desc,
                                                const void* buffer = nullptr) {
            std::shared_ptr<NodeInfo> node = std::make_shared<NodeInfo>();
            node->type = desc->type;
            if (desc->dimensionsCount == 0) {
                node->dimensions.push_back(static_cast<uint32_t>(1));
            } else {
                for (size_t i = 0; i < desc->dimensionsCount; i++) {
                    node->dimensions.push_back(static_cast<uint32_t>(desc->dimensions[i]));
                }
            }

            MaybeError error;
            if (buffer) {
                error = mNnapiMgr->CreateOperandAndSetMemory(name, node, buffer);
            } else {
                error = mNnapiMgr->CreateOperand(node);
            }

            if (error.IsError()) {
                return std::make_shared<NodeInfo>();
            }

            mIndexNodeMap[node->opIndex] = node;
            return node;
        }

        std::shared_ptr<NodeInfo> CreateIOOperand(std::string name,
                                                  const OperandDescriptor* desc,
                                                  bool input) {
            std::shared_ptr<NodeInfo> node = std::make_shared<NodeInfo>();
            node->type = desc->type;
            for (size_t i = 0; i < desc->dimensionsCount; i++) {
                node->dimensions.push_back(static_cast<uint32_t>(desc->dimensions[i]));
            }

            MaybeError error = mNnapiMgr->CreateInputOutputOperand(node->name, node, input);
            if (error.IsError()) {
                return std::make_shared<NodeInfo>();
            }

            mIndexNodeMap[node->opIndex] = node;
            if (input) {
                mInputNameMap[name] = node;
                mGraphInputs.push_back(node->opIndex);
            } else {
                mOutputNameMap[name] = node;
                mGraphOutputs.push_back(node->opIndex);
            }
            return node;
        }

        std::shared_ptr<NodeInfo> CreateIOOperand(std::string name,
                                                  const std::shared_ptr<NodeInfo>& node,
                                                  bool input) {
            MaybeError error = mNnapiMgr->CreateInputOutputOperand(name, node, input);
            if (error.IsError()) {
                return std::make_shared<NodeInfo>();
            }

            mIndexNodeMap[node->opIndex] = node;
            if (input) {
                mInputNameMap[name] = node;
                mGraphInputs.push_back(node->opIndex);
            } else {
                mOutputNameMap[name] = node;
                mGraphOutputs.push_back(node->opIndex);
            }
            return node;
        }

        MaybeError CompileImpl() override;
        // Map the input name to NNAPI internal input number.
        std::map<std::string, std::shared_ptr<NodeInfo>> mInputNameMap;
        // Map the output name to NNAPI internal original output name that will be updated after
        // TransposeSinking.
        std::map<std::string, std::shared_ptr<NodeInfo>> mOutputNameMap;
        // store the constant operands
        // std::unordered_set<const OperandBase*> mConstantSet;
        std::map<const OperandBase*, uint32_t> mGraphNodeMap;  // Add operand index
        std::vector<uint32_t> mGraphOutputs;
        std::vector<uint32_t> mGraphInputs;
        std::map<uint32_t, std::shared_ptr<NodeInfo>> mIndexNodeMap;
        uint32_t mOperandCount;
        // ANeuralNetworksOperandType mScalarInt32Operand, mScalarBoolOperand;
        std::shared_ptr<NnapiManager> mNnapiMgr;
        std::vector<std::unique_ptr<int32_t>> memInt32Vec;
    };

}  // namespace webnn::native::nnapi

#endif  // WEBNN_NATIVE_NNAPI_MODEL_NN_H_
