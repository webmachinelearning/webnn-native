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

#ifndef WEBNN_NATIVE_IE_MODEL_IE_H_
#define WEBNN_NATIVE_IE_MODEL_IE_H_

#include <ngraph_c_api.h>
#include <map>
#include <set>
#include <unordered_set>

#include "webnn/native/Error.h"
#include "webnn/native/Graph.h"
#include "webnn/native/Operand.h"
#include "webnn/native/Operator.h"
#include "webnn/native/openvino/ContextIE.h"
#include "webnn/native/ops/BatchNorm.h"
#include "webnn/native/ops/Binary.h"
#include "webnn/native/ops/Clamp.h"
#include "webnn/native/ops/Concat.h"
#include "webnn/native/ops/Constant.h"
#include "webnn/native/ops/Conv2d.h"
#include "webnn/native/ops/Gemm.h"
#include "webnn/native/ops/Gru.h"
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

namespace webnn::native::ie {

    class Graph : public GraphBase {
      public:
        explicit Graph(Context* context);
        ~Graph() override;

        virtual MaybeError AddConstant(const op::Constant* constant) override;
        virtual MaybeError AddInput(const op::Input* input) override;
        virtual MaybeError AddOutput(std::string_view name, const OperandBase* ouput) override;
        virtual MaybeError AddBatchNorm(const op::BatchNorm* batchNorm) override;
        virtual MaybeError AddBinary(const op::Binary* binary) override;
        virtual MaybeError AddClamp(const op::Clamp* clamp) override;
        virtual MaybeError AddConv2d(const op::Conv2d* conv2d) override;
        virtual MaybeError AddConvTranspose2d(const op::ConvTranspose2d* convTranspose2d) override;
        virtual MaybeError AddGru(const op::Gru* gru) override;
        virtual MaybeError AddPad(const op::Pad* pad) override;
        virtual MaybeError AddPool2d(const op::Pool2d* pool2d) override;
        virtual MaybeError AddReduce(const op::Reduce* reduce) override;
        virtual MaybeError AddResample2d(const op::Resample2d* resample2d) override;
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

        virtual MaybeError ComputeImpl(NamedInputsBase* inputs, NamedOutputsBase* outputs) override;

      private:
        MaybeError CompileImpl() override;

        // Map the input name to IE internal input number.
        std::map<std::string, size_t> mInputIdMap;
        // Map the output name to IE internal original output name that will be updated after
        // TransposeSinking.
        std::map<std::string, std::string> mOutputNameMap;
        // The outputs will be optimized after TransposeSinking, the name of it also will be
        // updated, so the mOriginalNameMap is to get the index of output in network.
        std::map<std::string, size_t> mOriginalNameMap;
        // Map the operand to IE internal id
        std::map<const OperandBase*, std::string> mOperandIdMap;
        // store the constant operands
        std::unordered_set<const OperandBase*> mConstantSet;
        std::map<const OperandBase*, const ngraph_node_t*> mGraphNodeMap;
        std::vector<ngraph_node_t*> mGraphOutputs;
        std::vector<ngraph_node_t*> mGraphInputs;
        ie_core_t* mInferEngineCore;
        ie_network_t* mInferEngineNetwork;
        ie_infer_request_t* mInferEngineRequest;
    };

}  // namespace webnn::native::ie

#endif  // WEBNN_NATIVE_IE_MODEL_IE_H_
