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

#ifndef WEBNN_NATIVE_XNNPACK_GRAPH_XNN_H_
#define WEBNN_NATIVE_XNNPACK_GRAPH_XNN_H_

#include <unordered_map>

#include <xnnpack.h>

#include "webnn/native/Graph.h"
#include "webnn/native/Operand.h"
#include "webnn/native/ops/Binary.h"
#include "webnn/native/ops/Clamp.h"
#include "webnn/native/ops/Concat.h"
#include "webnn/native/ops/Constant.h"
#include "webnn/native/ops/Conv2d.h"
#include "webnn/native/ops/Gemm.h"
#include "webnn/native/ops/Input.h"
#include "webnn/native/ops/LeakyRelu.h"
#include "webnn/native/ops/Pad.h"
#include "webnn/native/ops/Pool2d.h"
#include "webnn/native/ops/Reshape.h"
#include "webnn/native/ops/Split.h"
#include "webnn/native/ops/Squeeze.h"
#include "webnn/native/ops/Transpose.h"
#include "webnn/native/ops/Unary.h"
#include "webnn/native/xnnpack/ContextXNN.h"

namespace webnn::native::xnnpack {

    class Graph : public GraphBase {
      public:
        explicit Graph(Context* context);
        ~Graph() override;

        virtual MaybeError AddConstant(const op::Constant* constant) override;
        virtual MaybeError AddInput(const op::Input* input) override;
        virtual MaybeError AddOutput(std::string_view name, const OperandBase* output) override;
        virtual MaybeError AddBinary(const op::Binary* binary) override;
        virtual MaybeError AddConcat(const op::Concat* concat) override;
        virtual MaybeError AddConv2d(const op::Conv2d* conv2d) override;
        virtual MaybeError AddClamp(const op::Clamp* clamp) override;
        virtual MaybeError AddGemm(const op::Gemm* gemm) override;
        virtual MaybeError AddPad(const op::Pad* pad) override;
        virtual MaybeError AddPool2d(const op::Pool2d* pool2d) override;
        virtual MaybeError AddReshape(const op::Reshape* reshape) override;
        virtual MaybeError AddSplit(const op::Split* split) override;
        virtual MaybeError AddSqueeze(const op::Squeeze* squeeze) override;
        virtual MaybeError AddUnary(const op::Unary* unary) override;
        virtual MaybeError Finish() override;

        virtual MaybeError ComputeImpl(NamedInputsBase* inputs, NamedOutputsBase* outputs) override;

      private:
        MaybeError CompileImpl() override;

        pthreadpool_t GetThreadpool();

        xnn_status DefineXnnTensorValue(xnn_subgraph_t subgraph,
                                        const OperandBase* operand,
                                        uint32_t* id,
                                        const void* data = nullptr);
        xnn_status DefineXnnNode(xnn_subgraph_t subgraph, const op::Constant* constant);
        xnn_status DefineXnnNode(xnn_subgraph_t subgraph, const op::Input* Input);
        xnn_status DefineXnnNode(xnn_subgraph_t subgraph, const op::Binary* binary);
        xnn_status DefineXnnNode(xnn_subgraph_t subgraph, const op::Clamp* clamp);
        xnn_status DefineXnnNode(xnn_subgraph_t subgraph, const op::Concat* concat);
        xnn_status DefineXnnNode(xnn_subgraph_t subgraph, const op::Conv2d* conv2d);
        xnn_status DefineXnnNode(xnn_subgraph_t subgraph, const op::Gemm* gemm);
        xnn_status DefineXnnNode(xnn_subgraph_t subgraph, const op::Pad* pad);
        xnn_status DefineXnnNode(xnn_subgraph_t subgraph, const op::Pool2d* pool2d);
        xnn_status DefineXnnNode(xnn_subgraph_t subgraph, const op::Reshape* reshape);
        xnn_status DefineXnnNode(xnn_subgraph_t subgraph, const op::Split* split);
        xnn_status DefineXnnNode(xnn_subgraph_t subgraph, const op::Squeeze* squeeze);
        xnn_status DefineXnnNode(xnn_subgraph_t subgraph, const op::Unary* unary);

        enum OperatorType {
            Binary,
            Constant,
            Clamp,
            Concat,
            Conv2d,
            Input,
            Gemm,
            Pad,
            Pool2d,
            Reshape,
            Split,
            Squeeze,
            Unary
        };
        struct OperatorInfo {
            OperatorInfo(OperatorType type, const OperatorBase* op) : type(type), op(op) {
            }
            OperatorType type;
            const OperatorBase* op;
        };
        std::vector<OperatorInfo> mOperators;
        std::unordered_map<const OperandBase*, uint32_t> mOperands;
        std::unordered_map<const OperandBase*, uint32_t> mInputs;
        std::unordered_map<const OperandBase*, uint32_t> mOutputs;
        uint32_t mExternalId;

        std::vector<std::unique_ptr<char>> mBuffers;
        std::unordered_map<std::string, xnn_external_value> mExternals;

        xnn_runtime_t mRuntime;
        NamedInputsBase* mNamedInputs;
        NamedOutputsBase* mNamedOutputs;
    };

}  // namespace webnn::native::xnnpack

#endif  // WEBNN_NATIVE_XNNPACK_GRAPH_XNN_H_
