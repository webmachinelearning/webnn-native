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

#ifndef WEBNN_NATIVE_NULL_CONTEXT_NULL_H_
#define WEBNN_NATIVE_NULL_CONTEXT_NULL_H_

#include "webnn_native/Context.h"
#include "webnn_native/Graph.h"
#include "webnn_native/GraphBuilder.h"

namespace webnn_native { namespace null {

    // Context
    class Context : public ContextBase {
      public:
        explicit Context(ContextOptions const* options);
        ~Context() override = default;

      private:
        GraphBase* CreateGraphImpl() override;
    };

    // GraphBuilder
    class GraphBuilder : public GraphBuilderBase {
      public:
        explicit GraphBuilder(ContextBase* context);
        ~GraphBuilder() override = default;
    };

    // Graph
    class Graph : public GraphBase {
      public:
        explicit Graph(Context* context);
        ~Graph() override = default;
        virtual MaybeError AddConstant(const op::Constant* constant) override;
        virtual MaybeError AddInput(const op::Input* input) override;
        virtual MaybeError AddOutput(const std::string& name, const OperandBase* ouput) override;
        virtual MaybeError AddBinary(const op::Binary* binary) override;
        virtual MaybeError AddConv2d(const op::Conv2d* conv2d) override;
        virtual MaybeError AddPool2d(const op::Pool2d* pool2d) override;
        virtual MaybeError AddReshape(const op::Reshape* relu) override;
        virtual MaybeError AddTranspose(const op::Transpose* transpose) override;
        virtual MaybeError AddUnary(const op::Unary* unary) override;
        virtual MaybeError AddBatchNorm(const op::BatchNorm* batchNorm) override;
        virtual MaybeError AddLeakyRelu(const op::LeakyRelu* unary) override;
        virtual MaybeError AddConcat(const op::Concat* concat) override;
        virtual MaybeError AddGemm(const op::Gemm* gemm) override;
        virtual MaybeError AddClamp(const op::Clamp* clamp) override;
        virtual MaybeError Finish() override;

      private:
        MaybeError CompileImpl() override;
        MLComputeGraphStatus ComputeImpl(NamedInputsBase* inputs,
                                         NamedOutputsBase* outputs) override;
    };

}}  // namespace webnn_native::null

#endif  // WEBNN_NATIVE_NULL_CONTEXT_NULL_H_
