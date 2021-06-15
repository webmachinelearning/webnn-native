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

#include "webnn_native/null/ContextNull.h"
#include "common/RefCounted.h"

namespace webnn_native { namespace null {

    // Context
    ContextBase* Create(MLContextOptions const* options) {
        return new Context(reinterpret_cast<ContextOptions const*>(options));
    }

    Context::Context(ContextOptions const* options) {
    }

    GraphBase* Context::CreateGraphImpl() {
        return new Graph(this);
    }

    // GraphBuilder
    GraphBuilder::GraphBuilder(ContextBase* context) : GraphBuilderBase(context) {
    }

    // Graph
    Graph::Graph(Context* context) : GraphBase(context) {
    }

    void Graph::CompileImpl(BuildGraphCallbackDelegate delegate) {
        delegate(MLBuildGraphStatus_Error, nullptr);
    }

    void Graph::ComputeImpl(NamedInputsBase* inputs,
                            MLComputeGraphCallback callback,
                            void* userdata,
                            NamedOutputsBase* outputs) {
    }

    MaybeError Graph::AddConstant(const op::Constant* constant) {
        return {};
    }

    MaybeError Graph::AddInput(const op::Input* input) {
        return {};
    }

    MaybeError Graph::AddOutput(const std::string& name, const OperandBase* output) {
        return {};
    }

    MaybeError Graph::AddBinary(const op::Binary* binary) {
        return {};
    }

    MaybeError Graph::AddConv2d(const op::Conv2d* conv2d) {
        return {};
    }

    MaybeError Graph::AddPool2d(const op::Pool2d* pool2d) {
        return {};
    }

    MaybeError Graph::AddReshape(const op::Reshape* relu) {
        return {};
    }

    MaybeError Graph::AddTranspose(const op::Transpose* transpose) {
        return {};
    }

    MaybeError Graph::AddUnary(const op::Unary* unary) {
        return {};
    }

    MaybeError Graph::AddBatchNorm(const op::BatchNorm* batchNorm) {
        return {};
    }

    MaybeError Graph::AddLeakyRelu(const op::LeakyRelu* unary) {
        return {};
    }

    MaybeError Graph::AddConcat(const op::Concat* concat) {
        return {};
    }

    MaybeError Graph::AddGemm(const op::Gemm* gemm) {
        return {};
    }

    MaybeError Graph::AddClamp(const op::Clamp* clamp) {
        return {};
    }

    MaybeError Graph::Finish() {
        return {};
    }

}}  // namespace webnn_native::null
