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
#include "webnn_native/BackendConnection.h"
#include "webnn_native/Instance.h"

namespace webnn_native::null {

    class Backend : public BackendConnection {
      public:
        Backend(InstanceBase* instance) : BackendConnection(instance, wnn::BackendType::Null) {
        }

        ContextBase* CreateContext(ContextOptions const* options) {
            return new Context(options);
        }

#if defined(WEBNN_ENABLE_GPU_BUFFER)
        ContextBase* CreateContextWithGpuDevice(WGPUDevice device) {
            return new Context(device);
        }
#endif
    };

    BackendConnection* Connect(InstanceBase* instance) {
        return new Backend(instance);
    }

    // Context
    ContextBase* Create(WNNContextOptions const* options) {
        return new Context(reinterpret_cast<ContextOptions const*>(options));
    }

    Context::Context(ContextOptions const* options) {
    }

#if defined(WEBNN_ENABLE_GPU_BUFFER)
    Context::Context(WGPUDevice device) {
    }
#endif

    GraphBase* Context::CreateGraphImpl() {
        return new Graph(this);
    }

    // GraphBuilder
    GraphBuilder::GraphBuilder(ContextBase* context) : GraphBuilderBase(context) {
    }

    // Graph
    Graph::Graph(Context* context) : GraphBase(context) {
    }

    MaybeError Graph::CompileImpl() {
        return {};
    }

    MaybeError Graph::ComputeImpl(NamedInputsBase* inputs, NamedOutputsBase* outputs) {
        return {};
    }

    MaybeError Graph::AddConstant(const op::Constant* constant) {
        return {};
    }

    MaybeError Graph::AddInput(const op::Input* input) {
        return {};
    }

    MaybeError Graph::AddOutput(std::string_view name, const OperandBase* output) {
        return {};
    }

    MaybeError Graph::AddBinary(const op::Binary* binary) {
        return {};
    }

    MaybeError Graph::AddConv2d(const op::Conv2d* conv2d) {
        return {};
    }

    MaybeError Graph::AddGru(const op::Gru* gru) {
        return {};
    }

    MaybeError Graph::AddPad(const op::Pad* pad) {
        return {};
    }

    MaybeError Graph::AddPool2d(const op::Pool2d* pool2d) {
        return {};
    }

    MaybeError Graph::AddReduce(const op::Reduce* reduce) {
        return {};
    }

    MaybeError Graph::AddResample2d(const op::Resample2d* resample2d) {
        return {};
    }

    MaybeError Graph::AddReshape(const op::Reshape* relu) {
        return {};
    }

    MaybeError Graph::AddSqueeze(const op::Squeeze* squeeze) {
        return {};
    }

    MaybeError Graph::AddSlice(const op::Slice* slice) {
        return {};
    }

    MaybeError Graph::AddSplit(const op::Split* split) {
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

    MaybeError Graph::AddConcat(const op::Concat* concat) {
        return {};
    }

    MaybeError Graph::AddGemm(const op::Gemm* gemm) {
        return {};
    }

    MaybeError Graph::AddClamp(const op::Clamp* clamp) {
        return {};
    }

    MaybeError Graph::AddInstanceNorm(const op::InstanceNorm* instanceNorm) {
        return {};
    }

    MaybeError Graph::Finish() {
        return {};
    }

}  // namespace webnn_native::null
