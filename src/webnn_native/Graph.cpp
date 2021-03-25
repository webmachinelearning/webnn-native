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

#include "webnn_native/Graph.h"

#include <string>

#include "common/Assert.h"
#include "common/Log.h"
#include "common/RefCounted.h"

namespace webnn_native {

    GraphBase::GraphBase(ContextBase* context) : ObjectBase(context) {
    }

    void GraphBase::Compute(NamedInputsBase* inputs,
                            MLComputeGraphCallback callback,
                            void* userdata,
                            NamedOutputsBase* outputs) {
        ComputeImpl(inputs, callback, userdata, outputs);
    }

    MLComputeGraphStatus GraphBase::ComputeSync(NamedInputsBase* inputs,
                                                NamedOutputsBase* outputs) {
        return ComputeSyncImpl(inputs, outputs);
    }

    MaybeError GraphBase::AddConstant(const op::Constant* constant) {
        return DAWN_UNIMPLEMENTED_ERROR("AddConstant");
    }

    MaybeError GraphBase::AddInput(const op::Input* input) {
        return DAWN_UNIMPLEMENTED_ERROR("AddInput");
    }

    MaybeError GraphBase::AddOutput(const std::string& name, const OperandBase* output) {
        return DAWN_UNIMPLEMENTED_ERROR("AddOutput");
    }

    MaybeError GraphBase::AddBatchNorm(const op::BatchNorm* batchNorm) {
        return DAWN_UNIMPLEMENTED_ERROR("AddBatchNorm");
    }

    MaybeError GraphBase::AddBinary(const op::Binary* binary) {
        return DAWN_UNIMPLEMENTED_ERROR("AddBinary");
    }

    MaybeError GraphBase::AddConv2d(const op::Conv2d* conv2d) {
        return DAWN_UNIMPLEMENTED_ERROR("AddConv2d");
    }

    MaybeError GraphBase::AddPool2d(const op::Pool2d* pool2d) {
        return DAWN_UNIMPLEMENTED_ERROR("AddPool2d");
    }

    MaybeError GraphBase::AddReshape(const op::Reshape* relu) {
        return DAWN_UNIMPLEMENTED_ERROR("AddReshape");
    }

    MaybeError GraphBase::AddTranspose(const op::Transpose* transpose) {
        return DAWN_UNIMPLEMENTED_ERROR("AddTranspose");
    }

    MaybeError GraphBase::AddUnary(const op::Unary* unary) {
        return DAWN_UNIMPLEMENTED_ERROR("AddUnary");
    }

    MaybeError GraphBase::AddLeakyRelu(const op::LeakyRelu* leakyRelu) {
        return DAWN_UNIMPLEMENTED_ERROR("AddLeakyRelu");
    }

    MaybeError GraphBase::AddConcat(const op::Concat* concat) {
        return DAWN_UNIMPLEMENTED_ERROR("AddConcat");
    }

    MaybeError GraphBase::AddGemm(const op::Gemm* gemm) {
        return DAWN_UNIMPLEMENTED_ERROR("AddGemm");
    }

    MaybeError GraphBase::AddClamp(const op::Clamp* clamp) {
        return DAWN_UNIMPLEMENTED_ERROR("AddClamp");
    }

    MaybeError GraphBase::Finish() {
        UNREACHABLE();
    }

    void GraphBase::Compile(BuildGraphCallbackDelegate delegate) {
        CompileImpl(delegate);
    }

    MLBuildGraphStatus GraphBase::CompileSync() {
        return CompileSyncImpl();
    }

    MLBuildGraphStatus GraphBase::CompileSyncImpl() {
        dawn::ErrorLog() << "Unimplemented";
        return MLBuildGraphStatus_Error;
    }

    MLComputeGraphStatus GraphBase::ComputeSyncImpl(NamedInputsBase* inputs,
                                                    NamedOutputsBase* outputs) {
        dawn::ErrorLog() << "Unimplemented";
        return MLComputeGraphStatus_Error;
    }

}  // namespace webnn_native
