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

#include "webnn/native/Graph.h"

#include <string>

#include "common/Assert.h"
#include "common/Log.h"
#include "common/RefCounted.h"

namespace webnn::native {

    namespace {
        class ErrorGraph final : public GraphBase {
          public:
            ErrorGraph(ContextBase* context) : GraphBase(context, ObjectBase::kError) {
            }

          private:
            MaybeError CompileImpl() override {
                UNREACHABLE();
            }
        };
    }  // namespace

    GraphBase::GraphBase(ContextBase* context) : ObjectBase(context) {
    }

    MaybeError GraphBase::AddConstant(const op::Constant* constant) {
        return DAWN_UNIMPLEMENTED_ERROR("AddConstant");
    }

    MaybeError GraphBase::AddInput(const op::Input* input) {
        return DAWN_UNIMPLEMENTED_ERROR("AddInput");
    }

    MaybeError GraphBase::AddOutput(std::string_view name, const OperandBase* output) {
        return DAWN_UNIMPLEMENTED_ERROR("AddOutput");
    }

    MaybeError GraphBase::AddBatchNorm(const op::BatchNorm* batchNorm) {
        return DAWN_UNIMPLEMENTED_ERROR("AddBatchNorm");
    }

    MaybeError GraphBase::AddSlice(const op::Slice* batchNorm) {
        return DAWN_UNIMPLEMENTED_ERROR("AddSlice");
    }

    MaybeError GraphBase::AddBinary(const op::Binary* binary) {
        return DAWN_UNIMPLEMENTED_ERROR("AddBinary");
    }

    MaybeError GraphBase::AddConv2d(const op::Conv2d* conv2d) {
        return DAWN_UNIMPLEMENTED_ERROR("AddConv2d");
    }

    MaybeError GraphBase::AddConvTranspose2d(const op::ConvTranspose2d* convTranspose2d) {
        return DAWN_UNIMPLEMENTED_ERROR("AddConvTranspose2d");
    }

    MaybeError GraphBase::AddGru(const op::Gru* gru) {
        return DAWN_UNIMPLEMENTED_ERROR("AddGru");
    }

    MaybeError GraphBase::AddPool2d(const op::Pool2d* pool2d) {
        return DAWN_UNIMPLEMENTED_ERROR("AddPool2d");
    }

    MaybeError GraphBase::AddReduce(const op::Reduce* reduce) {
        return DAWN_UNIMPLEMENTED_ERROR("AddReduce");
    }

    MaybeError GraphBase::AddResample2d(const op::Resample2d* resample2d) {
        return DAWN_UNIMPLEMENTED_ERROR("AddResample2d");
    }

    MaybeError GraphBase::AddReshape(const op::Reshape* reshape) {
        return DAWN_UNIMPLEMENTED_ERROR("AddReshape");
    }

    MaybeError GraphBase::AddSqueeze(const op::Squeeze* squeeze) {
        return DAWN_UNIMPLEMENTED_ERROR("AddSqueeze");
    }

    MaybeError GraphBase::AddSplit(const op::Split* split) {
        return DAWN_UNIMPLEMENTED_ERROR("AddSplit");
    }

    MaybeError GraphBase::AddTranspose(const op::Transpose* transpose) {
        return DAWN_UNIMPLEMENTED_ERROR("AddTranspose");
    }

    MaybeError GraphBase::AddUnary(const op::Unary* unary) {
        return DAWN_UNIMPLEMENTED_ERROR("AddUnary");
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

    MaybeError GraphBase::AddPad(const op::Pad* pad) {
        return DAWN_UNIMPLEMENTED_ERROR("AddPad");
    }

    MaybeError GraphBase::AddInstanceNorm(const op::InstanceNorm* instanceNorm) {
        return DAWN_UNIMPLEMENTED_ERROR("AddInstanceNorm");
    }

    MaybeError GraphBase::Finish() {
        return DAWN_UNIMPLEMENTED_ERROR("Finish");
    }

    MaybeError GraphBase::Compile() {
        return CompileImpl();
    }

    MaybeError GraphBase::ComputeImpl(NamedInputsBase* inputs, NamedOutputsBase* outputs) {
        return DAWN_INTERNAL_ERROR("fail to build graph!");
    }

    GraphBase::GraphBase(ContextBase* context, ObjectBase::ErrorTag tag)
        : ObjectBase(context, tag) {
    }

    // static
    GraphBase* GraphBase::MakeError(ContextBase* context) {
        return new ErrorGraph(context);
    }

}  // namespace webnn::native
