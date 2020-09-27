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

#include "webnn_native/Model.h"

#include <string>

#include "common/Assert.h"
#include "common/RefCounted.h"

namespace webnn_native {

    ModelBase::ModelBase(ModelBuilderBase* modelBuilder) : ObjectBase(modelBuilder->GetContext()) {
    }

    void ModelBase::Compile(WebnnCompileCallback callback,
                            void* userdata,
                            CompilationOptions const* options) {
        if (DAWN_UNLIKELY(this->IsError())) {
            callback(WebnnCompileStatus_Error, nullptr, "This Model object is an error", userdata);
            return;
        }
        CompileImpl(callback, userdata, options);
    }

    ModelBase::ModelBase(ModelBuilderBase* modelBuilder, ObjectBase::ErrorTag tag)
        : ObjectBase(modelBuilder->GetContext(), tag) {
    }

    // static
    ModelBase* ModelBase::MakeError(ModelBuilderBase* modelBuilder) {
        return new ModelBase(modelBuilder, ObjectBase::kError);
    }

    MaybeError ModelBase::AddConstant(const op::Constant* constant) {
        UNREACHABLE();
    }

    MaybeError ModelBase::AddInput(const op::Input* input) {
        UNREACHABLE();
    }

    MaybeError ModelBase::AddOutput(const std::string& name, const OperandBase* output) {
        UNREACHABLE();
    }

    MaybeError ModelBase::AddBinary(const op::Binary* binary) {
        UNREACHABLE();
    }

    MaybeError ModelBase::AddConv2d(const op::Conv2d* conv2d) {
        UNREACHABLE();
    }

    MaybeError ModelBase::AddPool2d(const op::Pool2d* pool2d) {
        UNREACHABLE();
    }

    MaybeError ModelBase::AddReshape(const op::Reshape* relu) {
        UNREACHABLE();
    }

    MaybeError ModelBase::AddTranspose(const op::Transpose* transpose) {
        UNREACHABLE();
    }

    MaybeError ModelBase::AddUnary(const op::Unary* unary) {
        UNREACHABLE();
    }

    MaybeError ModelBase::Finish() {
        UNREACHABLE();
    }

    void ModelBase::CompileImpl(WebnnCompileCallback callback,
                                void* userdata,
                                CompilationOptions const* options) {
        UNREACHABLE();
    }

}  // namespace webnn_native
