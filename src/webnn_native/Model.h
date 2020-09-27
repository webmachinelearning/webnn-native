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

#ifndef WEBNN_NATIVE_MODEL_H_
#define WEBNN_NATIVE_MODEL_H_

#include "common/RefCounted.h"
#include "webnn_native/Compilation.h"
#include "webnn_native/Error.h"
#include "webnn_native/Forward.h"
#include "webnn_native/ModelBuilder.h"
#include "webnn_native/ObjectBase.h"
#include "webnn_native/Operand.h"
#include "webnn_native/webnn_platform.h"

namespace webnn_native {

    namespace op {
        class Constant;
        class Input;
        class Binary;
        class Conv2d;
        class Pool2d;
        class Reshape;
        class Transpose;
        class Unary;
    }  // namespace op

    class ModelBase : public ObjectBase {
      public:
        explicit ModelBase(ModelBuilderBase* modelBuilder);
        virtual ~ModelBase() = default;

        // static
        static ModelBase* MakeError(ModelBuilderBase* modelBuilder);

        // Dawn API
        void Compile(WebnnCompileCallback callback,
                     void* userdata,
                     CompilationOptions const* options);

        virtual MaybeError AddConstant(const op::Constant* constant);
        virtual MaybeError AddInput(const op::Input* input);
        virtual MaybeError AddOutput(const std::string& name, const OperandBase* output);
        virtual MaybeError AddBinary(const op::Binary* binary);
        virtual MaybeError AddConv2d(const op::Conv2d* conv2d);
        virtual MaybeError AddPool2d(const op::Pool2d* pool2d);
        virtual MaybeError AddReshape(const op::Reshape* relu);
        virtual MaybeError AddTranspose(const op::Transpose* transpose);
        virtual MaybeError AddUnary(const op::Unary* unary);
        virtual MaybeError Finish();

      private:
        ModelBase(ModelBuilderBase* modelBuilder, ObjectBase::ErrorTag tag);
        virtual void CompileImpl(WebnnCompileCallback callback,
                                 void* userdata,
                                 CompilationOptions const* options);
    };
}  // namespace webnn_native

#endif  // WEBNN_NATIVE_MODEL_H_