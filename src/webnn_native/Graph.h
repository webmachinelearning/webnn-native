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

#ifndef WEBNN_NATIVE_GRAPH_H_
#define WEBNN_NATIVE_GRAPH_H_

#include "common/RefCounted.h"
#include "webnn_native/Context.h"
#include "webnn_native/Error.h"
#include "webnn_native/Forward.h"
#include "webnn_native/GraphBuilder.h"
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

    class GraphBase : public ObjectBase {
      public:
        explicit GraphBase(ContextBase* context);
        virtual ~GraphBase() = default;

        // Webnn API
        void Compute(NamedInputsBase* inputs,
                     MLComputeGraphCallback callback,
                     void* userdata,
                     NamedOutputsBase* outputs = nullptr);

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
        virtual void Compile(BuildGraphCallbackDelgate delgate);

      private:
        virtual void CompileImpl(BuildGraphCallbackDelgate delgate) = 0;
        virtual void ComputeImpl(NamedInputsBase* inputs,
                                 MLComputeGraphCallback callback,
                                 void* userdata,
                                 NamedOutputsBase* outputs) = 0;
    };
}  // namespace webnn_native

#endif  // WEBNN_NATIVE_MODEL_H_