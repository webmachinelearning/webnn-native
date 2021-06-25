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

#ifndef NODE_MODEL_BUILDER_H_
#define NODE_MODEL_BUILDER_H_

#include <napi.h>
#include <webnn/webnn_cpp.h>

namespace node {

    class GraphBuilder : public Napi::ObjectWrap<GraphBuilder> {
      public:
        static Napi::Object Initialize(Napi::Env env, Napi::Object exports);
        static Napi::FunctionReference constructor;

        explicit GraphBuilder(const Napi::CallbackInfo& info);
        ~GraphBuilder() = default;

      private:
        Napi::Value Constant(const Napi::CallbackInfo& info);
        Napi::Value Input(const Napi::CallbackInfo& info);
        Napi::Value Add(const Napi::CallbackInfo& info);
        Napi::Value BatchNorm(const Napi::CallbackInfo& info);
        Napi::Value Mul(const Napi::CallbackInfo& info);
        Napi::Value Matmul(const Napi::CallbackInfo& info);
        Napi::Value Conv2d(const Napi::CallbackInfo& info);
        Napi::Value Concat(const Napi::CallbackInfo& info);
        Napi::Value Clamp(const Napi::CallbackInfo& info);
        Napi::Value Gemm(const Napi::CallbackInfo& info);
        Napi::Value MaxPool2d(const Napi::CallbackInfo& info);
        Napi::Value AveragePool2d(const Napi::CallbackInfo& info);
        Napi::Value Relu(const Napi::CallbackInfo& info);
        Napi::Value LeakyRelu(const Napi::CallbackInfo& info);
        Napi::Value Reshape(const Napi::CallbackInfo& info);
        Napi::Value Softmax(const Napi::CallbackInfo& info);
        Napi::Value Transpose(const Napi::CallbackInfo& info);
        Napi::Value Build(const Napi::CallbackInfo& info);
        Napi::Value BuildSync(const Napi::CallbackInfo& info);

        ml::GraphBuilder mImpl;
    };

}  // namespace node

#endif  // NODE_MODEL_BUILDER_H_
