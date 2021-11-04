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
        Napi::Value Abs(const Napi::CallbackInfo& info);
        Napi::Value Add(const Napi::CallbackInfo& info);
        Napi::Value AveragePool2d(const Napi::CallbackInfo& info);
        Napi::Value BatchNorm(const Napi::CallbackInfo& info);
        Napi::Value Ceil(const Napi::CallbackInfo& info);
        Napi::Value Clamp(const Napi::CallbackInfo& info);
        Napi::Value Concat(const Napi::CallbackInfo& info);
        Napi::Value Constant(const Napi::CallbackInfo& info);
        Napi::Value Conv2d(const Napi::CallbackInfo& info);
        Napi::Value Cos(const Napi::CallbackInfo& info);
        Napi::Value Div(const Napi::CallbackInfo& info);
        Napi::Value Exp(const Napi::CallbackInfo& info);
        Napi::Value Floor(const Napi::CallbackInfo& info);
        Napi::Value Gemm(const Napi::CallbackInfo& info);
        Napi::Value Gru(const Napi::CallbackInfo& info);
        Napi::Value HardSwish(const Napi::CallbackInfo& info);
        Napi::Value Input(const Napi::CallbackInfo& info);
        Napi::Value InstanceNorm(const Napi::CallbackInfo& info);
        Napi::Value LeakyRelu(const Napi::CallbackInfo& info);
        Napi::Value Log(const Napi::CallbackInfo& info);
        Napi::Value Matmul(const Napi::CallbackInfo& info);
        Napi::Value Max(const Napi::CallbackInfo& info);
        Napi::Value MaxPool2d(const Napi::CallbackInfo& info);
        Napi::Value Min(const Napi::CallbackInfo& info);
        Napi::Value Mul(const Napi::CallbackInfo& info);
        Napi::Value Neg(const Napi::CallbackInfo& info);
        Napi::Value Pad(const Napi::CallbackInfo& info);
        Napi::Value Pow(const Napi::CallbackInfo& info);
        Napi::Value ReduceL1(const Napi::CallbackInfo& info);
        Napi::Value ReduceL2(const Napi::CallbackInfo& info);
        Napi::Value ReduceMax(const Napi::CallbackInfo& info);
        Napi::Value ReduceMean(const Napi::CallbackInfo& info);
        Napi::Value ReduceMin(const Napi::CallbackInfo& info);
        Napi::Value ReduceProduct(const Napi::CallbackInfo& info);
        Napi::Value ReduceSum(const Napi::CallbackInfo& info);
        Napi::Value Relu(const Napi::CallbackInfo& info);
        Napi::Value Resample2d(const Napi::CallbackInfo& info);
        Napi::Value Reshape(const Napi::CallbackInfo& info);
        Napi::Value Sigmoid(const Napi::CallbackInfo& info);
        Napi::Value Sin(const Napi::CallbackInfo& info);
        Napi::Value Slice(const Napi::CallbackInfo& info);
        Napi::Value Softmax(const Napi::CallbackInfo& info);
        Napi::Value Split(const Napi::CallbackInfo& info);
        Napi::Value Squeeze(const Napi::CallbackInfo& info);
        Napi::Value Sub(const Napi::CallbackInfo& info);
        Napi::Value Tan(const Napi::CallbackInfo& info);
        Napi::Value Tanh(const Napi::CallbackInfo& info);
        Napi::Value Transpose(const Napi::CallbackInfo& info);

        Napi::Value Build(const Napi::CallbackInfo& info);

        ml::GraphBuilder mImpl;
    };

}  // namespace node

#endif  // NODE_MODEL_BUILDER_H_
