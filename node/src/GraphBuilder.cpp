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

#include "GraphBuilder.h"

#include <iostream>
#include <vector>

#include "Context.h"
#include "Graph.h"
#include "Operand.h"
#include "Utils.h"
#include "ops/BatchNorm.h"
#include "ops/Clamp.h"
#include "ops/Concat.h"
#include "ops/Constant.h"
#include "ops/Conv2d.h"
#include "ops/Gemm.h"
#include "ops/Input.h"
#include "ops/LeakyRelu.h"
#include "ops/Pad.h"
#include "ops/Pool2d.h"
#include "ops/ReduceMean.h"
#include "ops/Resample.h"
#include "ops/Reshape.h"
#include "ops/Transpose.h"

Napi::FunctionReference node::GraphBuilder::constructor;

#define BUILD_BINARY(op)                                                            \
    WEBNN_NODE_ASSERT(info.Length() == 2, "The number of arguments is invalid.");   \
    std::vector<napi_value> args;                                                   \
    ml::Operand a;                                                                  \
    WEBNN_NODE_ASSERT(GetOperand(info[0], a, args), "The a parameter is invalid."); \
    ml::Operand b;                                                                  \
    WEBNN_NODE_ASSERT(GetOperand(info[1], b, args), "The b parameter is invalid."); \
    Napi::Object object = Operand::constructor.New(args);                           \
    Operand* operand = Napi::ObjectWrap<Operand>::Unwrap(object);                   \
    operand->SetImpl(mImpl.op(a, b));                                               \
    return object;

#define BUILD_UNARY(op)                                                                     \
    WEBNN_NODE_ASSERT(info.Length() == 1, "The number of arguments is invalid.");           \
    std::vector<napi_value> args;                                                           \
    ml::Operand input;                                                                      \
    WEBNN_NODE_ASSERT(GetOperand(info[0], input, args), "The input parameter is invalid."); \
    Napi::Object object = Operand::constructor.New(args);                                   \
    Operand* operand = Napi::ObjectWrap<Operand>::Unwrap(object);                           \
    operand->SetImpl(mImpl.op(input));                                                      \
    return object;

namespace node {

    GraphBuilder::GraphBuilder(const Napi::CallbackInfo& info)
        : Napi::ObjectWrap<GraphBuilder>(info) {
        Napi::Object object = info[0].As<Napi::Object>();
        node::Context* context = Napi::ObjectWrap<node::Context>::Unwrap(object);
        mImpl = ml::CreateGraphBuilder(context->GetImpl());
    }

    Napi::Value GraphBuilder::Constant(const Napi::CallbackInfo& info) {
        return op::Constant::Build(info, mImpl);
    }

    Napi::Value GraphBuilder::Input(const Napi::CallbackInfo& info) {
        return op::Input::Build(info, mImpl);
    }

    Napi::Value GraphBuilder::Add(const Napi::CallbackInfo& info) {
        BUILD_BINARY(Add);
    }

    Napi::Value GraphBuilder::Mul(const Napi::CallbackInfo& info) {
        BUILD_BINARY(Mul);
    }

    Napi::Value GraphBuilder::Matmul(const Napi::CallbackInfo& info) {
        BUILD_BINARY(Matmul);
    }

    Napi::Value GraphBuilder::Div(const Napi::CallbackInfo& info) {
        BUILD_BINARY(Div);
    }

    Napi::Value GraphBuilder::BatchNorm(const Napi::CallbackInfo& info) {
        return op::BatchNorm::Build(info, mImpl);
    }

    Napi::Value GraphBuilder::Conv2d(const Napi::CallbackInfo& info) {
        return op::Conv2d::Build(info, mImpl);
    }

    Napi::Value GraphBuilder::Concat(const Napi::CallbackInfo& info) {
        return op::Concat::Build(info, mImpl);
    }

    Napi::Value GraphBuilder::Gemm(const Napi::CallbackInfo& info) {
        return op::Gemm::Build(info, mImpl);
    }

    Napi::Value GraphBuilder::Clamp(const Napi::CallbackInfo& info) {
        return op::Clamp::Build(info, mImpl);
    }

    Napi::Value GraphBuilder::MaxPool2d(const Napi::CallbackInfo& info) {
        return op::Pool2d::Build(info, mImpl, op::Pool2dType::kMaxPool2d);
    }

    Napi::Value GraphBuilder::AveragePool2d(const Napi::CallbackInfo& info) {
        return op::Pool2d::Build(info, mImpl, op::Pool2dType::kAveragePool2d);
    }

    Napi::Value GraphBuilder::ReduceMean(const Napi::CallbackInfo& info) {
        return op::ReduceMean::Build(info, mImpl);
    }

    Napi::Value GraphBuilder::Resample(const Napi::CallbackInfo& info) {
        return op::Resample::Build(info, mImpl);
    }

    Napi::Value GraphBuilder::Relu(const Napi::CallbackInfo& info) {
        BUILD_UNARY(Relu);
    }

    Napi::Value GraphBuilder::Softmax(const Napi::CallbackInfo& info) {
        BUILD_UNARY(Softmax);
    }

    Napi::Value GraphBuilder::LeakyRelu(const Napi::CallbackInfo& info) {
        return op::LeakyRelu::Build(info, mImpl);
    }

    Napi::Value GraphBuilder::Pad(const Napi::CallbackInfo& info) {
        return op::Pad::Build(info, mImpl);
    }

    Napi::Value GraphBuilder::Reshape(const Napi::CallbackInfo& info) {
        return op::Reshape::Build(info, mImpl);
    }

    Napi::Value GraphBuilder::Transpose(const Napi::CallbackInfo& info) {
        return op::Transpose::Build(info, mImpl);
    }

    Napi::Value GraphBuilder::Build(const Napi::CallbackInfo& info) {
        // MLGraph BuildSync(NamedOperands outputs);
        WEBNN_NODE_ASSERT(info.Length() == 1, "The number of arguments is invalid.");
        ml::NamedOperands namedOperands;
        std::vector<std::string> names;
        WEBNN_NODE_ASSERT(GetNamedOperands(info[0], namedOperands, names),
                          "The outputs parameter is invalid.");
        ml::Graph graph = mImpl.Build(namedOperands);
        WEBNN_NODE_ASSERT(graph != nullptr, "Failed to build graph.");
        Napi::Object object = node::Graph::constructor.New({});
        node::Graph* jsGraph = Napi::ObjectWrap<node::Graph>::Unwrap(object);
        jsGraph->mImpl = graph;
        jsGraph->mOutputNames = names;
        return object;
    }

    Napi::Object GraphBuilder::Initialize(Napi::Env env, Napi::Object exports) {
        Napi::HandleScope scope(env);
        Napi::Function func = DefineClass(
            env, "MLGraphBuilder",
            {InstanceMethod("constant", &GraphBuilder::Constant, napi_enumerable),
             InstanceMethod("input", &GraphBuilder::Input, napi_enumerable),
             InstanceMethod("add", &GraphBuilder::Add, napi_enumerable),
             InstanceMethod("batchNormalization", &GraphBuilder::BatchNorm, napi_enumerable),
             InstanceMethod("mul", &GraphBuilder::Mul, napi_enumerable),
             InstanceMethod("matmul", &GraphBuilder::Matmul, napi_enumerable),
             InstanceMethod("div", &GraphBuilder::Div, napi_enumerable),
             InstanceMethod("concat", &GraphBuilder::Concat, napi_enumerable),
             InstanceMethod("conv2d", &GraphBuilder::Conv2d, napi_enumerable),
             InstanceMethod("clamp", &GraphBuilder::Clamp, napi_enumerable),
             InstanceMethod("gemm", &GraphBuilder::Gemm, napi_enumerable),
             InstanceMethod("maxPool2d", &GraphBuilder::MaxPool2d, napi_enumerable),
             InstanceMethod("averagePool2d", &GraphBuilder::AveragePool2d, napi_enumerable),
             InstanceMethod("reduceMean", &GraphBuilder::ReduceMean, napi_enumerable),
             InstanceMethod("relu", &GraphBuilder::Relu, napi_enumerable),
             InstanceMethod("resample", &GraphBuilder::Resample, napi_enumerable),
             InstanceMethod("leakyRelu", &GraphBuilder::LeakyRelu, napi_enumerable),
             InstanceMethod("pad", &GraphBuilder::Pad, napi_enumerable),
             InstanceMethod("reshape", &GraphBuilder::Reshape, napi_enumerable),
             InstanceMethod("softmax", &GraphBuilder::Softmax, napi_enumerable),
             InstanceMethod("transpose", &GraphBuilder::Transpose, napi_enumerable),
             InstanceMethod("build", &GraphBuilder::Build, napi_enumerable)});
        constructor = Napi::Persistent(func);
        constructor.SuppressDestruct();
        exports.Set("MLGraphBuilder", func);
        return exports;
    }

}  // namespace node
