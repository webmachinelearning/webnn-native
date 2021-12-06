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
#include "Operator.h"
#include "Utils.h"
#include "ops/BatchNorm.h"
#include "ops/Clamp.h"
#include "ops/Concat.h"
#include "ops/Constant.h"
#include "ops/Conv2d.h"
#include "ops/Gemm.h"
#include "ops/Gru.h"
#include "ops/Input.h"
#include "ops/InstanceNorm.h"
#include "ops/LeakyRelu.h"
#include "ops/Pad.h"
#include "ops/Pool2d.h"
#include "ops/Reduce.h"
#include "ops/Resample2d.h"
#include "ops/Reshape.h"
#include "ops/Slice.h"
#include "ops/Split.h"
#include "ops/Squeeze.h"
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

#define BUILD_UNARY_OPERAND(op)                                                             \
    WEBNN_NODE_ASSERT(info.Length() == 1, "The number of arguments is invalid.");           \
    std::vector<napi_value> args;                                                           \
    ml::Operand input;                                                                      \
    WEBNN_NODE_ASSERT(GetOperand(info[0], input, args), "The input parameter is invalid."); \
    Napi::Object object = Operand::constructor.New(args);                                   \
    Operand* operand = Napi::ObjectWrap<Operand>::Unwrap(object);                           \
    operand->SetImpl(mImpl.op(input));                                                      \
    return object;

#define BUILD_UNARY_OPERATOR(op)                                                  \
    WEBNN_NODE_ASSERT(info.Length() == 0, "The number of arguments is invalid."); \
    Napi::Object object = Operator::constructor.New({});                          \
    Operator* mlOperator = Napi::ObjectWrap<Operator>::Unwrap({object});          \
    mlOperator->SetImpl(mImpl.op##Operator());                                    \
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

    Napi::Value GraphBuilder::Sub(const Napi::CallbackInfo& info) {
        BUILD_BINARY(Sub);
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

    Napi::Value GraphBuilder::Max(const Napi::CallbackInfo& info) {
        BUILD_BINARY(Max);
    }

    Napi::Value GraphBuilder::Min(const Napi::CallbackInfo& info) {
        BUILD_BINARY(Min);
    }

    Napi::Value GraphBuilder::Pow(const Napi::CallbackInfo& info) {
        BUILD_BINARY(Pow);
    }

    Napi::Value GraphBuilder::BatchNorm(const Napi::CallbackInfo& info) {
        return op::BatchNorm::Build(info, mImpl);
    }

    Napi::Value GraphBuilder::InstanceNorm(const Napi::CallbackInfo& info) {
        return op::InstanceNorm::Build(info, mImpl);
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

    Napi::Value GraphBuilder::Gru(const Napi::CallbackInfo& info) {
        return op::Gru::Build(info, mImpl);
    }

    Napi::Value GraphBuilder::Abs(const Napi::CallbackInfo& info) {
        BUILD_UNARY_OPERAND(Abs);
    }

    Napi::Value GraphBuilder::Ceil(const Napi::CallbackInfo& info) {
        BUILD_UNARY_OPERAND(Ceil);
    }

    Napi::Value GraphBuilder::Cos(const Napi::CallbackInfo& info) {
        BUILD_UNARY_OPERAND(Cos);
    }

    Napi::Value GraphBuilder::Exp(const Napi::CallbackInfo& info) {
        BUILD_UNARY_OPERAND(Exp);
    }

    Napi::Value GraphBuilder::Floor(const Napi::CallbackInfo& info) {
        BUILD_UNARY_OPERAND(Floor);
    }

    Napi::Value GraphBuilder::Log(const Napi::CallbackInfo& info) {
        BUILD_UNARY_OPERAND(Log);
    }

    Napi::Value GraphBuilder::Neg(const Napi::CallbackInfo& info) {
        BUILD_UNARY_OPERAND(Neg);
    }

    Napi::Value GraphBuilder::Sin(const Napi::CallbackInfo& info) {
        BUILD_UNARY_OPERAND(Sin);
    }

    Napi::Value GraphBuilder::Tan(const Napi::CallbackInfo& info) {
        BUILD_UNARY_OPERAND(Tan);
    }

    Napi::Value GraphBuilder::HardSwish(const Napi::CallbackInfo& info) {
        if (info.Length() == 0) {
            BUILD_UNARY_OPERATOR(HardSwish);
        } else {
            BUILD_UNARY_OPERAND(HardSwish);
        }
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

    Napi::Value GraphBuilder::ReduceL1(const Napi::CallbackInfo& info) {
        return op::Reduce::Build(op::ReduceType::kReduceL1, info, mImpl);
    }

    Napi::Value GraphBuilder::ReduceL2(const Napi::CallbackInfo& info) {
        return op::Reduce::Build(op::ReduceType::kReduceL2, info, mImpl);
    }

    Napi::Value GraphBuilder::ReduceMax(const Napi::CallbackInfo& info) {
        return op::Reduce::Build(op::ReduceType::kReduceMax, info, mImpl);
    }

    Napi::Value GraphBuilder::ReduceMean(const Napi::CallbackInfo& info) {
        return op::Reduce::Build(op::ReduceType::kReduceMean, info, mImpl);
    }

    Napi::Value GraphBuilder::ReduceMin(const Napi::CallbackInfo& info) {
        return op::Reduce::Build(op::ReduceType::kReduceMin, info, mImpl);
    }

    Napi::Value GraphBuilder::ReduceProduct(const Napi::CallbackInfo& info) {
        return op::Reduce::Build(op::ReduceType::kReduceProduct, info, mImpl);
    }

    Napi::Value GraphBuilder::ReduceSum(const Napi::CallbackInfo& info) {
        return op::Reduce::Build(op::ReduceType::kReduceSum, info, mImpl);
    }

    Napi::Value GraphBuilder::Resample2d(const Napi::CallbackInfo& info) {
        return op::Resample2d::Build(info, mImpl);
    }

    Napi::Value GraphBuilder::Relu(const Napi::CallbackInfo& info) {
        if (info.Length() == 0) {
            BUILD_UNARY_OPERATOR(Relu);
        } else {
            BUILD_UNARY_OPERAND(Relu);
        }
    }

    Napi::Value GraphBuilder::Softmax(const Napi::CallbackInfo& info) {
        BUILD_UNARY_OPERAND(Softmax);
    }

    Napi::Value GraphBuilder::Sigmoid(const Napi::CallbackInfo& info) {
        if (info.Length() == 0) {
            BUILD_UNARY_OPERATOR(Sigmoid);
        } else {
            BUILD_UNARY_OPERAND(Sigmoid);
        };
    }

    Napi::Value GraphBuilder::Slice(const Napi::CallbackInfo& info) {
        return op::Slice::Build(info, mImpl);
    }

    Napi::Value GraphBuilder::Split(const Napi::CallbackInfo& info) {
        return op::Split::Build(info, mImpl);
    }

    Napi::Value GraphBuilder::Squeeze(const Napi::CallbackInfo& info) {
        return op::Squeeze::Build(info, mImpl);
    }

    Napi::Value GraphBuilder::Tanh(const Napi::CallbackInfo& info) {
        if (info.Length() == 0) {
            BUILD_UNARY_OPERATOR(Tanh);
        } else {
            BUILD_UNARY_OPERAND(Tanh);
        };
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
             InstanceMethod("sub", &GraphBuilder::Sub, napi_enumerable),
             InstanceMethod("batchNormalization", &GraphBuilder::BatchNorm, napi_enumerable),
             InstanceMethod("instanceNormalization", &GraphBuilder::InstanceNorm, napi_enumerable),
             InstanceMethod("mul", &GraphBuilder::Mul, napi_enumerable),
             InstanceMethod("matmul", &GraphBuilder::Matmul, napi_enumerable),
             InstanceMethod("div", &GraphBuilder::Div, napi_enumerable),
             InstanceMethod("max", &GraphBuilder::Max, napi_enumerable),
             InstanceMethod("min", &GraphBuilder::Min, napi_enumerable),
             InstanceMethod("pow", &GraphBuilder::Pow, napi_enumerable),
             InstanceMethod("concat", &GraphBuilder::Concat, napi_enumerable),
             InstanceMethod("conv2d", &GraphBuilder::Conv2d, napi_enumerable),
             InstanceMethod("clamp", &GraphBuilder::Clamp, napi_enumerable),
             InstanceMethod("gemm", &GraphBuilder::Gemm, napi_enumerable),
             InstanceMethod("gru", &GraphBuilder::Gru, napi_enumerable),
             InstanceMethod("abs", &GraphBuilder::Abs, napi_enumerable),
             InstanceMethod("ceil", &GraphBuilder::Ceil, napi_enumerable),
             InstanceMethod("cos", &GraphBuilder::Cos, napi_enumerable),
             InstanceMethod("exp", &GraphBuilder::Exp, napi_enumerable),
             InstanceMethod("floor", &GraphBuilder::Floor, napi_enumerable),
             InstanceMethod("log", &GraphBuilder::Log, napi_enumerable),
             InstanceMethod("neg", &GraphBuilder::Neg, napi_enumerable),
             InstanceMethod("sin", &GraphBuilder::Sin, napi_enumerable),
             InstanceMethod("tan", &GraphBuilder::Tan, napi_enumerable),
             InstanceMethod("hardSwish", &GraphBuilder::HardSwish, napi_enumerable),
             InstanceMethod("maxPool2d", &GraphBuilder::MaxPool2d, napi_enumerable),
             InstanceMethod("averagePool2d", &GraphBuilder::AveragePool2d, napi_enumerable),
             InstanceMethod("reduceL1", &GraphBuilder::ReduceL1, napi_enumerable),
             InstanceMethod("reduceL2", &GraphBuilder::ReduceL2, napi_enumerable),
             InstanceMethod("reduceMax", &GraphBuilder::ReduceMax, napi_enumerable),
             InstanceMethod("reduceMean", &GraphBuilder::ReduceMean, napi_enumerable),
             InstanceMethod("reduceMin", &GraphBuilder::ReduceMin, napi_enumerable),
             InstanceMethod("reduceProduct", &GraphBuilder::ReduceProduct, napi_enumerable),
             InstanceMethod("reduceSum", &GraphBuilder::ReduceSum, napi_enumerable),
             InstanceMethod("relu", &GraphBuilder::Relu, napi_enumerable),
             InstanceMethod("resample2d", &GraphBuilder::Resample2d, napi_enumerable),
             InstanceMethod("leakyRelu", &GraphBuilder::LeakyRelu, napi_enumerable),
             InstanceMethod("pad", &GraphBuilder::Pad, napi_enumerable),
             InstanceMethod("reshape", &GraphBuilder::Reshape, napi_enumerable),
             InstanceMethod("softmax", &GraphBuilder::Softmax, napi_enumerable),
             InstanceMethod("sigmoid", &GraphBuilder::Sigmoid, napi_enumerable),
             InstanceMethod("slice", &GraphBuilder::Slice, napi_enumerable),
             InstanceMethod("split", &GraphBuilder::Split, napi_enumerable),
             InstanceMethod("squeeze", &GraphBuilder::Squeeze, napi_enumerable),
             InstanceMethod("tanh", &GraphBuilder::Tanh, napi_enumerable),
             InstanceMethod("transpose", &GraphBuilder::Transpose, napi_enumerable),
             InstanceMethod("build", &GraphBuilder::Build, napi_enumerable)});
        constructor = Napi::Persistent(func);
        constructor.SuppressDestruct();
        exports.Set("MLGraphBuilder", func);
        return exports;
    }

}  // namespace node
