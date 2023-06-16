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

#include "Context.h"

#include <napi.h>
#include <iostream>

#include "Graph.h"
#include "ML.h"
#include "Utils.h"

Napi::FunctionReference node::Context::constructor;

namespace node {

    Context::Context(const Napi::CallbackInfo& info) : Napi::ObjectWrap<Context>(info) {
        wnn::ContextOptions options = {wnn::DevicePreference::Default,
                                       wnn::PowerPreference::Default};
        if (info.Length() > 0) {
            Napi::Object optionsObject = info[0].As<Napi::Object>();
            if (optionsObject.Has("powerPreference")) {
                if (!optionsObject.Get("powerPreference").IsString()) {
                    Napi::Error::New(info.Env(), "Invaild powerPreference")
                        .ThrowAsJavaScriptException();
                    return;
                }
                std::string powerPreference = optionsObject.Get("powerPreference").ToString();
                if (powerPreference == "default") {
                    options.powerPreference = wnn::PowerPreference::Default;
                } else if (powerPreference == "low-power") {
                    options.powerPreference = wnn::PowerPreference::Low_power;
                } else if (powerPreference == "high-performance") {
                    options.powerPreference = wnn::PowerPreference::High_performance;
                } else {
                    Napi::Error::New(info.Env(), "Invaild powerPreference")
                        .ThrowAsJavaScriptException();
                    return;
                }
            }

            if (optionsObject.Has("devicePreference")) {
                if (!optionsObject.Get("devicePreference").IsString()) {
                    Napi::Error::New(info.Env(), "Invaild devicePreference")
                        .ThrowAsJavaScriptException();
                    return;
                }
                std::string devicePreference = optionsObject.Get("devicePreference").ToString();
                if (devicePreference == "default") {
                    options.devicePreference = wnn::DevicePreference::Default;
                } else if (devicePreference == "gpu") {
                    options.devicePreference = wnn::DevicePreference::Gpu;
                } else if (devicePreference == "cpu") {
                    options.devicePreference = wnn::DevicePreference::Cpu;
                } else {
                    Napi::Error::New(info.Env(), "Invaild devicePreference")
                        .ThrowAsJavaScriptException();
                    return;
                }
            }
        }

        mImpl = wnn::Context::Acquire(ML::GetInstance()->CreateContext(&options));
        if (!mImpl) {
            Napi::Error::New(info.Env(), "Failed to create Context").ThrowAsJavaScriptException();
            return;
        }
        mImpl.SetUncapturedErrorCallback(
            [](WNNErrorType type, char const* message, void* userData) {
                if (type != WNNErrorType_NoError) {
                    std::cout << "Uncaptured Error type is " << type << ", message is " << message
                              << std::endl;
                }
            },
            this);
    }

    wnn::Context Context::GetImpl() {
        return mImpl;
    }

    Napi::Object Context::Initialize(Napi::Env env, Napi::Object exports) {
        Napi::HandleScope scope(env);
        Napi::Function func = DefineClass(
            env, "MLContext", {InstanceMethod("compute", &Context::Compute, napi_enumerable)});
        constructor = Napi::Persistent(func);
        constructor.SuppressDestruct();
        exports.Set("MLContext", func);
        return exports;
    }

    Napi::Value Context::Compute(const Napi::CallbackInfo& info) {
        // status compute(NamedInputs inputs, NamedOutputs outputs);
        WEBNN_NODE_ASSERT(info.Length() == 3, "The number of arguments is invalid.");
        Napi::Object object = info[0].As<Napi::Object>();
        node::Graph* jsGraph = Napi::ObjectWrap<node::Graph>::Unwrap(object);

        std::map<std::string, Input> inputs;
        WEBNN_NODE_ASSERT(GetNamedInputs(info[1], inputs), "The inputs parameter is invalid.");

        std::map<std::string, wnn::Resource> outputs;
        WEBNN_NODE_ASSERT(GetNamedOutputs(info[2], outputs), "The outputs parameter is invalid.");

        wnn::NamedInputs namedInputs = wnn::CreateNamedInputs();
        for (auto& input : inputs) {
            namedInputs.Set(input.first.data(), input.second.AsPtr());
        }
        wnn::NamedOutputs namedOutputs = wnn::CreateNamedOutputs();
        for (auto& output : outputs) {
            namedOutputs.Set(output.first.data(), &output.second);
        }
        mImpl.ComputeSync(jsGraph->GetImpl(), namedInputs, namedOutputs);

        return Napi::Number::New(info.Env(), 0);
    }

}  // namespace node
