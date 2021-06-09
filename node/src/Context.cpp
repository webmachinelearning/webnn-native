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

#include <webnn/webnn_proc.h>
#include <webnn_native/WebnnNative.h>
#include <iostream>


Napi::FunctionReference node::Context::constructor;

namespace node {

    Context::Context(const Napi::CallbackInfo& info) : Napi::ObjectWrap<Context>(info) {
        WebnnProcTable backendProcs = webnn_native::GetProcs();
        webnnProcSetProcs(&backendProcs);
        mImpl = ml::Context::Acquire(webnn_native::CreateContext());
        if (!mImpl) {
            Napi::Error::New(info.Env(), "Failed to create Context").ThrowAsJavaScriptException();
            return;
        }
        mImpl.SetUncapturedErrorCallback(
            [](MLErrorType type, char const* message, void* userData) {
                if (type != MLErrorType_NoError) {
                    std::cout << "Uncaptured Error type is " << type << ", message is " << message
                              << std::endl;
                }
            },
            this);
    }

    ml::Context Context::GetImpl() {
        return mImpl;
    }

    Napi::Object Context::Initialize(Napi::Env env, Napi::Object exports) {
        Napi::HandleScope scope(env);
        Napi::Function func = DefineClass(env, "MLContext", {});
        constructor = Napi::Persistent(func);
        constructor.SuppressDestruct();
        exports.Set("MLContext", func);
        return exports;
    }
}  // namespace node
