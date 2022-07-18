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

#include "Graph.h"
#include "Utils.h"

namespace node {

    Graph::Graph(const Napi::CallbackInfo& info) : Napi::ObjectWrap<Graph>(info) {
    }

    wnn::Graph Graph::GetImpl() {
        return mImpl;
    }

    Napi::FunctionReference Graph::constructor;

    Napi::Object Graph::Initialize(Napi::Env env, Napi::Object exports) {
        Napi::HandleScope scope(env);
        Napi::Function func = DefineClass(env, "MLGraph", {});
        constructor = Napi::Persistent(func);
        constructor.SuppressDestruct();
        exports.Set("MLGraph", func);
        return exports;
    }

}  // namespace node
