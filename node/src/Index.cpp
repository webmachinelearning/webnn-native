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

#include "Index.h"

#include "Context.h"
#include "Graph.h"
#include "GraphBuilder.h"
#include "ML.h"
#include "Operand.h"
#include "Operator.h"

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    node::ML::Initialize(env, exports);
    node::Context::Initialize(env, exports);
    node::GraphBuilder::Initialize(env, exports);
    node::Graph::Initialize(env, exports);
    node::Operand::Initialize(env, exports);
    node::Operator::Initialize(env, exports);

    return exports;
}

NODE_API_MODULE(addon, Init)
