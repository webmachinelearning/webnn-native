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

#ifndef NODE_GRAPH_H_
#define NODE_GRAPH_H_

#include <napi.h>
#include <webnn/webnn_cpp.h>
#include <string>

namespace node {

    class BuildGraphWorker;
    class GraphBuilder;

    class Graph : public Napi::ObjectWrap<Graph> {
      public:
        static Napi::Object Initialize(Napi::Env env, Napi::Object exports);
        static Napi::FunctionReference constructor;

        explicit Graph(const Napi::CallbackInfo& info);
        ~Graph() = default;

      private:
        friend BuildGraphWorker;
        friend GraphBuilder;

        Napi::Value Compute(const Napi::CallbackInfo& info);

        ml::Graph mImpl;
        std::vector<std::string> mOutputNames;
    };

}  // namespace node

#endif  // NODE_GRAPH_H_
