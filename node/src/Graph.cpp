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

#include <iostream>
#include <map>

#include "Utils.h"

namespace node {

    struct Input {
      public:
        void* buffer = nullptr;
        size_t byteLength;
        std::vector<int32_t> dimensions;

        const ml::Input* AsPtr() {
            mInput.resource.buffer = buffer;
            mInput.resource.byteLength = byteLength;
            if (!dimensions.empty()) {
                mInput.dimensions = dimensions.data();
                mInput.dimensionsCount = dimensions.size();
            }
            return &mInput;
        }

      private:
        ml::Input mInput;
    };

    struct Output {
      public:
        void* buffer = nullptr;
        size_t byteLength;
        std::vector<int32_t> dimensions;

        const ml::ArrayBufferView* AsPtr() {
            mOutput.buffer = buffer;
            mOutput.byteLength = byteLength;
            return &mOutput;
        }

      private:
        ml::ArrayBufferView mOutput;
    };

    template <class T>
    bool GetNamedResources(const Napi::Value& jsValue, std::map<std::string, T>& namedResources) {
        if (!jsValue.IsObject()) {
            return false;
        }
        Napi::Object jsResources = jsValue.As<Napi::Object>();
        Napi::Array names = jsResources.GetPropertyNames();
        if (names.Length() == 0) {
            return false;
        }

        // typedef (MLBufferView or WebGLTexture or GPUTexture) MLResource;
        // dictionary MLInput {
        //   required MLResource resource;
        //   required sequence<long> dimensions;
        // };
        // typedef record<DOMString, (MLResource or MLInput)> MLNamedInputs;
        // typedef record<DOMString, MLResource> MLNamedOutputs;
        for (size_t i = 0; i < names.Length(); ++i) {
            T resource = {};
            std::string name = names.Get(i).As<Napi::String>().Utf8Value();
            // FIXME: validate the type of typed array.
            Napi::TypedArray jsTypedArray;
            if (jsResources.Get(name).IsTypedArray()) {
                jsTypedArray = jsResources.Get(name).As<Napi::TypedArray>();
            } else {
                Napi::Object jsResource = jsResources.Get(name).As<Napi::Object>();
                if (!jsResource.Has("resource") || !jsResource.Has("dimensions")) {
                    // Input buffer and dimensions are required.
                    return false;
                }
                if (!jsResource.Get("resource").IsTypedArray()) {
                    return false;
                }
                jsTypedArray = jsResource.Get("resource").As<Napi::TypedArray>();

                if (!GetInt32Array(jsResource.Get("dimensions"), resource.dimensions)) {
                    return false;
                }
                if (SizeOfShape(resource.dimensions) != jsTypedArray.ElementSize()) {
                    return false;
                }
            }
            resource.buffer = reinterpret_cast<void*>(
                reinterpret_cast<int8_t*>(jsTypedArray.ArrayBuffer().Data()) +
                jsTypedArray.ByteOffset());
            resource.byteLength = jsTypedArray.ByteLength();
            namedResources[name] = resource;
        }
        return true;
    }

    Napi::FunctionReference Graph::constructor;

    Graph::Graph(const Napi::CallbackInfo& info) : Napi::ObjectWrap<Graph>(info) {
    }

    Napi::Value Graph::Compute(const Napi::CallbackInfo& info) {
        // status compute(NamedInputs inputs, NamedOutputs outputs);
        WEBNN_NODE_ASSERT(info.Length() == 2, "The number of arguments is invalid.");
        std::map<std::string, Input> inputs;
        WEBNN_NODE_ASSERT(GetNamedResources<Input>(info[0], inputs),
                          "The inputs parameter is invalid.");

        std::map<std::string, Output> outputs;
        WEBNN_NODE_ASSERT(GetNamedResources<Output>(info[1], outputs),
                          "The outputs parameter is invalid.");

        ml::NamedInputs namedInputs = ml::CreateNamedInputs();
        for (auto& input : inputs) {
            namedInputs.Set(input.first.data(), input.second.AsPtr());
        }
        ml::NamedOutputs namedOutputs = ml::CreateNamedOutputs();
        for (auto& output : outputs) {
            namedOutputs.Set(output.first.data(), output.second.AsPtr());
        }
        ml::ComputeGraphStatus status = mImpl.Compute(namedInputs, namedOutputs);

        return Napi::Number::New(info.Env(), static_cast<uint32_t>(status));
    }

    Napi::Object Graph::Initialize(Napi::Env env, Napi::Object exports) {
        Napi::HandleScope scope(env);
        Napi::Function func = DefineClass(
            env, "MLGraph", {InstanceMethod("compute", &Graph::Compute, napi_enumerable)});
        constructor = Napi::Persistent(func);
        constructor.SuppressDestruct();
        exports.Set("MLGraph", func);
        return exports;
    }

}  // namespace node
