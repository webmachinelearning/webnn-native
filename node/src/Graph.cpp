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
        wnn::ArrayBufferView bufferView;
        std::vector<int32_t> dimensions;

        const wnn::Input* AsPtr() {
            mInput.resource.arrayBufferView = bufferView;
            mInput.resource.gpuBufferView = {};
            if (!dimensions.empty()) {
                mInput.dimensions = dimensions.data();
                mInput.dimensionsCount = dimensions.size();
            }
            return &mInput;
        }

      private:
        wnn::Input mInput;
    };

    bool GetNamedInputs(const Napi::Value& jsValue, std::map<std::string, Input>& namedInputs) {
        if (!jsValue.IsObject()) {
            return false;
        }
        Napi::Object jsNamedInputs = jsValue.As<Napi::Object>();
        Napi::Array names = jsNamedInputs.GetPropertyNames();
        if (names.Length() == 0) {
            return false;
        }
        // typedef (MLBufferView or WebGLTexture or GPUTexture) MLResource;
        // dictionary MLInput {
        //   required MLResource resource;
        //   required sequence<long> dimensions;
        // };
        // typedef record<DOMString, (MLResource or MLInput)> MLNamedInputs;
        for (size_t i = 0; i < names.Length(); ++i) {
            Input input = {};
            std::string name = names.Get(i).As<Napi::String>().Utf8Value();
            // FIXME: validate the type of typed array.
            Napi::TypedArray jsTypedArray;
            if (jsNamedInputs.Get(name).IsTypedArray()) {
                jsTypedArray = jsNamedInputs.Get(name).As<Napi::TypedArray>();
            } else {
                Napi::Object jsInput = jsNamedInputs.Get(name).As<Napi::Object>();
                if (!jsInput.Has("resource") || !jsInput.Has("dimensions")) {
                    // Input resource and dimensions are required.
                    return false;
                }
                if (!jsInput.Get("resource").IsTypedArray()) {
                    return false;
                }
                jsTypedArray = jsInput.Get("resource").As<Napi::TypedArray>();

                if (!GetArray(jsInput.Get("dimensions"), input.dimensions)) {
                    return false;
                }
                if (SizeOfShape(input.dimensions) != jsTypedArray.ElementSize()) {
                    return false;
                }
            }
            if (!GetArrayBufferView(jsTypedArray, input.bufferView)) {
                return false;
            }
            namedInputs[name] = input;
        }
        return true;
    }

    bool GetNamedOutputs(const Napi::Value& jsValue,
                         std::map<std::string, wnn::Resource>& namedOutputs) {
        if (!jsValue.IsObject()) {
            return false;
        }
        Napi::Object jsNamedOutputs = jsValue.As<Napi::Object>();
        Napi::Array names = jsNamedOutputs.GetPropertyNames();
        if (names.Length() == 0) {
            return false;
        }
        // typedef (MLBufferView or WebGLTexture or GPUTexture) MLResource;
        // typedef record<DOMString, MLResource> MLNamedOutputs;
        for (size_t i = 0; i < names.Length(); ++i) {
            wnn::ArrayBufferView arrayBuffer = {};
            std::string name = names.Get(i).As<Napi::String>().Utf8Value();
            if (!GetArrayBufferView(jsNamedOutputs.Get(name), arrayBuffer)) {
                return false;
            }
            namedOutputs[name] = {arrayBuffer, {}};
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
        WEBNN_NODE_ASSERT(GetNamedInputs(info[0], inputs), "The inputs parameter is invalid.");

        std::map<std::string, wnn::Resource> outputs;
        WEBNN_NODE_ASSERT(GetNamedOutputs(info[1], outputs), "The outputs parameter is invalid.");

        wnn::NamedInputs namedInputs = wnn::CreateNamedInputs();
        for (auto& input : inputs) {
            namedInputs.Set(input.first.data(), input.second.AsPtr());
        }
        wnn::NamedOutputs namedOutputs = wnn::CreateNamedOutputs();
        for (auto& output : outputs) {
            namedOutputs.Set(output.first.data(), &output.second);
        }
        wnn::ComputeGraphStatus status = mImpl.Compute(namedInputs, namedOutputs);

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
