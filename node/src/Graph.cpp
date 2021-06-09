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
        void const* buffer = nullptr;
        size_t size;
        std::vector<int32_t> dimensions;

        const ml::Input* AsPtr() {
            mInput.buffer = buffer;
            mInput.size = size;
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
        size_t size;
        std::vector<int32_t> dimensions;

        const ml::Output* AsPtr() {
            mOutput.buffer = buffer;
            mOutput.size = size;
            if (!dimensions.empty()) {
                mOutput.dimensions = dimensions.data();
                mOutput.dimensionsCount = dimensions.size();
            }
            return &mOutput;
        }

      private:
        ml::Output mOutput;
    };

    // Hold Promise::Deferred with AsyncWorker.
    class ComputeGraphWorker : public Napi::AsyncWorker {
      public:
        ComputeGraphWorker(Napi::Env& env,
                           Napi::Promise::Deferred& deferred,
                           ml::Graph graph,
                           std::map<std::string, Input> inputs,
                           std::map<std::string, Output> outputs,
                           std::vector<std::string> outputNames)
            : Napi::AsyncWorker(env),
              mEnv(env),
              mDeferred(deferred),
              mGraph(graph),
              mInputs(std::move(inputs)),
              mOutputs(std::move(outputs)),
              mOutputNames(std::move(outputNames)) {
        }

        ~ComputeGraphWorker() = default;

        void Execute() {
            ml::NamedInputs namedInputs = ml::CreateNamedInputs();
            for (auto& input : mInputs) {
                namedInputs.Set(input.first.data(), input.second.AsPtr());
            }
            ml::NamedOutputs namedOutputs = mOutputs.empty() ? nullptr : ml::CreateNamedOutputs();
            for (auto& output : mOutputs) {
                namedOutputs.Set(output.first.data(), output.second.AsPtr());
            }
            mGraph.Compute(
                namedInputs,
                [](MLComputeGraphStatus status, MLNamedResults results, char const* message,
                   void* userdata) {
                    ComputeGraphWorker* computeWorker =
                        reinterpret_cast<ComputeGraphWorker*>(userdata);
                    computeWorker->SetResults(status, results, message);
                },
                reinterpret_cast<void*>(this), namedOutputs);
        }

        void OnOK() {
            if (mStatus != ml::ComputeGraphStatus::Success) {
                return mDeferred.Reject(Napi::Error::New(mEnv, mMessage).Value());
            }
            Napi::Object jsResults = Napi::Object::New(mEnv);
            for (auto& name : mOutputNames) {
                ml::Result* result = new ml::Result(mNamedResults.Get(name.data()));
                if (result->GetHandle() == nullptr) {
                    // specified outputs.
                    continue;
                }
                Napi::Object jsOutput = Napi::Object::New(mEnv);
                typedef void (*FinalizerCallback)(Napi::Env env, void* buffer, ml::Result* reslut);
                Napi::ArrayBuffer arrayBuffer =
                    Napi::ArrayBuffer::New<FinalizerCallback, ml::Result>(
                        mEnv, const_cast<void*>(result->Buffer()), result->BufferSize(),
                        [](Napi::Env env, void* buffer, ml::Result* reslut) { delete reslut; },
                        result);
                // FIXME: handle other data types
                Napi::Float32Array float32Array = Napi::Float32Array::New(
                    mEnv, result->BufferSize() / sizeof(float), arrayBuffer, 0);
                jsOutput.Set("data", float32Array);
                if (result->Dimensions()) {
                    Napi::Array jsDimensions = Napi::Array::New(mEnv, result->DimensionsSize());
                    for (size_t i = 0; i < result->DimensionsSize(); ++i) {
                        jsDimensions[i] = Napi::Number::New(mEnv, result->Dimensions()[i]);
                    }
                    jsOutput.Set("dimensions", jsDimensions);
                }
                jsResults.Set(name, jsOutput);
            }
            mDeferred.Resolve(jsResults);
        }

        void SetResults(MLComputeGraphStatus status, MLNamedResults results, char const* message) {
            mStatus = static_cast<ml::ComputeGraphStatus>(status);
            mNamedResults = mNamedResults.Acquire(results);
            if (message) {
                mMessage = std::string(message);
            }
        }

      private:
        Napi::Env mEnv;
        Napi::Promise::Deferred mDeferred;
        ml::Graph mGraph;
        ml::ComputeGraphStatus mStatus;
        std::string mMessage;
        std::map<std::string, Input> mInputs;
        std::map<std::string, Output> mOutputs;
        std::vector<std::string> mOutputNames;
        ml::NamedResults mNamedResults;
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
        for (size_t i = 0; i < names.Length(); ++i) {
            std::string name = names.Get(i).As<Napi::String>().Utf8Value();
            Napi::Object jsResource = jsResources.Get(name).As<Napi::Object>();
            // dictionary Input {
            //   required ArrayBufferView buffer;
            //   sequence<long> dimensions;
            // };
            // dictionary Output {
            //   ArrayBufferView buffer;
            //   sequence<long> dimensions;
            // };
            T resource = {};
            if (!jsResource.Has("data")) {
                // Input buffer is required.
                return false;
            }
            int jsElementLength = 0;
            if (jsResource.Has("data")) {
                if (!jsResource.Get("data").IsTypedArray()) {
                    return false;
                }

                // FIXME: validate the type of typed array.

                Napi::TypedArray jsTypedArray = jsResource.Get("data").As<Napi::TypedArray>();
                resource.buffer = reinterpret_cast<void*>(
                    reinterpret_cast<int8_t*>(jsTypedArray.ArrayBuffer().Data()) +
                    jsTypedArray.ByteOffset());
                resource.size = jsTypedArray.ByteLength();
                jsElementLength = jsTypedArray.ElementSize();
            }
            if (HasOptionMember(jsResource, "dimensions")) {
                if (!GetInt32Array(jsResource.Get("dimensions"), resource.dimensions)) {
                    return false;
                }

                /***Dimensions Check ***/
                if (jsElementLength) {
                    int dimensionSize = 1;
                    for (auto& dim : resource.dimensions) {
                        dimensionSize *= dim;
                    }
                    if (dimensionSize != jsElementLength) {
                        return false;
                    }
                }
            }
            namedResources[name] = resource;
        }
        return true;
    }

    Napi::FunctionReference Graph::constructor;

    Graph::Graph(const Napi::CallbackInfo& info) : Napi::ObjectWrap<Graph>(info) {
    }

    Napi::Value Graph::Compute(const Napi::CallbackInfo& info) {
        // Promise<NamedOutputs> compute(NamedInputs inputs, optional NamedOutputs outputs = {});
        WEBNN_NODE_ASSERT(info.Length() == 1 || info.Length() == 2,
                          "The number of arguments is invalid.");
        std::map<std::string, Input> inputs;
        WEBNN_NODE_ASSERT(GetNamedResources<Input>(info[0], inputs),
                          "The inputs parameter is invalid.");

        std::map<std::string, Output> outputs;
        if (info.Length() > 1) {
            WEBNN_NODE_ASSERT(GetNamedResources<Output>(info[1], outputs),
                              "The outputs parameter is invalid.");
        }
        Napi::Env env = info.Env();
        auto deferred = Napi::Promise::Deferred::New(env);
        ComputeGraphWorker* worker = new ComputeGraphWorker(env, deferred, mImpl, std::move(inputs),
                                                            std::move(outputs), mOutputNames);
        worker->Queue();
        return deferred.Promise();
    }

    Napi::Value Graph::ComputeSync(const Napi::CallbackInfo& info) {
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
        ml::ComputeGraphStatus status = mImpl.ComputeSync(namedInputs, namedOutputs);

        return Napi::Number::New(info.Env(), static_cast<uint32_t>(status));
    }

    Napi::Object Graph::Initialize(Napi::Env env, Napi::Object exports) {
        Napi::HandleScope scope(env);
        Napi::Function func =
            DefineClass(env, "MLGraph",
                        {InstanceMethod("compute", &Graph::Compute, napi_enumerable),
                         InstanceMethod("computeSync", &Graph::ComputeSync, napi_enumerable)});
        constructor = Napi::Persistent(func);
        constructor.SuppressDestruct();
        exports.Set("MLGraph", func);
        return exports;
    }

}  // namespace node
