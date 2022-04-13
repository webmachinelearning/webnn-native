// Copyright 2019 The Webnn Authors
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

#ifndef WEBNN_WIRE_SERVER_SERVER_H_
#define WEBNN_WIRE_SERVER_SERVER_H_

#include "webnn_wire/ChunkedCommandSerializer.h"
#include "webnn_wire/server/ServerBase_autogen.h"

#include <string>

#if defined(WEBNN_ENABLE_GPU_BUFFER)
#    include <dawn/wire/WireServer.h>
#    include <webgpu/webgpu.h>
#endif

namespace webnn_wire::server {

    // CallbackUserdata and its derived classes are intended to be created by
    // Server::MakeUserdata<T> and then passed as the userdata argument for Dawn
    // callbacks.
    // It contains a pointer back to the Server so that the callback can call the
    // Server to perform operations like serialization, and it contains a weak pointer
    // |serverIsAlive|. If the weak pointer has expired, it means the server has
    // been destroyed and the callback must not use the Server pointer.
    // To assist with checking |serverIsAlive| and lifetime management of the userdata,
    // |ForwardToServer| (defined later in this file) can be used to acquire the userdata,
    // return early if |serverIsAlive| has expired, and then forward the arguments
    // to userdata->server->MyCallbackHandler.
    //
    // Example Usage:
    //
    // struct MyUserdata : CallbackUserdata { uint32_t foo; };
    //
    // auto userdata = MakeUserdata<MyUserdata>();
    // userdata->foo = 2;
    //
    // callMyCallbackHandler(
    //      ForwardToServer<&Server::MyCallbackHandler>;
    //      userdata.release());
    //
    // void Server::MyCallbackHandler(MyUserdata* userdata) { }
    struct CallbackUserdata {
        Server* const server;
        std::weak_ptr<bool> const serverIsAlive;

        CallbackUserdata() = delete;
        CallbackUserdata(Server* server, const std::shared_ptr<bool>& serverIsAlive)
            : server(server), serverIsAlive(serverIsAlive) {
        }
    };

    template <auto F>
    struct ForwardToServerHelper {
        template <typename _>
        struct ExtractedTypes;

        // An internal structure used to unpack the various types that compose the type of F
        template <typename Return, typename Class, typename Userdata, typename... Args>
        struct ExtractedTypes<Return (Class::*)(Userdata*, Args...)> {
            using UntypedCallback = Return (*)(Args..., void*);
            static Return Callback(Args... args, void* userdata) {

                // Acquire the userdata, and cast it to UserdataT.
                std::unique_ptr<Userdata> data(static_cast<Userdata*>(userdata));
                if (data->serverIsAlive.expired()) {
                   // Do nothing if the server has already been destroyed.
                  return;
                }
                // Forward the arguments and the typed userdata to the Server:: member function.
                (data->server->*F)(data.get(), std::forward<decltype(args)>(args)...);
            }
        };

        static constexpr typename ExtractedTypes<decltype(F)>::UntypedCallback Create() {
            return ExtractedTypes<decltype(F)>::Callback;
        }
    };

    template <auto F>
    constexpr auto ForwardToServer = ForwardToServerHelper<F>::Create();

    struct ErrorScopeUserdata : CallbackUserdata {
        using CallbackUserdata::CallbackUserdata;

        ObjectHandle context;
        uint64_t requestSerial;
    };

    struct ComputeAsyncUserdata : CallbackUserdata {
        using CallbackUserdata::CallbackUserdata;

        ObjectHandle graph;
        uint64_t requestSerial;
        ObjectId namedOutputsObjectID;
    };

    class Server : public ServerBase {
      public:
        Server(const WebnnProcTable& procs, CommandSerializer* serializer);
        ~Server() override;

        // ChunkedCommandHandler implementation
        const volatile char* HandleCommandsImpl(const volatile char* commands,
                                                size_t size) override;

        bool InjectInstance(WNNInstance instance, uint32_t id, uint32_t generation);
#if defined(WEBNN_ENABLE_GPU_BUFFER)
        bool InjectDawnWireServer(dawn_wire::WireServer* dawn_wire_server);
#endif
        bool InjectContext(WNNContext context, uint32_t id, uint32_t generation);
        bool InjectNamedInputs(WNNNamedInputs namedInputs,
                               uint32_t id,
                               uint32_t generation,
                               uint32_t contextId,
                               uint32_t contextGeneration);
        bool InjectNamedOperands(WNNNamedOperands namedOperands, uint32_t id, uint32_t generation);
        bool InjectNamedOutputs(WNNNamedOutputs namedOutputs, uint32_t id, uint32_t generation);

        template <typename T,
                  typename Enable = std::enable_if<std::is_base_of<CallbackUserdata, T>::value>>
        std::unique_ptr<T> MakeUserdata() {
            return std::unique_ptr<T>(new T(this, mIsAlive));
        }

      private:
        template <typename Cmd>
        void SerializeCommand(const Cmd& cmd) {
            mSerializer.SerializeCommand(cmd);
        }

        template <typename Cmd, typename ExtraSizeSerializeFn>
        void SerializeCommand(const Cmd& cmd,
                              size_t extraSize,
                              ExtraSizeSerializeFn&& SerializeExtraSize) {
            mSerializer.SerializeCommand(cmd, extraSize, SerializeExtraSize);
        }

        void ClearContextCallbacks(WNNContext context);

#if defined(WEBNN_ENABLE_GPU_BUFFER)
        WGPUDevice GetWGPUDevice(uint32_t id, uint32_t generation);
        WGPUBuffer GetWGPUBuffer(uint32_t id, uint32_t generation);
#endif
        // Error callbacks
        void OnUncapturedError(WNNErrorType type, const char* message);
        void OnContextLost(const char* message);
        void OnContextPopErrorScope(ErrorScopeUserdata* userdata,
                                    WNNErrorType type,
                                    const char* message);
        void OnGraphComputeAsyncCallback(ComputeAsyncUserdata* userdata,
                                         WNNComputeGraphStatus status,
                                         const char* message);
#include "webnn_wire/server/ServerPrototypes_autogen.inc"

        WireDeserializeAllocator mAllocator;
        ChunkedCommandSerializer mSerializer;
        WebnnProcTable mProcs;

#if defined(WEBNN_ENABLE_GPU_BUFFER)
        dawn::wire::WireServer* mDawnWireServer;
#endif
        // Save the output names in server because char** type isn't supported in webnn.json to get
        // name.
        std::map<ObjectId, std::vector<std::string>> mOutputNamesMap;
        bool SerializeComputeResult(ObjectId outputsId);

        std::shared_ptr<bool> mIsAlive;
    };

    bool TrackContextChild(ContextInfo* context, ObjectType type, ObjectId id);
    bool UntrackContextChild(ContextInfo* context, ObjectType type, ObjectId id);

}  // namespace webnn_wire::server

#endif  // WEBNN_WIRE_SERVER_SERVER_H_
