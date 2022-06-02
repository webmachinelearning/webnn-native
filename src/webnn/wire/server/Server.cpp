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

#include "webnn/wire/server/Server.h"
#include "webnn/wire/WireServer.h"

namespace webnn::wire::server {

    Server::Server(const WebnnProcTable& procs, CommandSerializer* serializer)
        : mSerializer(serializer), mProcs(procs), mIsAlive(std::make_shared<bool>(true)) {
    }

    Server::~Server() {
        // Un-set the error and lost callbacks since we cannot forward them
        // after the server has been destroyed.
        for (WNNContext context : ContextObjects().GetAllHandles()) {
            ClearContextCallbacks(context);
        }
        DestroyAllObjects(mProcs);
    }

    void Server::ClearContextCallbacks(WNNContext context) {
        // Un-set the error and lost callbacks since we cannot forward them
        // after the server has been destroyed.
        mProcs.contextSetUncapturedErrorCallback(context, nullptr, nullptr);
        // mProcs.contextSetContextLostCallback(context, nullptr, nullptr);
    }

    bool Server::InjectInstance(WNNInstance instance, uint32_t id, uint32_t generation) {
        ASSERT(instance != nullptr);
        ObjectData<WNNInstance>* data = InstanceObjects().Allocate(id);
        if (data == nullptr) {
            return false;
        }

        data->handle = instance;
        data->generation = generation;
        data->state = AllocationState::Allocated;
        mProcs.instanceReference(instance);

        return true;
    }

#if defined(WEBNN_ENABLE_GPU_BUFFER)
    bool Server::InjectDawnWireServer(dawn_wire::WireServer* dawn_wire_server) {
        mDawnWireServer = dawn_wire_server;
        return true;
    }
#endif

    bool Server::InjectContext(WNNContext context, uint32_t id, uint32_t generation) {
        ASSERT(context != nullptr);
        ObjectData<WNNContext>* data = ContextObjects().Allocate(id);
        if (data == nullptr) {
            return false;
        }

        data->handle = context;
        data->generation = generation;
        data->state = AllocationState::Allocated;
        data->info->server = this;
        data->info->self = ObjectHandle{id, generation};

        // The context is externally owned so it shouldn't be destroyed when we receive a destroy
        // message from the client. Add a reference to counterbalance the eventual release.
        mProcs.contextReference(context);

        // Set callbacks to forward errors to the client.
        // Note: these callbacks are manually inlined here since they do not acquire and
        // free their userdata. Also unlike other callbacks, these are cleared and unset when
        // the server is destroyed, so we don't need to check if the server is still alive
        // inside them.
        // mProcs.contextSetUncapturedErrorCallback(
        //     context,
        //     [](WNNErrorType type, const char* message, void* userdata) {
        //         ContextInfo* info = static_cast<ContextInfo*>(userdata);
        //         info->server->OnUncapturedError(info->self, type, message);
        //     },
        //     data->info.get());

        return true;
    }

    bool Server::InjectNamedInputs(WNNNamedInputs namedInputs,
                                   uint32_t id,
                                   uint32_t generation,
                                   uint32_t contextId,
                                   uint32_t contextGeneration) {
        ASSERT(namedInputs != nullptr);
        // ObjectData<WNNContext>* context = ContextObjects().Get(contextId);
        // if (context == nullptr || context->generation != contextGeneration) {
        //     return false;
        // }
        ObjectData<WNNNamedInputs>* data = NamedInputsObjects().Allocate(id);
        if (data == nullptr) {
            return false;
        }

        data->handle = namedInputs;
        data->generation = generation;
        data->state = AllocationState::Allocated;
        // data->contextInfo = context->info.get();

        // if (!TrackContextChild(data->contextInfo, ObjectType::NamedInputs, id)) {
        //     return false;
        // }

        // The context is externally owned so it shouldn't be destroyed when we receive a destroy
        // message from the client. Add a reference to counterbalance the eventual release.
        mProcs.namedInputsReference(namedInputs);

        return true;
    }

    bool Server::InjectNamedOperands(WNNNamedOperands namedOperands,
                                     uint32_t id,
                                     uint32_t generation) {
        ASSERT(namedOperands != nullptr);
        ObjectData<WNNNamedOperands>* data = NamedOperandsObjects().Allocate(id);
        if (data == nullptr) {
            return false;
        }

        data->handle = namedOperands;
        data->generation = generation;
        data->state = AllocationState::Allocated;
        mProcs.namedOperandsReference(namedOperands);

        return true;
    }

    bool Server::InjectNamedOutputs(WNNNamedOutputs namedOutputs,
                                    uint32_t id,
                                    uint32_t generation) {
        ASSERT(namedOutputs != nullptr);
        ObjectData<WNNNamedOutputs>* data = NamedOutputsObjects().Allocate(id);
        if (data == nullptr) {
            return false;
        }

        data->handle = namedOutputs;
        data->generation = generation;
        data->state = AllocationState::Allocated;
        mProcs.namedOutputsReference(namedOutputs);

        return true;
    }

#if defined(WEBNN_ENABLE_GPU_BUFFER)
    WGPUDevice Server::GetWGPUDevice(uint32_t id, uint32_t generation) {
        return mDawnWireServer->GetDevice(id, generation);
    }

    WGPUBuffer Server::GetWGPUBuffer(uint32_t id, uint32_t generation) {
        return mDawnWireServer->GetBuffer(id, generation);
    }
#endif

    bool TrackContextChild(ContextInfo* info, ObjectType type, ObjectId id) {
        auto it = info->childObjectTypesAndIds.insert(PackObjectTypeAndId(type, id));
        if (!it.second) {
            // An object of this type and id already exists.
            return false;
        }
        return true;
    }

    bool UntrackContextChild(ContextInfo* info, ObjectType type, ObjectId id) {
        auto& children = info->childObjectTypesAndIds;
        auto it = children.find(PackObjectTypeAndId(type, id));
        if (it == children.end()) {
            // An object of this type and id was already deleted.
            return false;
        }
        children.erase(it);
        return true;
    }

}  // namespace webnn::wire::server
