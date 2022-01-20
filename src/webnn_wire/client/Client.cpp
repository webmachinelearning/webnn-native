// Copyright 2019 The Dawn Authors
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

#include "webnn_wire/client/Client.h"

#include "common/Compiler.h"

namespace webnn_wire { namespace client {

    namespace {

        class NoopCommandSerializer final : public CommandSerializer {
          public:
            static NoopCommandSerializer* GetInstance() {
                static NoopCommandSerializer gNoopCommandSerializer;
                return &gNoopCommandSerializer;
            }

            ~NoopCommandSerializer() = default;

            size_t GetMaximumAllocationSize() const final {
                return 0;
            }
            void* GetCmdSpace(size_t size) final {
                return nullptr;
            }
            bool Flush() final {
                return false;
            }
        };

    }  // anonymous namespace

    Client::Client(CommandSerializer* serializer) : ClientBase(), mSerializer(serializer) {
    }

    Client::~Client() {
        DestroyAllObjects();
    }

    void Client::DestroyAllObjects() {
        for (auto& objectList : mObjects) {
            ObjectType objectType = static_cast<ObjectType>(&objectList - mObjects.data());
            while (!objectList.empty()) {
                ObjectBase* object = objectList.head()->value();

                DestroyObjectCmd cmd;
                cmd.objectType = objectType;
                cmd.objectId = object->id;
                SerializeCommand(cmd);
                FreeObject(objectType, object);
            }
        }
    }

    ReservedContext Client::ReserveContext() {
        auto* allocation = ContextAllocator().New(this);

        ReservedContext result;
        result.context = ToAPI(allocation->object.get());
        result.id = allocation->object->id;
        result.generation = allocation->generation;
        return result;
    }

    ReservedNamedInputs Client::ReserveNamedInputs(MLContext context) {
        auto* allocation = NamedInputsAllocator().New(this);

        ReservedNamedInputs result;
        result.namedInputs = ToAPI(allocation->object.get());
        result.id = allocation->object->id;
        result.generation = allocation->generation;
        // result.contextId = FromAPI(context)->id;
        // result.contextGeneration = ContextAllocator().GetGeneration(FromAPI(context)->id);
        return result;
    }

    ReservedNamedOperands Client::ReserveNamedOperands() {
        auto* allocation = NamedOperandsAllocator().New(this);

        ReservedNamedOperands result;
        result.namedOperands = ToAPI(allocation->object.get());
        result.id = allocation->object->id;
        result.generation = allocation->generation;
        return result;
    }

    ReservedNamedOutputs Client::ReserveNamedOutputs() {
        auto* allocation = NamedOutputsAllocator().New(this);

        ReservedNamedOutputs result;
        result.namedOutputs = ToAPI(allocation->object.get());
        result.id = allocation->object->id;
        result.generation = allocation->generation;
        return result;
    }

    void Client::Disconnect() {
        mDisconnected = true;
        mSerializer = ChunkedCommandSerializer(NoopCommandSerializer::GetInstance());
        for (auto& objectList : mObjects) {
            LinkNode<ObjectBase>* object = objectList.head();
            while (object != objectList.end()) {
                object->value()->CancelCallbacksForDisconnect();
                object = object->next();
            }
        }
    }

    bool Client::IsDisconnected() const {
        return mDisconnected;
    }

}}  // namespace webnn_wire::client
