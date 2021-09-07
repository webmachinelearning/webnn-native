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

    Client::Client(CommandSerializer* serializer)
        : ClientBase(), mSerializer(serializer) {
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
