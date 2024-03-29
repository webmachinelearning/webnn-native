// Copyright 2019 The WEBNN Authors
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

#ifndef WEBNN_WIRE_CLIENT_OBJECTBASE_H_
#define WEBNN_WIRE_CLIENT_OBJECTBASE_H_

#include <webnn/webnn.h>

#include "common/LinkedList.h"
#include "webnn/wire/ObjectType_autogen.h"

namespace webnn::wire::client {

    class Client;

    // All objects on the client side have:
    //  - A pointer to the Client to get where to serialize commands
    //  - The external reference count
    //  - An ID that is used to refer to this object when talking with the server side
    //  - A next/prev pointer. They are part of a linked list of objects of the same type.
    struct ObjectBase : public LinkNode<ObjectBase> {
        ObjectBase(Client* client, uint32_t refcount, uint32_t id)
            : client(client), refcount(refcount), id(id) {
        }

        ~ObjectBase() {
            RemoveFromList();
        }

        virtual void CancelCallbacksForDisconnect() {
        }

        Client* const client;
        uint32_t refcount;
        const uint32_t id;
    };

}  // namespace webnn::wire::client

#endif  // WEBNN_WIRE_CLIENT_OBJECTBASE_H_
