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

#ifndef WEBNN_WIRE_CLIENT_CLIENT_H_
#define WEBNN_WIRE_CLIENT_CLIENT_H_

#include <webnn/webnn.h>
#include <webnn_wire/Wire.h>

#include "common/LinkedList.h"
#include "webnn_wire/ChunkedCommandSerializer.h"
#include "webnn_wire/WireClient.h"
#include "webnn_wire/WireCmd_autogen.h"
#include "webnn_wire/WireDeserializeAllocator.h"
#include "webnn_wire/client/ClientBase_autogen.h"

namespace webnn_wire::client {

    class Client : public ClientBase {
      public:
        Client(CommandSerializer* serializer);
        ~Client() override;

        // ChunkedCommandHandler implementation
        const volatile char* HandleCommandsImpl(const volatile char* commands,
                                                size_t size) override;

        ReservedInstance ReserveInstance();
        ReservedContext ReserveContext();
        ReservedNamedInputs ReserveNamedInputs(WNNContext context);
        ReservedNamedOperands ReserveNamedOperands();
        ReservedNamedOutputs ReserveNamedOutputs();

        template <typename Cmd>
        void SerializeCommand(const Cmd& cmd) {
            mSerializer.SerializeCommand(cmd, *this);
        }

        template <typename Cmd, typename ExtraSizeSerializeFn>
        void SerializeCommand(const Cmd& cmd,
                              size_t extraSize,
                              ExtraSizeSerializeFn&& SerializeExtraSize) {
            mSerializer.SerializeCommand(cmd, *this, extraSize, SerializeExtraSize);
        }

        void Disconnect();
        bool IsDisconnected() const;

        template <typename T>
        void TrackObject(T* object) {
            mObjects[ObjectTypeToTypeEnum<T>::value].Append(object);
        }

      private:
        void DestroyAllObjects();

#include "webnn_wire/client/ClientPrototypes_autogen.inc"

        ChunkedCommandSerializer mSerializer;
        WireDeserializeAllocator mAllocator;

        PerObjectType<LinkedList<ObjectBase>> mObjects;
        bool mDisconnected = false;
    };

}  // namespace webnn_wire::client

#endif  // WEBNN_WIRE_CLIENT_CLIENT_H_
