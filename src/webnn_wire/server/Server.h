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

namespace webnn_wire { namespace server {

    class Server : public ServerBase {
      public:
        Server(const WebnnProcTable& procs,
               CommandSerializer* serializer);
        ~Server() override;

        // ChunkedCommandHandler implementation
        const volatile char* HandleCommandsImpl(const volatile char* commands,
                                                size_t size) override;

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

        void ClearContextCallbacks(MLContext context);

        // Error callbacks
        void OnUncapturedError(MLErrorType type, const char* message);
        void OnContextLost(const char* message);
        // void OnContextPopErrorScope(MLErrorType type,
        //                            const char* message,
        //                            ErrorScopeUserdata* userdata);

#include "webnn_wire/server/ServerPrototypes_autogen.inc"

        WireDeserializeAllocator mAllocator;
        ChunkedCommandSerializer mSerializer;
        WebnnProcTable mProcs;

        std::shared_ptr<bool> mIsAlive;
    };

    bool TrackDeviceChild(ContextInfo* context, ObjectType type, ObjectId id);
    bool UntrackDeviceChild(ContextInfo* context, ObjectType type, ObjectId id);

}}  // namespace webnn_wire::server

#endif  // WEBNN_WIRE_SERVER_SERVER_H_
