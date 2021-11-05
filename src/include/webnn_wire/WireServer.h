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

#ifndef WEBNN_WIRE_WIRESERVER_H_
#define WEBNN_WIRE_WIRESERVER_H_

#include <memory>

#include "webnn_wire/Wire.h"

struct DawnProcTable;

namespace webnn_wire {

    namespace server {
        class Server;
        class MemoryTransferService;
    }  // namespace server

    struct WEBNN_WIRE_EXPORT WireServerDescriptor {
        const DawnProcTable* procs;
        CommandSerializer* serializer;
    };

    class WEBNN_WIRE_EXPORT WireServer : public CommandHandler {
      public:
        WireServer(const WireServerDescriptor& descriptor);
        ~WireServer() override;

        const volatile char* HandleCommands(const volatile char* commands,
                                            size_t size) override final;

      private:
        std::unique_ptr<server::Server> mImpl;
    };

}  // namespace webnn_wire

#endif  // WEBNN_WIRE_WIRESERVER_H_
