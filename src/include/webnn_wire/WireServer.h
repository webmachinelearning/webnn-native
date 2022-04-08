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

#if defined(WEBNN_ENABLE_GPU_BUFFER)
namespace dawn::wire {
    class WireServer;
}
#endif

struct WebnnProcTable;

namespace webnn_wire {

    namespace server {
        class Server;
        class MemoryTransferService;
    }  // namespace server

    struct WEBNN_WIRE_EXPORT WireServerDescriptor {
        const WebnnProcTable* procs;
        CommandSerializer* serializer;
    };

    class WEBNN_WIRE_EXPORT WireServer : public CommandHandler {
      public:
        WireServer(const WireServerDescriptor& descriptor);
        ~WireServer() override;

        const volatile char* HandleCommands(const volatile char* commands,
                                            size_t size) override final;

        bool InjectInstance(WNNInstance instance, uint32_t id, uint32_t generation);

#if defined(WEBNN_ENABLE_GPU_BUFFER)
        bool InjectDawnWireServer(dawn::wire::WireServer* dawn_wire_server);
#endif
        bool InjectContext(WNNContext context, uint32_t id, uint32_t generation);
        bool InjectNamedInputs(WNNNamedInputs namedInputs,
                               uint32_t id,
                               uint32_t generation,
                               uint32_t contextId,
                               uint32_t contextGeneration);
        bool InjectNamedOperands(WNNNamedOperands namedOperands, uint32_t id, uint32_t generation);
        bool InjectNamedOutputs(WNNNamedOutputs namedOutputs, uint32_t id, uint32_t generation);

      private:
        std::unique_ptr<server::Server> mImpl;
    };

}  // namespace webnn_wire

#endif  // WEBNN_WIRE_WIRESERVER_H_
