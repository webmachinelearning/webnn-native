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

#ifndef WEBNN_WIRE_WIRECLIENT_H_
#define WEBNN_WIRE_WIRECLIENT_H_

#include "webnn/webnn_proc_table.h"
#include "webnn/wire/Wire.h"

#include <memory>
#include <vector>

namespace webnn::wire {

    namespace client {
        class Client;

        WEBNN_WIRE_EXPORT const WebnnProcTable& GetProcs();
    }  // namespace client

    struct WEBNN_WIRE_EXPORT WireClientDescriptor {
        CommandSerializer* serializer;
    };

    struct ReservedInstance {
        WNNInstance instance;
        uint32_t id;
        uint32_t generation;
    };

    struct ReservedContext {
        WNNContext context;
        uint32_t id;
        uint32_t generation;
    };

    struct ReservedNamedInputs {
        WNNNamedInputs namedInputs;
        uint32_t id;
        uint32_t generation;
        uint32_t contextId;
        uint32_t contextGeneration;
    };

    struct ReservedNamedOperands {
        WNNNamedOperands namedOperands;
        uint32_t id;
        uint32_t generation;
    };

    struct ReservedNamedOutputs {
        WNNNamedOutputs namedOutputs;
        uint32_t id;
        uint32_t generation;
    };

    class WEBNN_WIRE_EXPORT WireClient : public CommandHandler {
      public:
        WireClient(const WireClientDescriptor& descriptor);
        ~WireClient() override;

        const volatile char* HandleCommands(const volatile char* commands,
                                            size_t size) override final;

        ReservedInstance ReserveInstance();
        ReservedContext ReserveContext();
        ReservedNamedInputs ReserveNamedInputs(WNNContext context);
        ReservedNamedOperands ReserveNamedOperands();
        ReservedNamedOutputs ReserveNamedOutputs();

        // Disconnects the client.
        // Commands allocated after this point will not be sent.
        void Disconnect();

      private:
        std::unique_ptr<client::Client> mImpl;
    };

}  // namespace webnn::wire

#endif  // WEBNN_WIRE_WIRECLIENT_H_
