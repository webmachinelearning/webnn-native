// Copyright 2019 The Dawn Authors
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

#include "webnn_wire/WireServer.h"
#include "webnn_wire/server/Server.h"

namespace webnn_wire {

    WireServer::WireServer(const WireServerDescriptor& descriptor)
        : mImpl(new server::Server(*descriptor.procs, descriptor.serializer)) {
    }

    WireServer::~WireServer() {
        mImpl.reset();
    }

    const volatile char* WireServer::HandleCommands(const volatile char* commands, size_t size) {
        return mImpl->HandleCommands(commands, size);
    }

    bool WireServer::InjectInstance(WNNInstance instance, uint32_t id, uint32_t generation) {
        return mImpl->InjectInstance(instance, id, generation);
    }

#if defined(WEBNN_ENABLE_GPU_BUFFER)
    bool WireServer::InjectDawnWireServer(dawn_wire::WireServer* dawn_wire_server) {
        return mImpl->InjectDawnWireServer(dawn_wire_server);
    }
#endif

    bool WireServer::InjectContext(WNNContext context, uint32_t id, uint32_t generation) {
        return mImpl->InjectContext(context, id, generation);
    }

    bool WireServer::InjectNamedInputs(WNNNamedInputs namedInputs,
                                       uint32_t id,
                                       uint32_t generation,
                                       uint32_t contextId,
                                       uint32_t contextGeneration) {
        return mImpl->InjectNamedInputs(namedInputs, id, generation, contextId, contextGeneration);
    }

    bool WireServer::InjectNamedOperands(WNNNamedOperands namedOperands,
                                         uint32_t id,
                                         uint32_t generation) {
        return mImpl->InjectNamedOperands(namedOperands, id, generation);
    }

    bool WireServer::InjectNamedOutputs(WNNNamedOutputs namedOutputs,
                                        uint32_t id,
                                        uint32_t generation) {
        return mImpl->InjectNamedOutputs(namedOutputs, id, generation);
    }

}  // namespace webnn_wire
