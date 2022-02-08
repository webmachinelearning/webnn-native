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

#include "webnn_wire/WireCmd_autogen.h"
#include "webnn_wire/server/Server.h"

namespace webnn_wire { namespace server {

    bool Server::DoGraphCompute(ObjectId graphId, ObjectId inputsId, ObjectId outputsId) {
        auto* graph = GraphObjects().Get(graphId);
        auto* namedInputs = NamedInputsObjects().Get(inputsId);
        auto* namedOutputs = NamedOutputsObjects().Get(outputsId);
        if (graph == nullptr || namedInputs == nullptr || namedOutputs == nullptr) {
            return false;
        }

        mProcs.graphCompute(graph->handle, namedInputs->handle, namedOutputs->handle);

        MLArrayBufferView arrayBuffer = {};
        mProcs.namedOutputsGet(namedOutputs->handle, 0, &arrayBuffer);
        // Return the result.
        ReturnGraphComputeResultCmd cmd;
        cmd.name = "TODO: use the name getting from namedOutputs";
        cmd.buffer = static_cast<uint8_t*>(arrayBuffer.buffer);
        cmd.byteLength = arrayBuffer.byteLength;
        cmd.byteOffset = arrayBuffer.byteOffset;

        SerializeCommand(cmd);
        return true;
    }

}}  // namespace webnn_wire::server
