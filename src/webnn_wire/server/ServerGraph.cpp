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

namespace webnn_wire::server {

    bool Server::SerializeComputeResult(ObjectId outputsId) {
        auto* namedOutputs = NamedOutputsObjects().Get(outputsId);
        if (mOutputNamesMap.find(outputsId) == mOutputNamesMap.end()) {
            return false;
        }
        for (auto& name : mOutputNamesMap[outputsId]) {
            WNNArrayBufferView arrayBuffer = {};
            mProcs.namedOutputsGet(namedOutputs->handle, name.data(), &arrayBuffer);
            if (arrayBuffer.buffer == nullptr) {
                return false;
            }

            // Return the result.
            ReturnGraphComputeResultCmd cmd;
            cmd.namedOutputs = ObjectHandle{outputsId, namedOutputs->generation};
            cmd.name = name.data();
            cmd.buffer = static_cast<uint8_t*>(arrayBuffer.buffer);
            cmd.byteLength = arrayBuffer.byteLength;
            cmd.byteOffset = arrayBuffer.byteOffset;
            SerializeCommand(cmd);
        }
        // Reset the mOutputNamesMap which host in the server.
        mOutputNamesMap.erase(outputsId);
        return true;
    }

    bool Server::DoGraphCompute(ObjectId graphId, ObjectId inputsId, ObjectId outputsId) {
        auto* graph = GraphObjects().Get(graphId);
        auto* namedInputs = NamedInputsObjects().Get(inputsId);
        auto* namedOutputs = NamedOutputsObjects().Get(outputsId);
        if (graph == nullptr || namedInputs == nullptr || namedOutputs == nullptr) {
            return false;
        }

        mProcs.graphCompute(graph->handle, namedInputs->handle, namedOutputs->handle);
        return SerializeComputeResult(outputsId);
    }

    bool Server::DoGraphComputeAsync(ObjectId graphId,
                                     uint64_t requestSerial,
                                     ObjectId inputsId,
                                     ObjectId outputsId) {
        auto* graph = GraphObjects().Get(graphId);
        auto* namedInputs = NamedInputsObjects().Get(inputsId);
        auto* namedOutputs = NamedOutputsObjects().Get(outputsId);
        if (graph == nullptr || namedInputs == nullptr || namedOutputs == nullptr) {
            return false;
        }

        auto userdata = MakeUserdata<ComputeAsyncUserdata>();
        userdata->requestSerial = requestSerial;
        userdata->graph = ObjectHandle{graphId, graph->generation};
        userdata->namedOutputsObjectID = outputsId;

        mProcs.graphComputeAsync(graph->handle, namedInputs->handle, namedOutputs->handle,
                                 ForwardToServer<&Server::OnGraphComputeAsyncCallback>,
                                 userdata.release());
        return true;
    }

    void Server::OnGraphComputeAsyncCallback(ComputeAsyncUserdata* userdata,
                                             WNNComputeGraphStatus status,
                                             const char* message) {
        if (status == WNNComputeGraphStatus_Success) {
            SerializeComputeResult(userdata->namedOutputsObjectID);
        }
        ReturnGraphComputeAsyncCallbackCmd cmd;
        cmd.graph = userdata->graph;
        cmd.requestSerial = userdata->requestSerial;
        cmd.status = status;
        cmd.message = message;

        SerializeCommand(cmd);
    }

}  // namespace webnn_wire::server
