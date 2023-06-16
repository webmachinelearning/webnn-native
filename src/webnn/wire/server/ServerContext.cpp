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

#include "webnn/wire/server/Server.h"

namespace webnn::wire::server {

    bool Server::DoContextPopErrorScope(ObjectId contextId, uint64_t requestSerial) {
        auto* context = ContextObjects().Get(contextId);
        if (context == nullptr) {
            return false;
        }

        auto userdata = MakeUserdata<ErrorScopeUserdata>();
        userdata->requestSerial = requestSerial;
        userdata->context = ObjectHandle{contextId, context->generation};

        ErrorScopeUserdata* unownedUserdata = userdata.release();
        bool success = mProcs.contextPopErrorScope(
            context->handle, ForwardToServer<&Server::OnContextPopErrorScope>, unownedUserdata);
        if (!success) {
            delete unownedUserdata;
        }
        return success;
    }

    void Server::OnContextPopErrorScope(ErrorScopeUserdata* userdata,
                                        WNNErrorType type,
                                        const char* message) {
        ReturnContextPopErrorScopeCallbackCmd cmd;
        cmd.context = userdata->context;
        cmd.requestSerial = userdata->requestSerial;
        cmd.type = type;
        cmd.message = message;

        SerializeCommand(cmd);
    }

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
            ReturnContextComputeSyncResultCmd cmd;
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

    void Server::OnContextComputeCallback(ComputeAsyncUserdata* userdata,
                                          WNNErrorType type,
                                          const char* message) {
        if (type == WNNErrorType_NoError) {
            SerializeComputeResult(userdata->namedOutputsObjectID);
        }
        ReturnContextComputeCallbackCmd cmd;
        cmd.context = userdata->context;
        cmd.requestSerial = userdata->requestSerial;
        cmd.type = type;
        cmd.message = message;

        SerializeCommand(cmd);
    }

    bool Server::DoContextComputeSync(ObjectId contextId,
                                      ObjectId graphId,
                                      ObjectId inputsId,
                                      ObjectId outputsId) {
        auto* context = ContextObjects().Get(contextId);
        auto* graph = GraphObjects().Get(graphId);
        auto* namedInputs = NamedInputsObjects().Get(inputsId);
        auto* namedOutputs = NamedOutputsObjects().Get(outputsId);
        if (context == nullptr || graph == nullptr || namedInputs == nullptr ||
            namedOutputs == nullptr) {
            return false;
        }

        mProcs.contextComputeSync(context->handle, graph->handle, namedInputs->handle,
                                  namedOutputs->handle);

#if defined(WEBNN_ENABLE_GPU_BUFFER)
        return true;
#else
        return SerializeComputeResult(outputsId);
#endif
    }

    bool Server::DoContextCompute(ObjectId contextId,
                                  ObjectId graphId,
                                  uint64_t requestSerial,
                                  ObjectId inputsId,
                                  ObjectId outputsId) {
        auto* context = ContextObjects().Get(contextId);
        auto* graph = GraphObjects().Get(graphId);
        auto* namedInputs = NamedInputsObjects().Get(inputsId);
        auto* namedOutputs = NamedOutputsObjects().Get(outputsId);
        if (context == nullptr || graph == nullptr || namedInputs == nullptr ||
            namedOutputs == nullptr) {
            return false;
        }

        auto userdata = MakeUserdata<ComputeAsyncUserdata>();
        userdata->requestSerial = requestSerial;
        userdata->context = ObjectHandle{contextId, context->generation};
        userdata->namedOutputsObjectID = outputsId;

        mProcs.contextCompute(
            context->handle, graph->handle, namedInputs->handle, namedOutputs->handle,
            ForwardToServer<&Server::OnContextComputeCallback>, userdata.release());
        return true;
    }

}  // namespace webnn::wire::server
