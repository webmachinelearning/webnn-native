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

#include "webnn/wire/client/Context.h"

#include "webnn/wire/WireCmd_autogen.h"
#include "webnn/wire/client/Client.h"

namespace webnn::wire::client {

    void Context::PushErrorScope(WNNErrorFilter filter) {
        mErrorScopeStackSize++;

        ContextPushErrorScopeCmd cmd;
        cmd.self = ToAPI(this);
        cmd.filter = filter;

        client->SerializeCommand(cmd);
    }

    bool Context::PopErrorScope(WNNErrorCallback callback, void* userdata) {
        if (mErrorScopeStackSize == 0) {
            return false;
        }
        mErrorScopeStackSize--;

        if (client->IsDisconnected()) {
            callback(WNNErrorType_DeviceLost, "GPU device disconnected", userdata);
            return true;
        }

        uint64_t serial = mErrorScopeRequestSerial++;
        ASSERT(mErrorScopes.find(serial) == mErrorScopes.end());

        mErrorScopes[serial] = {callback, userdata};

        ContextPopErrorScopeCmd cmd;
        cmd.contextId = this->id;
        cmd.requestSerial = serial;

        client->SerializeCommand(cmd);

        return true;
    }

    bool Context::OnPopErrorScopeCallback(uint64_t requestSerial,
                                          WNNErrorType type,
                                          const char* message) {
        switch (type) {
            case WNNErrorType_NoError:
            case WNNErrorType_Validation:
            case WNNErrorType_OutOfMemory:
            case WNNErrorType_Unknown:
            case WNNErrorType_DeviceLost:
                break;
            default:
                return false;
        }

        auto requestIt = mErrorScopes.find(requestSerial);
        if (requestIt == mErrorScopes.end()) {
            return false;
        }

        ErrorScopeData request = std::move(requestIt->second);

        mErrorScopes.erase(requestIt);
        request.callback(type, message, request.userdata);
        return true;
    }

    void Context::SetUncapturedErrorCallback(WNNErrorCallback callback, void* userdata) {
    }

    void Context::Compute(WNNGraph wnnGraph,
                          WNNNamedInputs inputs,
                          WNNNamedOutputs outputs,
                          WNNComputeAsyncCallback callback,
                          void* userdata) {
        if (client->IsDisconnected()) {
            callback(WNNErrorType_DeviceLost, "WebNN context disconnected", userdata);
            return;
        }

        uint64_t serial = mComputeAsyncRequestSerial++;
        ASSERT(mComputeAsyncRequests.find(serial) == mComputeAsyncRequests.end());

        mComputeAsyncRequests[serial] = {callback, userdata};

        Graph* graph = FromAPI(wnnGraph);
        NamedInputs* namedInputs = FromAPI(inputs);
        NamedOutputs* namedOutputs = FromAPI(outputs);

        ContextComputeCmd cmd;
        cmd.contextId = this->id;
        cmd.graphId = graph->id;
        cmd.requestSerial = serial;
        cmd.inputsId = namedInputs->id;
        cmd.outputsId = namedOutputs->id;

        client->SerializeCommand(cmd);
    }

    void Context::ComputeSync(WNNGraph wnnGraph, WNNNamedInputs inputs, WNNNamedOutputs outputs) {
        Graph* graph = FromAPI(wnnGraph);
        NamedInputs* namedInputs = FromAPI(inputs);
        NamedOutputs* namedOutputs = FromAPI(outputs);

        ContextComputeSyncCmd cmd;
        cmd.contextId = this->id;
        cmd.graphId = graph->id;
        cmd.inputsId = namedInputs->id;
        cmd.outputsId = namedOutputs->id;

        client->SerializeCommand(cmd);
    }

    bool Context::OnComputeAsyncCallback(uint64_t requestSerial,
                                         WNNErrorType type,
                                         const char* message) {
        auto requestIt = mComputeAsyncRequests.find(requestSerial);
        if (requestIt == mComputeAsyncRequests.end()) {
            return false;
        }

        ComputeAsyncRequest request = std::move(requestIt->second);

        mComputeAsyncRequests.erase(requestIt);
        request.callback(type, message, request.userdata);
        return true;
    }

}  // namespace webnn::wire::client
