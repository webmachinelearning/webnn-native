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

#include "webnn_wire/server/Server.h"

namespace webnn_wire { namespace server {

    // void Server::OnUncapturedError(ObjectHandle context, MLErrorType type, const char* message) {
    //     ReturnContextUncapturedErrorCallbackCmd cmd;
    //     cmd.context = context;
    //     cmd.type = type;
    //     cmd.message = message;

    //     SerializeCommand(cmd);
    // }

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
            context->handle,
            ForwardToServer<decltype(
                &Server::OnContextPopErrorScope)>::Func<&Server::OnContextPopErrorScope>(),
            unownedUserdata);
        if (!success) {
            delete unownedUserdata;
        }
        return success;
    }

    void Server::OnContextPopErrorScope(MLErrorType type,
                                        const char* message,
                                        ErrorScopeUserdata* userdata) {
        ReturnContextPopErrorScopeCallbackCmd cmd;
        cmd.context = userdata->context;
        cmd.requestSerial = userdata->requestSerial;
        cmd.type = type;
        cmd.message = message;

        SerializeCommand(cmd);
    }

}}  // namespace webnn_wire::server
