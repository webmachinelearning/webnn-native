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
#include "webnn_wire/WireServer.h"

namespace webnn_wire { namespace server {

    Server::Server(const WebnnProcTable& procs,
                   CommandSerializer* serializer)
        : mSerializer(serializer),
          mProcs(procs),
          mIsAlive(std::make_shared<bool>(true)) {
    }

    Server::~Server() {
        // Un-set the error and lost callbacks since we cannot forward them
        // after the server has been destroyed.
        for (MLContext context : ContextObjects().GetAllHandles()) {
            ClearContextCallbacks(context);
        }
        DestroyAllObjects(mProcs);
    }

    void Server::ClearContextCallbacks(MLContext context) {
        // Un-set the error and lost callbacks since we cannot forward them
        // after the server has been destroyed.
        mProcs.contextSetUncapturedErrorCallback(context, nullptr, nullptr);
        // mProcs.contextSetContextLostCallback(context, nullptr, nullptr);
    }

    bool TrackDeviceChild(ContextInfo* info, ObjectType type, ObjectId id) {
        auto it = info->childObjectTypesAndIds.insert(PackObjectTypeAndId(type, id));
        if (!it.second) {
            // An object of this type and id already exists.
            return false;
        }
        return true;
    }

    bool UntrackDeviceChild(ContextInfo* info, ObjectType type, ObjectId id) {
        auto& children = info->childObjectTypesAndIds;
        auto it = children.find(PackObjectTypeAndId(type, id));
        if (it == children.end()) {
            // An object of this type and id was already deleted.
            return false;
        }
        children.erase(it);
        return true;
    }

}}  // namespace webnn_wire::server
