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

#include "webnn_wire/client/Instance.h"

#include "webnn_wire/WireCmd_autogen.h"
#include "webnn_wire/client/ApiObjects_autogen.h"
#include "webnn_wire/client/Client.h"

namespace webnn_wire { namespace client {

    WNNContext Instance::CreateContextWithGpuDevice(WNNGpuDevice const* value) {
        InstanceCreateContextWithGpuDeviceInternalCmd cmd;
        cmd.instanceId = this->id;
        // The value of device in client is nullptr that will be set in ServerInstance.
        cmd.device = nullptr;
        cmd.id = value->id;
        cmd.generation = value->generation;

        // Create the Context and send the create context command.
        auto* allocation = client->ContextAllocator().New(client);
        Context* context = allocation->object.get();
        cmd.result = ObjectHandle{context->id, allocation->generation};
        client->SerializeCommand(cmd);

        return ToAPI(context);
    }

}}  // namespace webnn_wire::client
