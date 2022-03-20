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

    bool Server::DoInstanceCreateContextWithGpuDeviceInternal(ObjectId instanceId,
                                                              uint8_t const* device,
                                                              uint32_t id,
                                                              uint32_t generation,
                                                              ObjectHandle result) {
        auto* instance = InstanceObjects().Get(instanceId);
        if (instance == nullptr) {
            return false;
        }

        // Create and register the context object.
        auto* resultData = ContextObjects().Allocate(result.id);
        if (resultData == nullptr) {
            return false;
        }
        resultData->generation = result.generation;
        resultData->contextInfo = instance->contextInfo;
        if (resultData->contextInfo != nullptr) {
            if (!TrackContextChild(resultData->contextInfo, ObjectType::Context, result.id)) {
                return false;
            }
        }
        WNNGpuDevice value;
        value.device = GetWGPUDevice(id, generation);
        value.id = id;
        value.generation = generation;
        resultData->handle = mProcs.instanceCreateContextWithGpuDevice(instance->handle, &value);
        return true;
    }

}}  // namespace webnn_wire::server
