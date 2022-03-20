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

namespace webnn_wire::server {

    bool Server::DoNamedOutputsSet(ObjectId namedOutputsId,
                                   char const* name,
                                   uint8_t const* buffer,
                                   size_t byteLength,
                                   size_t byteOffset,
                                   uint32_t gpuBufferId,
                                   uint32_t gpuBufferGeneration) {
        auto* namedOutputs = NamedOutputsObjects().Get(namedOutputsId);
        if (namedOutputs == nullptr) {
            return false;
        }

        // The type of output data is ArrayBufferView
        WNNResource resource = {};
        if (buffer != nullptr) {
            resource.arrayBufferView.buffer = const_cast<void*>(static_cast<const void*>(buffer));
            resource.arrayBufferView.byteLength = byteLength;
            resource.arrayBufferView.byteOffset = byteOffset;
        } else {
            resource.gpuBufferView.buffer = GetWGPUBuffer(gpuBufferId, gpuBufferGeneration);
            resource.gpuBufferView.id = gpuBufferId;
            resource.gpuBufferView.generation = gpuBufferGeneration;
        }
        mProcs.namedOutputsSet(namedOutputs->handle, name, &resource);
        return true;
    }

    bool Server::DoNamedOutputsGet(ObjectId namedOutputsId,
                                   size_t index,
                                   uint8_t const* buffer,
                                   size_t byteLength,
                                   size_t byteOffset) {
        UNREACHABLE();
        return true;
    }

}  // namespace webnn_wire::server
