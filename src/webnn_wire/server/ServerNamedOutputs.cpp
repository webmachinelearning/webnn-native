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
                                   size_t byteLength,
                                   size_t byteOffset,
                                   uint32_t gpuBufferId,
                                   uint32_t gpuBufferGeneration) {
        auto* namedOutputs = NamedOutputsObjects().Get(namedOutputsId);
        if (namedOutputs == nullptr) {
            return false;
        }

        WNNResource resource = {};
        if (gpuBufferId != 0) {
#if defined(WEBNN_ENABLE_GPU_BUFFER)
            resource.gpuBufferView.buffer = GetWGPUBuffer(gpuBufferId, gpuBufferGeneration);
            resource.gpuBufferView.id = gpuBufferId;
            resource.gpuBufferView.generation = gpuBufferGeneration;
#endif
        } else {
            resource.arrayBufferView.byteLength = byteLength;
            resource.arrayBufferView.byteOffset = byteOffset;

            // Save the output names in server because char** type isn't supported in webnn.json to
            // get name.
            if (mOutputNamesMap.find(namedOutputsId) == mOutputNamesMap.end()) {
                std::vector<std::string> names;
                names.push_back(std::string(name));
                mOutputNamesMap.insert(std::make_pair(namedOutputsId, std::move(names)));
            } else {
                auto& outputNames = mOutputNamesMap[namedOutputsId];
                outputNames.push_back(std::string(name));
            }
        }
        mProcs.namedOutputsSet(namedOutputs->handle, name, &resource);

        return true;
    }

}  // namespace webnn_wire::server
