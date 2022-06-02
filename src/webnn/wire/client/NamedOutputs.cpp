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

#include "webnn/wire/client/NamedOutputs.h"

#include "webnn/wire/WireCmd_autogen.h"
#include "webnn/wire/client/Client.h"

namespace webnn::wire::client {

    void NamedOutputs::SetOutput(char const* name, WNNResource const* resource) {
        // The type of output data is WNNArrayBufferView.
        NamedOutputsSetOutputCmd cmd = {};
        cmd.namedOutputsId = this->id;
        cmd.name = name;
        WNNArrayBufferView arrayBufferView = resource->arrayBufferView;
        if (arrayBufferView.buffer != nullptr) {
            // The output array buffer is nullptr so that it isn't serialized across process.
            cmd.byteLength = arrayBufferView.byteLength;
            cmd.byteOffset = arrayBufferView.byteOffset;

            // Save the WNNArrayBufferView in order to be copied after computing from server.
            mNamedOutputMap.insert(std::make_pair(std::string(name), arrayBufferView));
        } else {
            cmd.gpuBufferId = resource->gpuBufferView.id;
            cmd.gpuBufferGeneration = resource->gpuBufferView.generation;
        }

        client->SerializeCommand(cmd);
    }

    void NamedOutputs::GetOutput(char const* name, WNNArrayBufferView const* resource) {
        NamedOutputsGetOutputCmd cmd = {};
        cmd.namedOutputsId = this->id;
        cmd.name = name;
        if (resource->buffer != nullptr) {
            cmd.arrayBuffer = static_cast<const uint8_t*>(resource->buffer);
            cmd.byteLength = resource->byteLength;
            cmd.byteOffset = resource->byteOffset;
        } 
        client->SerializeCommand(cmd);
    }

    bool NamedOutputs::OutputResult(char const* name,
                                    uint8_t const* buffer,
                                    size_t byteLength,
                                    size_t byteOffset) {
        if (buffer == nullptr || name == nullptr) {
            return false;
        }
        if (mNamedOutputMap.find(std::string(name)) == mNamedOutputMap.end()) {
            return false;
        }
        WNNArrayBufferView arrayBufferView = mNamedOutputMap[std::string(name)];
        memcpy(static_cast<uint8_t*>(arrayBufferView.buffer) + arrayBufferView.byteOffset,
               buffer + byteOffset, byteLength);
        return true;
    }

}  // namespace webnn::wire::client
