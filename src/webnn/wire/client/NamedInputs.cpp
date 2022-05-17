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

#include "webnn/wire/client/NamedInputs.h"

#include "webnn/wire/WireCmd_autogen.h"
#include "webnn/wire/client/Client.h"

namespace webnn::wire::client {

    void NamedInputs::Set(char const* name, WNNInput const* input) {
        NamedInputsSetCmd cmd = {};
        cmd.namedInputsId = this->id;
        cmd.name = name;
        // Input type is ArrayBufferView
        WNNArrayBufferView arrayBufferView = input->resource.arrayBufferView;
        if (arrayBufferView.buffer != nullptr) {
            cmd.arrayBuffer = static_cast<const uint8_t*>(arrayBufferView.buffer);
            cmd.byteLength = arrayBufferView.byteLength;
            cmd.byteOffset = arrayBufferView.byteOffset;
        } else {
            cmd.gpuBufferId = input->resource.gpuBufferView.id;
            cmd.gpuBufferGeneration = input->resource.gpuBufferView.generation;
        }
        cmd.dimensions = input->dimensions;
        cmd.dimensionsCount = input->dimensionsCount;

        client->SerializeCommand(cmd);
    }

}  // namespace webnn::wire::client
