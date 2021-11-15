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

    bool Server::DoNamedInputsSet(ObjectId namedInputsId,
                                  char const* name,
                                  uint8_t const* buffer,
                                  size_t byteLength,
                                  size_t byteOffset,
                                  int32_t const* dimensions,
                                  uint32_t dimensionsCount) {
        auto* namedInputs = NamedInputsObjects().Get(namedInputsId);
        if (namedInputs == nullptr) {
            return false;
        }

        MLArrayBufferView value;
        value.buffer = const_cast<void*>(static_cast<const void*>(buffer));
        value.byteLength = byteLength;
        value.byteOffset = byteOffset;
        MLInput input;
        input.dimensions = dimensions;
        input.dimensionsCount = dimensionsCount;
        input.resource = value;
        mProcs.namedInputsSet(namedInputs->handle, name, &input);
        return true;
    }

}}  // namespace webnn_wire::server
