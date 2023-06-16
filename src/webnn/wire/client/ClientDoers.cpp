// Copyright 2019 The Dawn Authors
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

#include "webnn/wire/client/Client.h"

namespace webnn::wire::client {

    bool Client::DoContextPopErrorScopeCallback(Context* context,
                                                uint64_t requestSerial,
                                                WNNErrorType errorType,
                                                const char* message) {
        if (context == nullptr) {
            // The device might have been deleted or recreated so this isn't an error.
            return true;
        }
        return context->OnPopErrorScopeCallback(requestSerial, errorType, message);
    }

    bool Client::DoContextComputeCallback(Context* context,
                                          uint64_t requestSerial,
                                          WNNErrorType type,
                                          const char* message) {
        return context->OnComputeAsyncCallback(requestSerial, type, message);
    }

    bool Client::DoContextComputeSyncResult(NamedOutputs* namedOutputs,
                                            char const* name,
                                            uint8_t const* buffer,
                                            size_t byteLength,
                                            size_t byteOffset) {
        return namedOutputs->OutputResult(name, buffer, byteLength, byteOffset);
    }

}  // namespace webnn::wire::client
