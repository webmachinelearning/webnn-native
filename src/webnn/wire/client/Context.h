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

#ifndef WEBNN_WIRE_CLIENT_CONTEXT_H_
#define WEBNN_WIRE_CLIENT_CONTEXT_H_

#include <webnn/webnn.h>

#include "webnn/wire/WireClient.h"
#include "webnn/wire/client/ObjectBase.h"

#include <map>

namespace webnn::wire::client {

    class Context final : public ObjectBase {
      public:
        using ObjectBase::ObjectBase;

        void PushErrorScope(WNNErrorFilter filter);
        bool PopErrorScope(WNNErrorCallback callback, void* userdata);
        void SetUncapturedErrorCallback(WNNErrorCallback callback, void* userdata);

        bool OnPopErrorScopeCallback(uint64_t requestSerial,
                                     WNNErrorType type,
                                     const char* message);

        void Compute(WNNGraph wnnGraph,
                     WNNNamedInputs inputs,
                     WNNNamedOutputs outputs,
                     WNNComputeAsyncCallback callback,
                     void* userdata);
        void ComputeSync(WNNGraph wnnGraph, WNNNamedInputs inputs, WNNNamedOutputs outputs);

        bool OnComputeAsyncCallback(uint64_t requestSerial, WNNErrorType type, const char* message);

      private:
        struct ErrorScopeData {
            WNNErrorCallback callback = nullptr;
            void* userdata = nullptr;
        };
        std::map<uint64_t, ErrorScopeData> mErrorScopes;
        uint64_t mErrorScopeRequestSerial = 0;
        uint64_t mErrorScopeStackSize = 0;

        struct ComputeAsyncRequest {
            WNNComputeAsyncCallback callback = nullptr;
            void* userdata = nullptr;
        };
        std::map<uint64_t, ComputeAsyncRequest> mComputeAsyncRequests;
        uint64_t mComputeAsyncRequestSerial = 0;
    };

}  // namespace webnn::wire::client

#endif  // WEBNN_WIRE_CLIENT_CONTEXT_H_
