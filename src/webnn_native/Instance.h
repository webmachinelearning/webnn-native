// Copyright 2018 The Dawn Authors
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

#ifndef WEBNNNATIVE_INSTANCE_H_
#define WEBNNNATIVE_INSTANCE_H_

#include "common/RefCounted.h"
#include "common/ityp_bitset.h"
#include "webnn_native/BackendConnection.h"
#include "webnn_native/webnn_platform.h"

#include <array>
#include <map>
#include <memory>
#include <unordered_map>

namespace webnn_native {

    using BackendsBitset = ityp::bitset<wnn::BackendType, kEnumCount<wnn::BackendType>>;

    // This is called InstanceBase for consistency across the frontend, even if the backends don't
    // specialize this class.
    class InstanceBase final : public RefCounted {
      public:
        static InstanceBase* Create(const InstanceDescriptor* descriptor = nullptr);

        // WebNN API
        ContextBase* CreateContext(const ContextOptions* options);
        ContextBase* CreateContextWithGpuDevice(const GpuDevice* device);
        GraphBuilderBase* CreateGraphBuilder(ContextBase* context);
        NamedInputsBase* CreateNamedInputs();
        NamedOperandsBase* CreateNamedOperands();
        NamedOutputsBase* CreateNamedOutputs();

        ContextBase* CreateTestContext(const ContextOptions* options);

        // Used to handle error that happen up to device creation.
        bool ConsumedError(MaybeError maybeError);

      private:
        InstanceBase() = default;
        ~InstanceBase() = default;

        InstanceBase(const InstanceBase& other) = delete;
        InstanceBase& operator=(const InstanceBase& other) = delete;

        bool Initialize(const InstanceDescriptor* descriptor);

        void ConnectBackend(wnn::BackendType backendType);

        std::map<wnn::BackendType, std::unique_ptr<BackendConnection>> mBackends;
    };

}  // namespace webnn_native

#endif  // WEBNNNATIVE_INSTANCE_H_
