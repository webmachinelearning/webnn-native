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

#ifndef WEBNNNATIVE_BACKENDCONNECTION_H_
#define WEBNNNATIVE_BACKENDCONNECTION_H_

#include "webnn/native/Context.h"
#include "webnn/native/WebnnNative.h"
#include "webnn/native/webnn_platform.h"

#include <memory>

namespace webnn::native {

    // An common interface for all backends. Mostly used to create adapters for a particular
    // backend.
    class BackendConnection {
      public:
        BackendConnection(InstanceBase* instance, wnn::BackendType type);
        virtual ~BackendConnection() = default;

        wnn::BackendType GetType() const;
        InstanceBase* GetInstance() const;

        virtual ContextBase* CreateContext(ContextOptions const* options = nullptr) = 0;

#if defined(WEBNN_ENABLE_GPU_BUFFER)
        virtual ContextBase* CreateContextWithGpuDevice(WGPUDevice device) = 0;
#endif

      private:
        InstanceBase* mInstance = nullptr;
        wnn::BackendType mType;
    };

}  // namespace webnn::native

#endif  // WEBNNNATIVE_BACKENDCONNECTION_H_
