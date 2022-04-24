// Copyright 2019 The Dawn Authors
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

#include "webnn_native/dml/BackendDML.h"

#include "webnn_native/Instance.h"
#include "webnn_native/dml/ContextDML.h"

namespace webnn::native::dml {

    Backend::Backend(InstanceBase* instance)
        : BackendConnection(instance, wnn::BackendType::DirectML) {
    }

    MaybeError Backend::Initialize() {
        return {};
    }

    ContextBase* Backend::CreateContext(ContextOptions const* options) {
        return new Context(options);
    }

#if defined(WEBNN_ENABLE_GPU_BUFFER)
    ContextBase* Backend::CreateContextWithGpuDevice(WGPUDevice device) {
        return new Context(device);
    }
#endif

    BackendConnection* Connect(InstanceBase* instance) {
        Backend* backend = new Backend(instance);

        if (instance->ConsumedError(backend->Initialize())) {
            delete backend;
            return nullptr;
        }

        return backend;
    }

}  // namespace webnn::native::dml
