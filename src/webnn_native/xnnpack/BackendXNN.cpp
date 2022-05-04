// Copyright 2022 The Dawn Authors
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

#include "webnn_native/xnnpack/BackendXNN.h"

#include "common/Log.h"
#include "webnn_native/Instance.h"
#include "webnn_native/xnnpack/ContextXNN.h"

#include <thread>

namespace webnn_native::xnnpack {

    Backend::Backend(InstanceBase* instance)
        : BackendConnection(instance, wnn::BackendType::XNNPACK) {
    }

    Backend::~Backend() {
        xnn_status status = xnn_deinitialize();
        if (status != xnn_status_success) {
            dawn::ErrorLog() << "xnn_deinitialize failed: " << status;
            return;
        }
        if (mThreadpool != NULL) {
            pthreadpool_destroy(mThreadpool);
        }
    }

    MaybeError Backend::Initialize() {
        xnn_status status = xnn_initialize(NULL);
        if (status != xnn_status_success) {
            dawn::ErrorLog() << "xnn_initialize failed: " << status;
            return DAWN_INTERNAL_ERROR("Failed to intialize XNNPACK.");
        }
        // Create a thread pool with as half of the logical processors in the system.
        mThreadpool = pthreadpool_create(std::thread::hardware_concurrency() / 2);
        if (mThreadpool == NULL) {
            dawn::ErrorLog() << "pthreadpool_create failed";
            return DAWN_INTERNAL_ERROR("Failed to create thread pool.");
        }
        dawn::InfoLog() << "XNNPACK backend thread numbers: "
                        << pthreadpool_get_threads_count(mThreadpool);
        return {};
    }

    ContextBase* Backend::CreateContext(ContextOptions const* options) {
        if (options->devicePreference == wnn::DevicePreference::Gpu) {
            dawn::ErrorLog() << "XNNPACK backend only supports CPU device.";
            return nullptr;
        }
        Ref<ContextBase> context = AcquireRef(new Context(mThreadpool));
        return context.Detach();
    }

    BackendConnection* Connect(InstanceBase* instance) {
        Backend* backend = new Backend(instance);

        if (instance->ConsumedError(backend->Initialize())) {
            delete backend;
            return nullptr;
        }

        return backend;
    }

}  // namespace webnn_native::xnnpack
