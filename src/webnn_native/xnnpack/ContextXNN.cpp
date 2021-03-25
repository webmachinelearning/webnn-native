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

#include "webnn_native/xnnpack/ContextXNN.h"

#include <thread>

#include "common/Log.h"
#include "common/RefCounted.h"
#include "webnn_native/xnnpack/GraphXNN.h"

namespace webnn_native { namespace xnnpack {

    ContextBase* Create() {
        Ref<ContextBase> context = AcquireRef(new Context());
        xnn_status status = reinterpret_cast<Context*>(context.Get())->Init();
        if (status != xnn_status_success) {
            dawn::ErrorLog() << "Failed to init XNNPack:" << status;
            return nullptr;
        }
        return context.Detach();
    }

    Context::Context() {
    }

    Context::~Context() {
        xnn_status status = xnn_deinitialize();
        if (status != xnn_status_success) {
            dawn::ErrorLog() << "xnn_deinitialize failed: " << status;
            return;
        }
        if (mThreadpool != NULL) {
            pthreadpool_destroy(mThreadpool);
        }
    }

    xnn_status Context::Init() {
        xnn_status status = xnn_initialize(NULL);
        if (status != xnn_status_success) {
            dawn::ErrorLog() << "xnn_initialize failed: " << status;
            return status;
        }
        // Create a thread pool with as half of the logical processors in the system.
        mThreadpool = pthreadpool_create(std::thread::hardware_concurrency() / 2);
        if (mThreadpool == NULL) {
            dawn::ErrorLog() << "pthreadpool_create failed";
            return xnn_status_out_of_memory;
        }
        dawn::InfoLog() << "XNNPACK backend thread numbers: "
                        << pthreadpool_get_threads_count(mThreadpool);
        return xnn_status_success;
    }

    pthreadpool_t Context::GetThreadpool() {
        return mThreadpool;
    }

    GraphBase* Context::CreateGraphImpl() {
        return new Graph(this);
    }

}}  // namespace webnn_native::xnnpack
