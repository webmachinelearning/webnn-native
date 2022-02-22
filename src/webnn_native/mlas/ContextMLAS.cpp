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

#include "webnn_native/mlas/ContextMLAS.h"

#include "common/Log.h"
#include "common/RefCounted.h"
#include "webnn_native/mlas/GraphMLAS.h"

#include <core/platform/threadpool.h>

namespace webnn_native { namespace mlas {

    ContextBase* Create() {
        Ref<ContextBase> context = AcquireRef(new Context());
        reinterpret_cast<Context*>(context.Get())->CreateThreadPool();
        return context.Detach();
    }

    Context::Context() : mThreadPool(nullptr) {
    }

    Context::~Context() {
        if (mThreadPool)
            delete mThreadPool;
    }

    void Context::CreateThreadPool() {
        std::vector<size_t> cpuList = onnxruntime::Env::Default().GetThreadAffinityMasks();
        if (cpuList.empty() || cpuList.size() == 1)
            return;
        int threadPoolSize = static_cast<int>(cpuList.size());
        onnxruntime::ThreadOptions options;
        options.affinity = cpuList;
        mThreadPool = new onnxruntime::concurrency::ThreadPool(
            &onnxruntime::Env::Default(), options, nullptr, threadPoolSize, false);
    }

    WNNAS_THREADPOOL* Context::GetThreadPool() {
        return mThreadPool;
    }

    GraphBase* Context::CreateGraphImpl() {
        return new Graph(this);
    }

}}  // namespace webnn_native::mlas
