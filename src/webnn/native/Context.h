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

#ifndef WEBNN_NATIVE_CONTEXT_H_
#define WEBNN_NATIVE_CONTEXT_H_

#include "common/RefCounted.h"
#include "webnn/native/Error.h"
#include "webnn/native/ErrorScope.h"
#include "webnn/native/webnn_platform.h"

#if defined(WEBNN_ENABLE_GPU_BUFFER)
#    include <webgpu/webgpu.h>
#endif

class WebGLRenderingContext;
namespace webnn::native {

    class ContextBase : public RefCounted {
      public:
        explicit ContextBase(ContextOptions const* options = nullptr);
#if defined(WEBNN_ENABLE_GPU_BUFFER)
        explicit ContextBase(WGPUDevice wgpuDevice);
#endif
        virtual ~ContextBase();

        bool ConsumedError(MaybeError maybeError) {
            if (DAWN_UNLIKELY(maybeError.IsError())) {
                HandleError(maybeError.AcquireError());
                return true;
            }
            return false;
        }

        template <typename T>
        bool ConsumedError(ResultOrError<T> resultOrError, T* result) {
            if (DAWN_UNLIKELY(resultOrError.IsError())) {
                HandleError(resultOrError.AcquireError());
                return true;
            }
            *result = resultOrError.AcquireSuccess();
            return false;
        }

        GraphBase* CreateGraph();
#if defined(WEBNN_ENABLE_GPU_BUFFER)
        WGPUDevice GetWGPUDevice();
#endif

        // Webnn API
        void APIInjectError(wnn::ErrorType type, const char* message);
        void APIPushErrorScope(wnn::ErrorFilter filter);
        bool APIPopErrorScope(wnn::ErrorCallback callback, void* userdata);
        void APISetUncapturedErrorCallback(wnn::ErrorCallback callback, void* userdata);

        ContextOptions GetContextOptions() {
            return mContextOptions;
        }

      private:
        // Create concrete model.
        virtual GraphBase* CreateGraphImpl() = 0;

        void HandleError(std::unique_ptr<ErrorData> error);

        Ref<ErrorScope> mRootErrorScope;
        Ref<ErrorScope> mCurrentErrorScope;

        ContextOptions mContextOptions;
#if defined(WEBNN_ENABLE_GPU_BUFFER)
        WGPUDevice mWGPUDevice;
#endif
    };

}  // namespace webnn::native

#endif  // WEBNN_NATIVE_CONTEXT_H_
