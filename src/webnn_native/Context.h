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
#include "webnn_native/Error.h"
#include "webnn_native/ErrorScope.h"
#include "webnn_native/webnn_platform.h"

#include <webgpu/webgpu.h>

class WebGLRenderingContext;
namespace webnn_native {

    class ContextBase : public RefCounted {
      public:
        explicit ContextBase(ContextOptions const* options = nullptr);
        explicit ContextBase(WGPUDevice wgpuDevice);
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
        WGPUDevice GetWGPUDevice();

        // Dawn API
        void InjectError(wnn::ErrorType type, const char* message);
        void PushErrorScope(wnn::ErrorFilter filter);
        bool PopErrorScope(wnn::ErrorCallback callback, void* userdata);
        void SetUncapturedErrorCallback(wnn::ErrorCallback callback, void* userdata);
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
        WGPUDevice mWGPUDevice;
    };

}  // namespace webnn_native

#endif  // WEBNN_NATIVE_CONTEXT_H_
