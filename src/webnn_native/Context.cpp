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

#include "webnn_native/Context.h"

#include "webnn_native/ValidationUtils_autogen.h"
#include "webnn_native/webnn_platform.h"

#if defined(WEBNN_ENABLE_GPU_BUFFER)
#    include <dawn/dawn_proc.h>
#    include <dawn_native/DawnNative.h>
#endif
#include <sstream>

namespace webnn_native {

    ContextBase::ContextBase(ContextOptions const* options)
#if defined(WEBNN_ENABLE_GPU_BUFFER)
        : mWGPUDevice(nullptr)
#endif
    {
        if (options != nullptr) {
            mContextOptions = *options;
        }
        mRootErrorScope = AcquireRef(new ErrorScope());
        mCurrentErrorScope = mRootErrorScope.Get();
    }

#if defined(WEBNN_ENABLE_GPU_BUFFER)
    ContextBase::ContextBase(WGPUDevice wgpuDevice) {
        DawnProcTable backend_procs = dawn_native::GetProcs();
        dawnProcSetProcs(&backend_procs);
        mWGPUDevice = wgpuDevice;
        wgpuDeviceReference(mWGPUDevice);
        mRootErrorScope = AcquireRef(new ErrorScope());
        mCurrentErrorScope = mRootErrorScope.Get();
    }
#endif

    ContextBase::~ContextBase() {
#if defined(WEBNN_ENABLE_GPU_BUFFER)
        if (mWGPUDevice)
            wgpuDeviceRelease(mWGPUDevice);
#endif
    }

    GraphBase* ContextBase::CreateGraph() {
        return CreateGraphImpl();
    }

#if defined(WEBNN_ENABLE_GPU_BUFFER)
    WGPUDevice ContextBase::GetWGPUDevice() {
        return mWGPUDevice;
    }
#endif

    void ContextBase::InjectError(wnn::ErrorType type, const char* message) {
        if (ConsumedError(ValidateErrorType(type))) {
            return;
        }

        // This method should only be used to make error scope reject.
        if (type != wnn::ErrorType::Validation && type != wnn::ErrorType::OutOfMemory) {
            HandleError(
                DAWN_VALIDATION_ERROR("Invalid injected error, must be Validation or OutOfMemory"));
            return;
        }

        HandleError(DAWN_MAKE_ERROR(FromWNNErrorType(type), message));
    }

    void ContextBase::PushErrorScope(wnn::ErrorFilter filter) {
        if (ConsumedError(ValidateErrorFilter(filter))) {
            return;
        }
        mCurrentErrorScope = AcquireRef(new ErrorScope(filter, mCurrentErrorScope.Get()));
    }

    bool ContextBase::PopErrorScope(wnn::ErrorCallback callback, void* userdata) {
        if (DAWN_UNLIKELY(mCurrentErrorScope.Get() == mRootErrorScope.Get())) {
            return false;
        }
        mCurrentErrorScope->SetCallback(callback, userdata);
        mCurrentErrorScope = Ref<ErrorScope>(mCurrentErrorScope->GetParent());

        return true;
    }

    void ContextBase::SetUncapturedErrorCallback(wnn::ErrorCallback callback, void* userdata) {
        mRootErrorScope->SetCallback(callback, userdata);
    }

    void ContextBase::HandleError(std::unique_ptr<ErrorData> error) {
        ASSERT(error != nullptr);
        std::ostringstream ss;
        ss << error->GetMessage();
        for (const auto& callsite : error->GetBacktrace()) {
            ss << "\n    at " << callsite.function << " (" << callsite.file << ":" << callsite.line
               << ")";
        }

        // Still forward device loss and internal errors to the error scopes so they
        // all reject.
        mCurrentErrorScope->HandleError(ToWNNErrorType(error->GetType()), ss.str().c_str());
    }

}  // namespace webnn_native
