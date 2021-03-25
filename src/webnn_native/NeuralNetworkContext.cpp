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

#include "webnn_native/NeuralNetworkContext.h"

#include <sstream>

#include "webnn_native/ValidationUtils_autogen.h"
#include "webnn_native/webnn_platform.h"

namespace webnn_native {

    NeuralNetworkContextBase::NeuralNetworkContextBase() {
        mRootErrorScope = AcquireRef(new ErrorScope());
        mCurrentErrorScope = mRootErrorScope.Get();
    }

    ModelBuilderBase* NeuralNetworkContextBase::CreateModelBuilder() {
        return CreateModelBuilderImpl();
    }

    ModelBuilderBase* NeuralNetworkContextBase::CreateModelBuilderImpl() {
        UNREACHABLE();
    }

    void NeuralNetworkContextBase::PushErrorScope(webnn::ErrorFilter filter) {
        if (ConsumedError(ValidateErrorFilter(filter))) {
            return;
        }
        mCurrentErrorScope = AcquireRef(new ErrorScope(filter, mCurrentErrorScope.Get()));
    }

    bool NeuralNetworkContextBase::PopErrorScope(webnn::ErrorCallback callback, void* userdata) {
        if (DAWN_UNLIKELY(mCurrentErrorScope.Get() == mRootErrorScope.Get())) {
            return false;
        }
        mCurrentErrorScope->SetCallback(callback, userdata);
        mCurrentErrorScope = Ref<ErrorScope>(mCurrentErrorScope->GetParent());

        return true;
    }

    void NeuralNetworkContextBase::SetUncapturedErrorCallback(webnn::ErrorCallback callback,
                                                              void* userdata) {
        mRootErrorScope->SetCallback(callback, userdata);
    }

    void NeuralNetworkContextBase::HandleError(std::unique_ptr<ErrorData> error) {
        ASSERT(error != nullptr);
        std::ostringstream ss;
        ss << error->GetMessage();
        for (const auto& callsite : error->GetBacktrace()) {
            ss << "\n    at " << callsite.function << " (" << callsite.file << ":" << callsite.line
               << ")";
        }

        // Still forward device loss and internal errors to the error scopes so they
        // all reject.
        mCurrentErrorScope->HandleError(ToWebnnErrorType(error->GetType()), ss.str().c_str());
    }

}  // namespace webnn_native
