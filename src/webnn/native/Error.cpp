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

#include "webnn/native/Error.h"

#include "webnn/native/ErrorData.h"
#include "webnn/native/webnn_platform.h"

namespace webnn::native {

    void IgnoreErrors(MaybeError maybeError) {
        if (maybeError.IsError()) {
            std::unique_ptr<ErrorData> errorData = maybeError.AcquireError();
            // During shutdown and destruction, device lost errors can be ignored.
            // We can also ignore other unexpected internal errors on shut down and treat it as
            // device lost so that we can continue with destruction.
            ASSERT(errorData->GetType() == InternalErrorType::DeviceLost ||
                   errorData->GetType() == InternalErrorType::Internal);
        }
    }

    wnn::ErrorType ToWNNErrorType(InternalErrorType type) {
        switch (type) {
            case InternalErrorType::Validation:
                return wnn::ErrorType::Validation;
            case InternalErrorType::OutOfMemory:
                return wnn::ErrorType::OutOfMemory;

            // There is no equivalent of Internal errors in the WebGPU API. Internal
            // errors cause the device at the API level to be lost, so treat it like a
            // DeviceLost error.
            case InternalErrorType::Internal:
            case InternalErrorType::DeviceLost:
                return wnn::ErrorType::DeviceLost;

            default:
                return wnn::ErrorType::Unknown;
        }
    }

    InternalErrorType FromWNNErrorType(wnn::ErrorType type) {
        switch (type) {
            case wnn::ErrorType::Validation:
                return InternalErrorType::Validation;
            case wnn::ErrorType::OutOfMemory:
                return InternalErrorType::OutOfMemory;
            case wnn::ErrorType::DeviceLost:
                return InternalErrorType::DeviceLost;
            default:
                return InternalErrorType::Internal;
        }
    }

}  // namespace webnn::native
