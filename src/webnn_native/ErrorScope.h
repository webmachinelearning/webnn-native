// Copyright 2019 The Dawn Authors
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

#ifndef WEBNN_NATIVE_ERRORSCOPE_H_
#define WEBNN_NATIVE_ERRORSCOPE_H_

#include "webnn_native/webnn_platform.h"

#include "common/RefCounted.h"

#include <string>

namespace webnn_native {

    // Errors can be recorded into an ErrorScope by calling |HandleError|.
    // Because an error scope should not resolve until contained
    // commands are complete, calling the callback is deferred until it is
    // destructed. In-flight commands or asynchronous events should hold a reference
    // to the ErrorScope for their duration.
    //
    // Because parent ErrorScopes should not resolve before child ErrorScopes,
    // ErrorScopes hold a reference to their parent.
    //
    // To simplify ErrorHandling, there is a sentinel root error scope which has
    // no parent. All uncaptured errors are handled by the root error scope. Its
    // callback is called immediately once it encounters an error.
    class ErrorScope final : public RefCounted {
      public:
        ErrorScope();  // Constructor for the root error scope.
        ErrorScope(wnn::ErrorFilter errorFilter, ErrorScope* parent);

        void SetCallback(wnn::ErrorCallback callback, void* userdata);
        ErrorScope* GetParent();

        void HandleError(wnn::ErrorType type, const char* message);
        void UnlinkForShutdown();

      private:
        ~ErrorScope() override;
        bool IsRoot() const;
        void RunNonRootCallback();

        static void HandleErrorImpl(ErrorScope* scope, wnn::ErrorType type, const char* message);
        static void UnlinkForShutdownImpl(ErrorScope* scope);

        wnn::ErrorFilter mErrorFilter = wnn::ErrorFilter::None;
        Ref<ErrorScope> mParent = nullptr;
        bool mIsRoot;

        wnn::ErrorCallback mCallback = nullptr;
        void* mUserdata = nullptr;

        wnn::ErrorType mErrorType = wnn::ErrorType::NoError;
        std::string mErrorMessage = "";
    };

}  // namespace webnn_native

#endif  // WEBNN_NATIVE_ERRORSCOPE_H_
