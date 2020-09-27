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

#include "webnn_native/ErrorScope.h"

#include "common/Assert.h"

namespace webnn_native {

    ErrorScope::ErrorScope() : mIsRoot(true) {
    }

    ErrorScope::ErrorScope(webnn::ErrorFilter errorFilter, ErrorScope* parent)
        : RefCounted(), mErrorFilter(errorFilter), mParent(parent), mIsRoot(false) {
        ASSERT(mParent.Get() != nullptr);
    }

    ErrorScope::~ErrorScope() {
        if (!IsRoot()) {
            RunNonRootCallback();
        }
    }

    void ErrorScope::SetCallback(webnn::ErrorCallback callback, void* userdata) {
        mCallback = callback;
        mUserdata = userdata;
    }

    ErrorScope* ErrorScope::GetParent() {
        return mParent.Get();
    }

    bool ErrorScope::IsRoot() const {
        return mIsRoot;
    }

    void ErrorScope::RunNonRootCallback() {
        ASSERT(!IsRoot());

        if (mCallback != nullptr) {
            // For non-root error scopes, the callback can run at most once.
            mCallback(static_cast<WebnnErrorType>(mErrorType), mErrorMessage.c_str(), mUserdata);
            mCallback = nullptr;
        }
    }

    void ErrorScope::HandleError(webnn::ErrorType type, const char* message) {
        HandleErrorImpl(this, type, message);
    }

    void ErrorScope::UnlinkForShutdown() {
        UnlinkForShutdownImpl(this);
    }

    // static
    void ErrorScope::HandleErrorImpl(ErrorScope* scope,
                                     webnn::ErrorType type,
                                     const char* message) {
        ErrorScope* currentScope = scope;
        for (; !currentScope->IsRoot(); currentScope = currentScope->GetParent()) {
            ASSERT(currentScope != nullptr);

            bool consumed = false;
            switch (type) {
                case webnn::ErrorType::Validation:
                    if (currentScope->mErrorFilter != webnn::ErrorFilter::Validation) {
                        // Error filter does not match. Move on to the next scope.
                        continue;
                    }
                    consumed = true;
                    break;

                case webnn::ErrorType::OutOfMemory:
                    if (currentScope->mErrorFilter != webnn::ErrorFilter::OutOfMemory) {
                        // Error filter does not match. Move on to the next scope.
                        continue;
                    }
                    consumed = true;
                    break;

                // Unknown and DeviceLost are fatal. All error scopes capture them.
                // |consumed| is false because these should bubble to all scopes.
                case webnn::ErrorType::Unknown:
                case webnn::ErrorType::DeviceLost:
                    consumed = false;
                    break;

                case webnn::ErrorType::NoError:
                    UNREACHABLE();
                    return;
            }

            // Record the error if the scope doesn't have one yet.
            if (currentScope->mErrorType == webnn::ErrorType::NoError) {
                currentScope->mErrorType = type;
                currentScope->mErrorMessage = message;
            }

            if (consumed) {
                return;
            }
        }

        // The root error scope captures all uncaptured errors.
        ASSERT(currentScope->IsRoot());
        if (currentScope->mCallback) {
            currentScope->mCallback(static_cast<WebnnErrorType>(type), message,
                                    currentScope->mUserdata);
        }
    }

    // static
    void ErrorScope::UnlinkForShutdownImpl(ErrorScope* scope) {
        Ref<ErrorScope> currentScope = scope;
        Ref<ErrorScope> parentScope = nullptr;
        for (; !currentScope->IsRoot(); currentScope = parentScope.Get()) {
            ASSERT(!currentScope->IsRoot());
            ASSERT(currentScope.Get() != nullptr);
            parentScope = std::move(currentScope->mParent);
            ASSERT(parentScope.Get() != nullptr);

            // On shutdown, error scopes that have yet to have a status get Unknown.
            if (currentScope->mErrorType == webnn::ErrorType::NoError) {
                currentScope->mErrorType = webnn::ErrorType::Unknown;
                currentScope->mErrorMessage = "Error scope destroyed";
            }

            // Run the callback if it hasn't run already.
            currentScope->RunNonRootCallback();
        }
    }

}  // namespace webnn_native
