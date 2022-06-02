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

#ifndef WEBNN_NATIVE_ERROR_H_
#define WEBNN_NATIVE_ERROR_H_

#include "absl/strings/str_format.h"
#include "common/Result.h"
#include "webnn/native/ErrorData.h"
#include "webnn/native/webnn_absl_format_autogen.h"

#include <string>

namespace webnn::native {

    enum class InternalErrorType : uint32_t {
        Validation,
        DeviceLost,
        Internal,
        Unimplemented,
        OutOfMemory
    };

    // MaybeError and ResultOrError are meant to be used as return value for function that are not
    // expected to, but might fail. The handling of error is potentially much slower than successes.
    using MaybeError = Result<void, ErrorData>;

    template <typename T>
    using ResultOrError = Result<T, ErrorData>;

    // Returning a success is done like so:
    //   return {}; // for Error
    //   return SomethingOfTypeT; // for ResultOrError<T>
    //
    // Returning an error is done via:
    //   return DAWN_MAKE_ERROR(errorType, "My error message");
    //
    // but shorthand version for specific error types are preferred:
    //   return DAWN_VALIDATION_ERROR("My error message");
    //
    // There are different types of errors that should be used for different purpose:
    //
    //   - Validation: these are errors that show the user did something bad, which causes the
    //     whole call to be a no-op. It's most commonly found in the frontend but there can be some
    //     backend specific validation in non-conformant backends too.
    //
    //   - Out of memory: creation of a Buffer or Texture failed because there isn't enough memory.
    //     This is similar to validation errors in that the call becomes a no-op and returns an
    //     error object, but is reported separated from validation to the user.
    //
    //   - Device loss: the backend driver reported that the GPU has been lost, which means all
    //     previous commands magically disappeared and the only thing left to do is clean up.
    //     Note: Device loss should be used rarely and in most case you want to use Internal
    //     instead.
    //
    //   - Internal: something happened that the backend didn't expect, and it doesn't know
    //     how to recover from that situation. This causes the device to be lost, but is separate
    //     from device loss, because the GPU execution is still happening so we need to clean up
    //     more gracefully.
    //
    //   - Unimplemented: same as Internal except it puts "unimplemented" in the error message for
    //     more clarity.

#define DAWN_MAKE_ERROR(TYPE, MESSAGE) \
    ::webnn::native::ErrorData::Create(TYPE, MESSAGE, __FILE__, __func__, __LINE__)
#define DAWN_VALIDATION_ERROR(MESSAGE) DAWN_MAKE_ERROR(InternalErrorType::Validation, MESSAGE)
#define DAWN_DEVICE_LOST_ERROR(MESSAGE) DAWN_MAKE_ERROR(InternalErrorType::DeviceLost, MESSAGE)
#define DAWN_INTERNAL_ERROR(MESSAGE) DAWN_MAKE_ERROR(InternalErrorType::Internal, MESSAGE)
#define DAWN_UNIMPLEMENTED_ERROR(MESSAGE) \
    DAWN_MAKE_ERROR(InternalErrorType::Internal, std::string("Unimplemented: ") + MESSAGE)
#define DAWN_OUT_OF_MEMORY_ERROR(MESSAGE) DAWN_MAKE_ERROR(InternalErrorType::OutOfMemory, MESSAGE)

#define DAWN_INVALID_IF(EXPR, ...)                                                           \
    if (DAWN_UNLIKELY(EXPR)) {                                                               \
        return DAWN_MAKE_ERROR(InternalErrorType::Validation, absl::StrFormat(__VA_ARGS__)); \
    }                                                                                        \
    for (;;)                                                                                 \
    break

#define DAWN_CONCAT1(x, y) x##y
#define DAWN_CONCAT2(x, y) DAWN_CONCAT1(x, y)
#define DAWN_LOCAL_VAR DAWN_CONCAT2(_localVar, __LINE__)

    // When Errors aren't handled explicitly, calls to functions returning errors should be
    // wrapped in an DAWN_TRY. It will return the error if any, otherwise keep executing
    // the current function.
#define DAWN_TRY(EXPR)                                                                         \
    {                                                                                          \
        auto DAWN_LOCAL_VAR = EXPR;                                                            \
        if (DAWN_UNLIKELY(DAWN_LOCAL_VAR.IsError())) {                                         \
            std::unique_ptr<::webnn::native::ErrorData> error = DAWN_LOCAL_VAR.AcquireError(); \
            error->AppendBacktrace(__FILE__, __func__, __LINE__);                              \
            return {std::move(error)};                                                         \
        }                                                                                      \
    }                                                                                          \
    for (;;)                                                                                   \
    break

    // DAWN_TRY_ASSIGN is the same as DAWN_TRY for ResultOrError and assigns the success value, if
    // any, to VAR.
#define DAWN_TRY_ASSIGN(VAR, EXPR)                                            \
    {                                                                         \
        auto DAWN_LOCAL_VAR = EXPR;                                           \
        if (DAWN_UNLIKELY(DAWN_LOCAL_VAR.IsError())) {                        \
            std::unique_ptr<ErrorData> error = DAWN_LOCAL_VAR.AcquireError(); \
            error->AppendBacktrace(__FILE__, __func__, __LINE__);             \
            return {std::move(error)};                                        \
        }                                                                     \
        VAR = DAWN_LOCAL_VAR.AcquireSuccess();                                \
    }                                                                         \
    for (;;)                                                                  \
    break

    // Assert that errors are device loss so that we can continue with destruction
    void IgnoreErrors(MaybeError maybeError);

    wnn::ErrorType ToWNNErrorType(InternalErrorType type);
    InternalErrorType FromWNNErrorType(wnn::ErrorType type);

}  // namespace webnn::native

#endif  // WEBNN_NATIVE_ERROR_H_
