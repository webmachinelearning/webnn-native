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

#ifndef WEBNN_NATIVE_ERRORDATA_H_
#define WEBNN_NATIVE_ERRORDATA_H_

#include "common/Compiler.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace wnn {
    enum class ErrorType : uint32_t;
}

namespace dawn {
    using ErrorType = wnn::ErrorType;
}

namespace webnn::native {
    enum class InternalErrorType : uint32_t;

    class ErrorData {
      public:
        [[nodiscard]] static std::unique_ptr<ErrorData> Create(InternalErrorType type,
                                                               std::string message,
                                                               const char* file,
                                                               const char* function,
                                                               int line);
        ErrorData(InternalErrorType type, std::string message);

        struct BacktraceRecord {
            const char* file;
            const char* function;
            int line;
        };
        void AppendBacktrace(const char* file, const char* function, int line);

        InternalErrorType GetType() const;
        const std::string& GetMessage() const;
        const std::vector<BacktraceRecord>& GetBacktrace() const;

      private:
        InternalErrorType mType;
        std::string mMessage;
        std::vector<BacktraceRecord> mBacktrace;
    };

}  // namespace webnn::native

#endif  // WEBNN_NATIVE_ERRORDATA_H_
