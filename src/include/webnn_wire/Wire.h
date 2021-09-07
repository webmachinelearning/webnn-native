// Copyright 2017 The Dawn Authors
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

#ifndef WEBNN_WIRE_WIRE_H_
#define WEBNN_WIRE_WIRE_H_

#include <cstdint>
#include <limits>

#include "webnn/webnn.h"
#include "webnn_wire/webnn_wire_export.h"

namespace webnn_wire {

    class WEBNN_WIRE_EXPORT CommandSerializer {
      public:
        virtual ~CommandSerializer() = default;

        // Get space for serializing commands.
        // GetCmdSpace will never be called with a value larger than
        // what GetMaximumAllocationSize returns. Return nullptr to indicate
        // a fatal error.
        virtual void* GetCmdSpace(size_t size) = 0;
        virtual bool Flush() = 0;
        virtual size_t GetMaximumAllocationSize() const = 0;
    };

    class WEBNN_WIRE_EXPORT CommandHandler {
      public:
        virtual ~CommandHandler() = default;
        virtual const volatile char* HandleCommands(const volatile char* commands, size_t size) = 0;
    };

}  // namespace webnn_wire

#endif  // WEBNN_WIRE_WIRE_H_
