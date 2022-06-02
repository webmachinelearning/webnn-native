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

#ifndef WEBNN_NATIVE_WEBNN_NATIVE_H_
#define WEBNN_NATIVE_WEBNN_NATIVE_H_

#include <webnn/native/webnn_native_export.h>
#include <webnn/webnn.h>
#include <webnn/webnn_proc_table.h>
#include <string>
#include <vector>

namespace wnn {
    struct ContextOptions;
}

namespace webnn::native {

    class InstanceBase;

    // Represents a connection to dawn_native and is used for dependency injection.
    //
    // This is an RAII class for Webnn instances and also controls the lifetime of all contexts
    // for this instance.
    class WEBNN_NATIVE_EXPORT Instance {
      public:
        Instance();
        ~Instance();

        Instance(const Instance& other) = delete;
        Instance& operator=(const Instance& other) = delete;

        WNNContext CreateTestContext(const wnn::ContextOptions* options = nullptr);
        WNNContext CreateContext(const wnn::ContextOptions* options = nullptr);
        WNNGraphBuilder CreateGraphBuilder(const WNNContext context);

        // Returns the underlying WNNInstance object.
        WNNInstance Get() const;

      private:
        InstanceBase* mImpl = nullptr;
    };

    // Backend-agnostic API for webnn_native
    WEBNN_NATIVE_EXPORT const WebnnProcTable& GetProcs();

}  // namespace webnn::native

#endif  // WEBNN_NATIVE_WEBNN_NATIVE_H_
