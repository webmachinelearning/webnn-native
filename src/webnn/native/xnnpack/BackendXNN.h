// Copyright 2022 The Dawn Authors
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

#ifndef WEBNN_NATIVE_XNNPACK_BACKENDXNN_H_
#define WEBNN_NATIVE_XNNPACK_BACKENDXNN_H_

#include "webnn/native/BackendConnection.h"
#include "webnn/native/Context.h"
#include "webnn/native/Error.h"

#include <xnnpack.h>

#include <memory>

namespace webnn::native::xnnpack {

    class Backend : public BackendConnection {
      public:
        Backend(InstanceBase* instance);
        virtual ~Backend() override;

        MaybeError Initialize();
        ContextBase* CreateContext(ContextOptions const* options = nullptr) override;

      private:
        pthreadpool_t mThreadpool;
    };

}  // namespace webnn/native::xnnpack

#endif  // WEBNN_NATIVE_XNNPACK_BACKENDXNN_H_
