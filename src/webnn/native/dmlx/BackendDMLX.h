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

#ifndef WEBNN_NATIVE_DMLX_BACKENDDMLX_H_
#define WEBNN_NATIVE_DMLX_BACKENDDMLX_H_

#include "webnn/native/BackendConnection.h"
#include "webnn/native/Context.h"
#include "webnn/native/Error.h"

#if defined(WEBNN_ENABLE_GPU_BUFFER)
#    include <webgpu/webgpu.h>
#endif
#include <memory>

namespace webnn::native::dmlx {

    class Backend : public BackendConnection {
      public:
        Backend(InstanceBase* instance);

        MaybeError Initialize();
        ContextBase* CreateContext(ContextOptions const* options = nullptr) override;

#if defined(WEBNN_ENABLE_GPU_BUFFER)
        ContextBase* CreateContextWithGpuDevice(WGPUDevice device) override;
#endif

      private:
    };

}  // namespace webnn::native::dmlx

#endif  // WEBNN_NATIVE_DMLX_BACKENDDMLX_H_
