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

#include "webnn_native/dml/ContextDML.h"

#include "common/RefCounted.h"
#include "webnn_native/dml/GraphDML.h"

namespace webnn::native::dml {

    Context::Context(ContextOptions const* options) : ContextBase(options) {
    }

#if defined(WEBNN_ENABLE_GPU_BUFFER)
    Context::Context(WGPUDevice device) : ContextBase(device) {
    }
#endif

    GraphBase* Context::CreateGraphImpl() {
        return new Graph(this);
    }

}  // namespace webnn::native::dml
