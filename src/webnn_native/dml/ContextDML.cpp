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

namespace webnn_native { namespace dml {

    ContextBase* Create(MLContextOptions const* options) {
        Ref<ContextBase> context =
            AcquireRef(new Context(reinterpret_cast<ContextOptions const*>(options)));
        if (FAILED(reinterpret_cast<Context*>(context.Get())->CreateDevice())) {
            dawn::ErrorLog() << "Failed to create DirectML device.";
            return nullptr;
        }
        return context.Detach();
    }

    Context::Context(ContextOptions const* options) {
        if (options == nullptr) {
            return;
        }
        mOptions = *options;
    }

    HRESULT Context::CreateDevice() {
#if defined(_DEBUG)
        mDevice.reset(new ::pydml::Device(true, true));
#else
        mDevice.reset(new ::pydml::Device(true, false));
#endif
        return mDevice->Init();
    }

    GraphBase* Context::CreateGraphImpl() {
        return new Graph(this);
    }

}}  // namespace webnn_native::dml
