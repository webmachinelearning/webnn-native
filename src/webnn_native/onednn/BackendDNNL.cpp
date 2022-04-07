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

#include "webnn_native/onednn/BackendDNNL.h"

#include "common/Log.h"
#include "webnn_native/Instance.h"
#include "webnn_native/onednn/ContextDNNL.h"

namespace webnn_native::onednn {

    Backend::Backend(InstanceBase* instance)
        : BackendConnection(instance, wnn::BackendType::OneDNN) {
    }

    MaybeError Backend::Initialize() {
        return {};
    }

    ContextBase* Backend::CreateContext(ContextOptions const* options) {
        Ref<ContextBase> context = AcquireRef(new Context());
        dnnl_status_t status = reinterpret_cast<Context*>(context.Get())->CreateEngine();
        if (status != dnnl_success) {
            dawn::ErrorLog() << "Failed to create oneDNN engine.";
            return nullptr;
        }

        dnnl_engine_t engine = reinterpret_cast<Context*>(context.Get())->GetEngine();
        dnnl_engine_kind_t engineKind;
        status = dnnl_engine_get_kind(engine, &engineKind);
        if (status != dnnl_success) {
            dawn::ErrorLog() << "Failed to get oneDNN engine kind.";
            return nullptr;
        }
        if (engineKind == dnnl_cpu) {
            dawn::InfoLog() << "Created oneDNN CPU engine.";
        } else if (engineKind == dnnl_gpu) {
            dawn::InfoLog() << "Created oneDNN GPU engine.";
        }
        return context.Detach();
    }

    BackendConnection* Connect(InstanceBase* instance) {
        Backend* backend = new Backend(instance);

        if (instance->ConsumedError(backend->Initialize())) {
            delete backend;
            return nullptr;
        }

        return backend;
    }

}  // namespace webnn_native::onednn
