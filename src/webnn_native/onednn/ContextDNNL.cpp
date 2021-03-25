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

#include "webnn_native/onednn/ContextDNNL.h"

#include "common/Log.h"
#include "common/RefCounted.h"
#include "webnn_native/onednn/GraphDNNL.h"

namespace webnn_native { namespace onednn {

    ContextBase* Create() {
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

    Context::Context() : mEngine(nullptr) {
    }

    Context::~Context() {
        if (mEngine != nullptr) {
            dnnl_engine_destroy(mEngine);
        }
    }

    dnnl_status_t Context::CreateEngine(dnnl_engine_kind_t engineKind) {
        return dnnl_engine_create(&mEngine, engineKind, 0);
    }

    GraphBase* Context::CreateGraphImpl() {
        return new Graph(this);
    }

}}  // namespace webnn_native::onednn
