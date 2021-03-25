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

#ifndef WEBNN_NATIVE_ONEDNN_CONTEXT_DNNL_H_
#define WEBNN_NATIVE_ONEDNN_CONTEXT_DNNL_H_

#include "webnn_native/Context.h"

#include <dnnl.h>

namespace webnn_native { namespace onednn {

    class Context : public ContextBase {
      public:
        Context();
        ~Context() override;

        dnnl_status_t CreateEngine(dnnl_engine_kind_t engineKind = dnnl_cpu);

        dnnl_engine_t GetEngine() {
            return mEngine;
        }

      private:
        GraphBase* CreateGraphImpl() override;

        dnnl_engine_t mEngine;
    };

}}  // namespace webnn_native::onednn

#endif  // WEBNN_NATIVE_ONEDNN_CONTEXT_DNNL_H_
