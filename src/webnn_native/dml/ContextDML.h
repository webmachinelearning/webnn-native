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

#ifndef WEBNN_NATIVE_DML_CONTEXT_DML_H_
#define WEBNN_NATIVE_DML_CONTEXT_DML_H_

#include "webnn_native/Context.h"
#include "webnn_native/Graph.h"
#include "webnn_native/dml/deps/src/precomp.h"

namespace webnn_native { namespace dml {

    class Context : public ContextBase {
      public:
        Context(ContextOptions const* options);
        ~Context() override = default;

        HRESULT CreateDevice();
        GraphBase* CreateGraphImpl() override;

        std::shared_ptr<::pydml::Device> GetDevice() {
            return mDevice;
        }

      private:
        std::shared_ptr<::pydml::Device> mDevice;
        ContextOptions mOptions;
    };

}}  // namespace webnn_native::dml

#endif  // WEBNN_NATIVE_DML_CONTEXT_DML_H_
