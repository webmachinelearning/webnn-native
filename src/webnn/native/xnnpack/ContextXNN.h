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

#ifndef WEBNN_NATIVE_XNNPACK_CONTEXT_XNN_H_
#define WEBNN_NATIVE_XNNPACK_CONTEXT_XNN_H_

#include "webnn/native/Context.h"

#include <xnnpack.h>

namespace webnn::native::xnnpack {

    class Context : public ContextBase {
      public:
        explicit Context(pthreadpool_t threadpool);
        ~Context() override = default;

        pthreadpool_t GetThreadpool();

      private:
        GraphBase* CreateGraphImpl() override;

        pthreadpool_t mThreadpool;
    };

}  // namespace webnn::native::xnnpack

#endif  // WEBNN_NATIVE_XNNPACK_CONTEXT_XNN_H_
