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

#ifndef WEBNN_NATIVE_COMPILATION_H_
#define WEBNN_NATIVE_COMPILATION_H_

#include "common/RefCounted.h"
#include "webnn_native/Forward.h"
#include "webnn_native/NamedInputs.h"
#include "webnn_native/NamedOutputs.h"
#include "webnn_native/ObjectBase.h"
#include "webnn_native/webnn_platform.h"

namespace webnn_native {

    class CompilationBase : public RefCounted {
      public:
        CompilationBase() = default;
        virtual ~CompilationBase() = default;

        // Dawn API
        void Compute(NamedInputsBase* inputs,
                     WebnnComputeCallback callback,
                     void* userdata,
                     NamedOutputsBase* outputs = nullptr);

      private:
        virtual void ComputeImpl(NamedInputsBase* inputs,
                                 WebnnComputeCallback callback,
                                 void* userdata,
                                 NamedOutputsBase* outputs) = 0;
    };
}  // namespace webnn_native

#endif  // WEBNN_NATIVE_COMPILATION_H_