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

#ifndef WEBNN_NATIVE_NEURAL_NETWORK_CONTEXT_H_
#define WEBNN_NATIVE_NEURAL_NETWORK_CONTEXT_H_

#include "common/RefCounted.h"
#include "webnn_native/Error.h"
#include "webnn_native/ErrorScope.h"
#include "webnn_native/webnn_platform.h"

namespace webnn_native {

    class NeuralNetworkContextBase : public RefCounted {
      public:
        NeuralNetworkContextBase();
        virtual ~NeuralNetworkContextBase() = default;

        bool ConsumedError(MaybeError maybeError) {
            if (DAWN_UNLIKELY(maybeError.IsError())) {
                HandleError(maybeError.AcquireError());
                return true;
            }
            return false;
        }

        // Dawn API
        ModelBuilderBase* CreateModelBuilder();
        void PushErrorScope(webnn::ErrorFilter filter);
        bool PopErrorScope(webnn::ErrorCallback callback, void* userdata);
        void SetUncapturedErrorCallback(webnn::ErrorCallback callback, void* userdata);

      private:
        void HandleError(std::unique_ptr<ErrorData> error);
        virtual ModelBuilderBase* CreateModelBuilderImpl();

        Ref<ErrorScope> mRootErrorScope;
        Ref<ErrorScope> mCurrentErrorScope;
    };

}  // namespace webnn_native

#endif  // WEBNN_NATIVE_NEURAL_NETWORK_CONTEXT_H_
