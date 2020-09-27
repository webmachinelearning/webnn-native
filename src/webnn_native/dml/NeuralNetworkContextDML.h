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

#ifndef WEBNN_NATIVE_DML_NEURAL_NETWORK_CONTEXT_DML_H_
#define WEBNN_NATIVE_DML_NEURAL_NETWORK_CONTEXT_DML_H_

#include "webnn_native/NeuralNetworkContext.h"
#include "webnn_native/dml/deps/src/precomp.h"

namespace webnn_native { namespace dml {

    class NeuralNetworkContext : public NeuralNetworkContextBase {
      public:
        NeuralNetworkContext() = default;
        ~NeuralNetworkContext() override = default;

        ModelBuilderBase* CreateModelBuilderImpl() override;

        HRESULT CreateDevice();

        std::shared_ptr<::pydml::Device> GetDevice() {
            return mDevice;
        }

      private:
        std::shared_ptr<::pydml::Device> mDevice;
    };

}}  // namespace webnn_native::dml

#endif  // WEBNN_NATIVE_DML_NEURAL_NETWORK_CONTEXT_DML_H_
