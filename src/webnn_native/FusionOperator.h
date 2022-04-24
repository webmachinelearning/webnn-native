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

#ifndef WEBNN_NATIVE_FUSION_OPERATOR_H_
#define WEBNN_NATIVE_FUSION_OPERATOR_H_

#include "webnn_native/GraphBuilder.h"
#include "webnn_native/ObjectBase.h"

namespace webnn::native {

    enum class FusionType : uint32_t {
        Clamp = 0x00000000,
        Relu = 0x00000001,
        Sigmoid = 0x00000002,
        LeakyRelu = 0x00000003,
        HardSwish = 0x00000004,
        Tanh = 0x00000005,
    };

    class FusionOperatorBase : public ObjectBase {
      public:
        FusionOperatorBase(GraphBuilderBase* graphBuilder, FusionType fusionType)
            : ObjectBase(graphBuilder->GetContext()), mFusionType(fusionType) {
        }

        FusionType GetFusionType() const {
            return mFusionType;
        }

      private:
        FusionType mFusionType;
    };

}  // namespace webnn::native

#endif  // WEBNN_NATIVE_FUSION_OPERATOR_H_
