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

#include "webnn_native/dml/NeuralNetworkContextDML.h"

#include "common/RefCounted.h"
#include "webnn_native/dml/ModelBuilderDML.h"

namespace webnn_native { namespace dml {

    NeuralNetworkContextBase* Create() {
        Ref<NeuralNetworkContextBase> context = AcquireRef(new NeuralNetworkContext());
        if (FAILED(reinterpret_cast<NeuralNetworkContext*>(context.Get())->CreateDevice())) {
            dawn::ErrorLog() << "Failed to create DirectML device.";
            return nullptr;
        }
        return context.Detach();
    }

    HRESULT NeuralNetworkContext::CreateDevice() {
#if defined(_DEBUG)
        mDevice.reset(new ::pydml::Device(true, true));
#else
        mDevice.reset(new ::pydml::Device(true, false));
#endif
        return mDevice->Init();
    }

    ModelBuilderBase* NeuralNetworkContext::CreateModelBuilderImpl() {
        return new ModelBuilder(this);
    }

}}  // namespace webnn_native::dml
