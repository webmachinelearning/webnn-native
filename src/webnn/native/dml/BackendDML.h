// Copyright 2019 The Dawn Authors
// Copyright 2022 The WebNN-native Authors
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

#ifndef WEBNN_NATIVE_DML_BACKEND_DML_H_
#define WEBNN_NATIVE_DML_BACKEND_DML_H_

#include <map>

#include "dml_platform.h"
#include "webnn/native/BackendConnection.h"
#include "webnn/native/Context.h"

namespace webnn::native::dml {

    struct Adapter {
        ComPtr<IDMLDevice> DMLDevice;
        ComPtr<ID3D12Device> D3D12Device;
        ComPtr<ID3D12CommandQueue> commandQueue;
        ComPtr<IDXGIAdapter1> adapter;
    };

    class Backend : public BackendConnection {
      public:
        Backend(InstanceBase* instance);

        HRESULT EnumAdapter(DXGI_GPU_PREFERENCE gpuPreference);
        MaybeError Initialize();
        ContextBase* CreateContext(ContextOptions const* options = nullptr) override;

      private:
        std::map<DXGI_GPU_PREFERENCE, Adapter> mAdapters;
        bool mUseDebugLayer;
    };

}  // namespace webnn::native::dml

#endif  // WEBNN_NATIVE_DML_BACKEND_DML_H_
