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

#include "webnn/native/dml/BackendDML.h"

#include "webnn/native/Instance.h"
#include "webnn/native/dml/ContextDML.h"

namespace webnn::native::dml {
    HRESULT Backend::EnumAdapter(DXGI_GPU_PREFERENCE gpuPreference) {
        Adapter adapter;
        ComPtr<IDXGIFactory6> dxgiFactory;
        WEBNN_RETURN_IF_FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory)));
        UINT adapterIndex = 0;
        while (dxgiFactory->EnumAdapterByGpuPreference(adapterIndex++, gpuPreference,
                                                       IID_PPV_ARGS(&adapter.adapter)) !=
               DXGI_ERROR_NOT_FOUND) {
            DXGI_ADAPTER_DESC1 pDesc;
            adapter.adapter->GetDesc1(&pDesc);
            // An adapter called the "Microsoft Basic Render Driver" is always present.
            // This adapter is a render-only device that has no display outputs. See here
            // for documentation on filtering WARP adapter:
            // https://docs.microsoft.com/en-us/windows/desktop/direct3ddxgi/d3d10-graphics-programming-guide-dxgi#new-info-about-enumerating-adapters-for-windows-8
            bool isSoftwareAdapter = pDesc.Flags == DXGI_ADAPTER_FLAG_SOFTWARE ||
                                     (pDesc.VendorId == 0x1414 && pDesc.DeviceId == 0x8c);
            if (!isSoftwareAdapter) {
                break;
            }
        }

        // Create the D3D device.
        WEBNN_RETURN_IF_FAILED(D3D12CreateDevice(adapter.adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                                 IID_PPV_ARGS(&adapter.D3D12Device)));

        // Create the DirectML device.
        ComPtr<ID3D12DebugDevice> debugDevice;
        if (mUseDebugLayer && SUCCEEDED(adapter.D3D12Device.As(&debugDevice))) {
            WEBNN_RETURN_IF_FAILED(DMLCreateDevice(adapter.D3D12Device.Get(),
                                                   DML_CREATE_DEVICE_FLAG_DEBUG,
                                                   IID_PPV_ARGS(&adapter.DMLDevice)));
        } else {
            WEBNN_RETURN_IF_FAILED(DMLCreateDevice(adapter.D3D12Device.Get(),
                                                   DML_CREATE_DEVICE_FLAG_NONE,
                                                   IID_PPV_ARGS(&adapter.DMLDevice)));
        }

        D3D12_COMMAND_QUEUE_DESC commandQueueDesc{};
        commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        commandQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        WEBNN_RETURN_IF_FAILED(adapter.D3D12Device->CreateCommandQueue(
            &commandQueueDesc, IID_PPV_ARGS(&adapter.commandQueue)));
        if (adapter.adapter) {
            mAdapters[gpuPreference] = adapter;
        }
        return S_OK;
    }

    Backend::Backend(InstanceBase* instance)
        : BackendConnection(instance, wnn::BackendType::DirectML) {
    }

    MaybeError Backend::Initialize() {
        mUseDebugLayer = false;
#ifdef _DEBUG
        mUseDebugLayer = true;
#endif
        if (mUseDebugLayer) {
            ComPtr<ID3D12Debug> debug;
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
                debug->EnableDebugLayer();
            }
        }

        ComPtr<IDXGIAdapter1> adapter;
        EnumAdapter(DXGI_GPU_PREFERENCE::DXGI_GPU_PREFERENCE_UNSPECIFIED);
        EnumAdapter(DXGI_GPU_PREFERENCE::DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE);
        EnumAdapter(DXGI_GPU_PREFERENCE::DXGI_GPU_PREFERENCE_MINIMUM_POWER);
        return {};
    }

    ContextBase* Backend::CreateContext(ContextOptions const* options) {
        wnn::DevicePreference devicePreference =
            options == nullptr ? wnn::DevicePreference::Default : options->devicePreference;
        bool useGpu = devicePreference == wnn::DevicePreference::Cpu ? false : true;
        if (!useGpu) {
            dawn::ErrorLog() << "Only support to create Context with Gpu.";
            return nullptr;
        }

        Adapter adapter;
        wnn::PowerPreference powerPreference =
            options == nullptr ? wnn::PowerPreference::Default : options->powerPreference;
        switch (powerPreference) {
            case wnn::PowerPreference::High_performance:
                if (mAdapters.find(DXGI_GPU_PREFERENCE::DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE) !=
                    mAdapters.end()) {
                    adapter = mAdapters[DXGI_GPU_PREFERENCE::DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE];
                }
                break;
            case wnn::PowerPreference::Low_power:
                if (mAdapters.find(DXGI_GPU_PREFERENCE::DXGI_GPU_PREFERENCE_MINIMUM_POWER) !=
                    mAdapters.end()) {
                    adapter = mAdapters[DXGI_GPU_PREFERENCE::DXGI_GPU_PREFERENCE_MINIMUM_POWER];
                }
                break;
            default:
                if (mAdapters.find(DXGI_GPU_PREFERENCE::DXGI_GPU_PREFERENCE_UNSPECIFIED) !=
                    mAdapters.end()) {
                    adapter = mAdapters[DXGI_GPU_PREFERENCE::DXGI_GPU_PREFERENCE_UNSPECIFIED];
                } else {
                    dawn::ErrorLog() << "Failed to create the context with none adapter.";
                    return nullptr;
                }
                break;
        }

        return Context::Create(adapter.DMLDevice, adapter.D3D12Device, adapter.commandQueue);
    }

    BackendConnection* Connect(InstanceBase* instance) {
        Backend* backend = new Backend(instance);

        if (instance->ConsumedError(backend->Initialize())) {
            delete backend;
            return nullptr;
        }

        return backend;
    }

}  // namespace webnn::native::dml
