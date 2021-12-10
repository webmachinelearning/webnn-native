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

#ifndef WEBNN_NATIVE_DMLUTILS_H_
#define WEBNN_NATIVE_DMLUTILS_H_

#include <webnn/webnn_cpp.h>

#include "common/Log.h"

#define WEBNN_CHECK(hr)                             \
    if (((HRESULT)(hr)) < 0) {                      \
        dawn::ErrorLog() << "Failed to do " << #hr; \
        DAWN_ASSERT(0);                             \
    }

namespace webnn_native { namespace dml { namespace utils {
    using namespace Microsoft::WRL;

    template <typename T>
    T RoundUpToMultiple(T value, T multiple) {
        static_assert(std::is_integral_v<T>);

        T remainder = value % multiple;
        if (remainder != 0) {
            value += multiple - remainder;
        }

        return value;
    }

    // An adapter called the "Microsoft Basic Render Driver" is always present. This adapter is a
    // render-only device that has no display outputs.
    HRESULT IsWarpAdapter(IDXGIAdapter1* pAdapter, bool* isWarpAdapter) {
        DXGI_ADAPTER_DESC1 pDesc;
        WEBNN_CHECK(pAdapter->GetDesc1(&pDesc));
        // see here for documentation on filtering WARP adapter:
        // https://docs.microsoft.com/en-us/windows/desktop/direct3ddxgi/d3d10-graphics-programming-guide-dxgi#new-info-about-enumerating-adapters-for-windows-8
        auto isBasicRenderDriverVendorId = pDesc.VendorId == 0x1414;
        auto isBasicRenderDriverDeviceId = pDesc.DeviceId == 0x8c;
        auto isSoftwareAdapter = pDesc.Flags == DXGI_ADAPTER_FLAG_SOFTWARE;
        *isWarpAdapter =
            isSoftwareAdapter || (isBasicRenderDriverVendorId && isBasicRenderDriverDeviceId);
        return S_OK;
    }

    void InitD3D12(ComPtr<ID3D12GraphicsCommandList>& commandList,
                   ComPtr<ID3D12CommandQueue>& commandQueue,
                   ComPtr<ID3D12CommandAllocator>& commandAllocator,
                   ComPtr<ID3D12Device>& D3D12Device,
                   DXGI_GPU_PREFERENCE gpuPreference,
                   bool useGpu) {
#if defined(_DEBUG)
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
            debug->EnableDebugLayer();
        }
#endif
        ComPtr<IDXGIAdapter1> dxgiAdapter;
        if (useGpu) {
            ComPtr<IDXGIFactory6> dxgiFactory;
            WEBNN_CHECK(CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory)));
            UINT i = 0;
            while (dxgiFactory->EnumAdapterByGpuPreference(
                       i++, gpuPreference, IID_PPV_ARGS(&dxgiAdapter)) != DXGI_ERROR_NOT_FOUND) {
                bool isWarpAdapter = false;
                WEBNN_CHECK(IsWarpAdapter(dxgiAdapter.Get(), &isWarpAdapter));
                if (!isWarpAdapter) {
                    break;
                }
            }
        }
        if (!useGpu || FAILED(D3D12CreateDevice(dxgiAdapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                                IID_PPV_ARGS(&D3D12Device)))) {
            // If a computer's display driver is not functioning or is disabled, the computer's
            // primary (NULL) adapter might also be called "Microsoft Basic Render Driver."
            ComPtr<IDXGIFactory4> dxgiFactory;
            WEBNN_CHECK(CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory)));
            WEBNN_CHECK(dxgiFactory->EnumWarpAdapter(IID_PPV_ARGS(&dxgiAdapter)));
            WEBNN_CHECK(D3D12CreateDevice(dxgiAdapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                          IID_PPV_ARGS(&D3D12Device)));
        }

        D3D12_COMMAND_QUEUE_DESC commandQueueDesc{};
        commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        commandQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        WEBNN_CHECK(
            D3D12Device->CreateCommandQueue(&commandQueueDesc, IID_PPV_ARGS(&commandQueue)));
        WEBNN_CHECK(D3D12Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                        IID_PPV_ARGS(&commandAllocator)));
        WEBNN_CHECK(D3D12Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                   commandAllocator.Get(), nullptr,
                                                   IID_PPV_ARGS(&commandList)));
    }

    void CloseExecuteResetWait(ComPtr<ID3D12GraphicsCommandList> commandList,
                               ComPtr<ID3D12CommandQueue> commandQueue,
                               ComPtr<ID3D12CommandAllocator> commandAllocator,
                               ComPtr<ID3D12Device> D3D12Device) {
        WEBNN_CHECK(commandList->Close());
        ID3D12CommandList* commandLists[] = {commandList.Get()};
        commandQueue->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);
        WEBNN_CHECK(commandQueue.Get()->GetDevice(IID_PPV_ARGS(D3D12Device.GetAddressOf())));
        ComPtr<ID3D12Fence> fence;
        WEBNN_CHECK(
            D3D12Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.GetAddressOf())));
        WEBNN_CHECK(commandQueue.Get()->Signal(fence.Get(), 1));
        WEBNN_CHECK(fence->SetEventOnCompletion(1, nullptr));
        WEBNN_CHECK(commandAllocator->Reset());
        WEBNN_CHECK(commandList->Reset(commandAllocator.Get(), nullptr));
    }

}}}  // namespace webnn_native::dml::utils

#endif  // WEBNN_NATIVE_DML_UTILS_H_