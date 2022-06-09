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

#ifndef WEBNN_NATIVE_DEVICEDML_H_
#define WEBNN_NATIVE_DEVICEDML_H_

#include <webnn/webnn_cpp.h>
#include <unordered_map>

#include "UtilsDML.h"
#include "webnn/native/NamedOutputs.h"
#include "webnn/native/webnn_platform.h"

namespace webnn::native::dml {

    inline D3D12_HEAP_PROPERTIES CreateHeapProperties(
        D3D12_HEAP_TYPE type = D3D12_HEAP_TYPE_DEFAULT) {
        return {type, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
    };

    inline D3D12_RESOURCE_DESC CreateResourceDesc(
        UINT64 width,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
        return {D3D12_RESOURCE_DIMENSION_BUFFER, 0,    width, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                D3D12_TEXTURE_LAYOUT_ROW_MAJOR,  flags};
    };

    // An adapter called the "Microsoft Basic Render Driver" is always present. This adapter is a
    // render-only device that has no display outputs.
    bool IsWarpAdapter(IDXGIAdapter1* pAdapter);

    uint64_t RoundUpToMultiple(uint64_t value, uint64_t multiple);

    struct DeviceDescriptor {
        DXGI_GPU_PREFERENCE gpuPreference;
        bool useGpu = true;
        bool useDebugLayer = false;
    };

    class Device {
      public:
        static std::unique_ptr<Device> Create(DeviceDescriptor desc);
        HRESULT InitializeGraph(ComPtr<IDMLCompiledOperator>& compiledOperator,
                                const std::vector<std::shared_ptr<InputNode>>& inputNodes,
                                const std::vector<Node>& outputNodes);
        HRESULT ExecuteGraph(ComPtr<IDMLCompiledOperator>& compiledOperator,
                             const std::vector<std::shared_ptr<InputNode>>& inputNodes,
                             const std::vector<Node>& outputNodes,
                             std::unordered_map<std::string, Input> namedInputs,
                             std::unordered_map<std::string, Resource> namedOutputs);
        IDMLDevice* GetIDMLDevice() {
            return mDevice.Get();
        }

      private:
        Device(DeviceDescriptor desc);
        HRESULT Init();
        HRESULT CreateResourcesForCompiledOperatorInitializer();
        HRESULT CreateResourcesForCompiledOperator();
        void BindTemporaryResource(bool bindForInitializer = true);
        void BindPersistentResource(bool bindForInitializer = true);
        void CopyBufferRegion(ComPtr<ID3D12Resource> srcResource,
                              ComPtr<ID3D12Resource> destResource,
                              UINT64 resourceSize,
                              D3D12_RESOURCE_STATES state,
                              bool needBarrierEnd = true);
        HRESULT FillUploadResourceAndInputBindings(
            std::vector<DML_BUFFER_BINDING>& inputBufferBinding,
            const std::vector<std::shared_ptr<InputNode>>& inputNodes,
            std::unordered_map<std::string, Input> namedInputs = {});
        HRESULT CloseExecuteResetWait();

        ComPtr<IDMLDevice> mDevice;
        ComPtr<ID3D12Device> mD3D12Device;
        ComPtr<IDMLCommandRecorder> mCommandRecorder;
        ComPtr<ID3D12CommandQueue> mCommandQueue;
        ComPtr<ID3D12CommandAllocator> mCommandAllocator;
        ComPtr<ID3D12GraphicsCommandList> mCommandList;

        ComPtr<ID3D12DescriptorHeap> mDescriptorHeap;
        ComPtr<IDMLBindingTable> mBindingTable;
        DML_BINDING_TABLE_DESC mBindingTableDesc;

        ComPtr<ID3D12Resource> mUploadResource;
        ComPtr<ID3D12Resource> mInputResource;
        ComPtr<ID3D12Resource> mOutputResource;
        ComPtr<ID3D12Resource> mReadBackResource;
        ComPtr<ID3D12Resource> mTemporaryResource;
        ComPtr<ID3D12Resource> mPersistentResource;
        uint64_t mTemporaryResourceSize = 0;
        uint64_t mInitializedTemporaryResourceSize = 0;
        uint64_t mPersistentResourceSize = 0;
        uint64_t mConstantInputsResourceSize = 0;
        uint64_t mNonConstantInputsResourceSize = 0;
        uint64_t mOutputResourceSize = 0;

        DeviceDescriptor mDesc;
    };

}  // namespace webnn::native::dml

#endif  // WEBNN_NATIVE_DEVICEDML_H_
