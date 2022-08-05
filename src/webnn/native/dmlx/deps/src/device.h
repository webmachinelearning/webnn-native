//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once

#include <gpgmm_d3d12.h>

namespace pydml
{
    class SVDescriptorHeap {
      public:
        SVDescriptorHeap(ComPtr<gpgmm::d3d12::Heap> heap);
        ID3D12DescriptorHeap* GetDescriptorHeap() const;

        ComPtr<gpgmm::d3d12::Heap> m_Heap;
    };

    class Device
    {
    public:
        explicit Device(bool useGpu = true, bool useDebugLayer = false, bool beginCaptureOnStartup = false, DXGI_GPU_PREFERENCE gpuPreference = DXGI_GPU_PREFERENCE::DXGI_GPU_PREFERENCE_UNSPECIFIED);

        HRESULT Init();

        inline bool UseGpu() const
        {
            return m_useGpu;
        }

        inline IDMLDevice* GetDevice() const
        {
            return m_dmlDevice.Get();
        }

        HRESULT InitializeOperator(
            IDMLCompiledOperator* op,
            const std::vector<pydml::Binding*>& inputs
            );

        HRESULT DispatchOperator(
            IDMLCompiledOperator* op,
            const std::vector<pydml::Binding*>& inputs,
            const std::vector<dml::Expression*>& outputs,
            std::vector<pydml::TensorData*>& outputData
            );

    protected:
        

        void RecordOutputReadBack(uint64_t outputsResourceSize);

        HRESULT DownloadFromReadBackHeap(
            uint64_t outputsResourceSize, 
            const std::vector<dml::Expression*>& outputs,
            const std::vector<DmlBufferBinding>& outputBindings,
            std::vector<pydml::TensorData*>& outputData
            );

        HRESULT EnsureUploadHeapSize(uint64_t requestedSizeInBytes);
        HRESULT EnsureReadBackHeapSize(uint64_t requestedSizeInBytes);
        HRESULT EnsureCpuOrDefaultBufferSize(uint64_t requestedSizeInBytes, _Inout_ Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation>& buffer);
        HRESULT EnsureCpuBufferSize(uint64_t requestedSizeInBytes, _Inout_ Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation>& buffer);
        HRESULT EnsureDefaultBufferSize(uint64_t requestedSizeInBytes, _Inout_ Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation>& buffer);
        HRESULT EnsureDescriptorHeapSize(uint32_t requestedSizeInDescriptors);

        HRESULT ClearGpuBuffers(dml::Span<ID3D12Resource*> buffers);

        HRESULT ExecuteCommandListAndWait();

        Microsoft::WRL::ComPtr<ID3D12Device> m_d3d12Device;
        Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_commandQueue;
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_commandAllocator;
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> m_commandList;
        Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocator> m_resourceAllocator;
        Microsoft::WRL::ComPtr<gpgmm::d3d12::ResidencyManager> m_residencyManager;

        // GPU- and CPU-visible descriptor heaps used for ClearUnorderedAccessView
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_clearUavDescriptorHeapGpu;
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_clearUavDescriptorHeapCpu;

        Microsoft::WRL::ComPtr<IDMLDevice> m_dmlDevice;
        Microsoft::WRL::ComPtr<IDMLCommandRecorder> m_commandRecorder;
        Microsoft::WRL::ComPtr<IDMLOperatorInitializer> m_operatorInitializer;
        Microsoft::WRL::ComPtr<IDMLBindingTable> m_bindingTable;

        // Lazily-initialized resources for operator initialization/execution
        std::unique_ptr<SVDescriptorHeap> m_descriptorHeap;
        Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation> m_uploadHeap;
        Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation> m_readbackHeap;

        // DEFAULT heap buffers to hold input tensors, output tensors, and temporary and persistent resources. The input
        // and output resources are suballocated for operators that have multiple inputs or outputs.
        Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation> m_inputsResource;
        Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation> m_outputsResource;
        Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation> m_temporaryResource;
        Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation> m_persistentResource;

        gpgmm::d3d12::ResidencyList m_residencyList;

        bool m_useCpuCustomHeapResources = false;
        bool m_useGpu = true;
        bool m_useDebugLayer = false;
        bool m_beginCaptureOnStartup = false;
        DXGI_GPU_PREFERENCE m_gpuPreference;
    };
}