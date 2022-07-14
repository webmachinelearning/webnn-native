//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#include "precomp.h"

#pragma warning(push)
#pragma warning(disable:4238)

using namespace pydml;
using Microsoft::WRL::ComPtr;

SVDescriptorHeap::SVDescriptorHeap(
    ComPtr<gpgmm::d3d12::Heap> heap)
    : m_Heap(std::move(heap)) {
}

ID3D12DescriptorHeap* SVDescriptorHeap::GetDescriptorHeap() const {
    ComPtr<ID3D12DescriptorHeap> descriptorHeap;
    m_Heap->As(&descriptorHeap);
    return descriptorHeap.Get();
}

Device::Device(bool useGpu, bool useDebugLayer, DXGI_GPU_PREFERENCE gpuPreference) : m_useGpu(useGpu), m_useDebugLayer(useDebugLayer), m_gpuPreference(gpuPreference) {}

// An adapter called the "Microsoft Basic Render Driver" is always present. This adapter is a render-only device that has no display outputs.
HRESULT IsWarpAdapter(IDXGIAdapter1* pAdapter, bool* isWarpAdapter) {
    DXGI_ADAPTER_DESC1 pDesc;
    ReturnIfFailed(pAdapter->GetDesc1(&pDesc));
    // see here for documentation on filtering WARP adapter:
    // https://docs.microsoft.com/en-us/windows/desktop/direct3ddxgi/d3d10-graphics-programming-guide-dxgi#new-info-about-enumerating-adapters-for-windows-8
    auto isBasicRenderDriverVendorId = pDesc.VendorId == 0x1414;
    auto isBasicRenderDriverDeviceId = pDesc.DeviceId == 0x8c;
    auto isSoftwareAdapter = pDesc.Flags == DXGI_ADAPTER_FLAG_SOFTWARE;
    *isWarpAdapter = isSoftwareAdapter || (isBasicRenderDriverVendorId && isBasicRenderDriverDeviceId);
    return S_OK;
}

HRESULT Device::Init()
{
    // 
    // Create D3D12 resources
    //

    if (m_useDebugLayer)
    {
        ComPtr<ID3D12Debug> debugController;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController))))
        {
            debugController->EnableDebugLayer();
        }
    }

    Microsoft::WRL::ComPtr<IDXGIAdapter1> dxgiAdapter;
    if(m_useGpu)
    {
        ComPtr<IDXGIFactory6> spFactory;
        ReturnIfFailed(CreateDXGIFactory1(IID_PPV_ARGS(&spFactory)));
        UINT i =   0;
        while ( spFactory->EnumAdapterByGpuPreference(i, m_gpuPreference, IID_PPV_ARGS(&dxgiAdapter))!= DXGI_ERROR_NOT_FOUND) {
            bool isWarpAdapter = false;
            ReturnIfFailed(IsWarpAdapter(dxgiAdapter.Get(), &isWarpAdapter));
            if (!isWarpAdapter) {
                break;
            }
            ++i;
        }
    }

    if (    !m_useGpu 
        ||  FAILED(D3D12CreateDevice(dxgiAdapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_d3d12Device))))
    {
        //If a computer's display driver is not functioning or is disabled, the computer's primary (NULL) adapter might also be called "Microsoft Basic Render Driver."
        Microsoft::WRL::ComPtr<IDXGIFactory4> dxgiFactory;
        ReturnIfFailed(CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory)));
        ReturnIfFailed(dxgiFactory->EnumWarpAdapter(IID_PPV_ARGS(&dxgiAdapter)));
        ReturnIfFailed(D3D12CreateDevice(dxgiAdapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_d3d12Device)));
    }

    // Get the hardware adapter used by the device.
    if (dxgiAdapter == nullptr){
        LUID adapterLUID = m_d3d12Device->GetAdapterLuid();
        Microsoft::WRL::ComPtr<IDXGIFactory1> dxgiFactory;
        ReturnIfFailed(CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory)));
        ComPtr<IDXGIFactory4> dxgiFactory4;
        ReturnIfFailed(dxgiFactory.As(&dxgiFactory4));
        dxgiFactory4->EnumAdapterByLuid(adapterLUID, IID_PPV_ARGS(&dxgiAdapter));
    }

    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    ReturnIfFailed(m_d3d12Device->CreateCommandQueue(&queueDesc, IID_GRAPHICS_PPV_ARGS(m_commandQueue.GetAddressOf())));

    ReturnIfFailed(m_d3d12Device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_COMPUTE,
        IID_GRAPHICS_PPV_ARGS(m_commandAllocator.GetAddressOf())));

    ReturnIfFailed(m_d3d12Device->CreateCommandList(
        0, // node mask
        D3D12_COMMAND_LIST_TYPE_COMPUTE,
        m_commandAllocator.Get(),
        nullptr, // initial pipeline state
        IID_GRAPHICS_PPV_ARGS(m_commandList.GetAddressOf())));

    D3D12_FEATURE_DATA_D3D12_OPTIONS options = {};
    ReturnIfFailed(m_d3d12Device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options)));

    gpgmm::d3d12::ALLOCATOR_DESC allocatorDesc = {};
    allocatorDesc.Adapter = dxgiAdapter;
    allocatorDesc.Device = m_d3d12Device;
    allocatorDesc.ResourceHeapTier = options.ResourceHeapTier;

#ifdef WEBNN_ENABLE_RESOURCE_DUMP
    allocatorDesc.RecordOptions.Flags |= gpgmm::d3d12::EVENT_RECORD_FLAG_ALL_EVENTS;
    allocatorDesc.RecordOptions.MinMessageLevel = D3D12_MESSAGE_SEVERITY_MESSAGE;
    allocatorDesc.RecordOptions.UseDetailedTimingEvents = true;
#endif

    ReturnIfFailed(gpgmm::d3d12::ResourceAllocator::CreateAllocator(allocatorDesc, &m_resourceAllocator, &m_residencyManager));

    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc = {};
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptorHeapDesc.NumDescriptors = 4; // One each for the input, output, persistent, and temporary resources
    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    ReturnIfFailed(m_d3d12Device->CreateDescriptorHeap(&descriptorHeapDesc, IID_GRAPHICS_PPV_ARGS(m_clearUavDescriptorHeapCpu.GetAddressOf())));

    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ReturnIfFailed(m_d3d12Device->CreateDescriptorHeap(&descriptorHeapDesc, IID_GRAPHICS_PPV_ARGS(m_clearUavDescriptorHeapGpu.GetAddressOf())));


    // 
    // Create DML resources
    //

    // DMLCreateDevice1 is supported since DML 1.1.0 and dml_resource_version specified by download_dml.py must exceed this version.
    // TODO: Consider relaxing DML_FEATURE_LEVEL_3_0 requirement to support more hardware.
    if (    !m_useDebugLayer 
        ||  FAILED(DMLCreateDevice1(m_d3d12Device.Get(), DML_CREATE_DEVICE_FLAG_DEBUG, DML_FEATURE_LEVEL_3_0, IID_PPV_ARGS(&m_dmlDevice))))
    {
        ReturnIfFailed(DMLCreateDevice1(m_d3d12Device.Get(), DML_CREATE_DEVICE_FLAG_NONE, DML_FEATURE_LEVEL_3_0, IID_PPV_ARGS(&m_dmlDevice)));
    }

    ReturnIfFailed(m_dmlDevice->CreateCommandRecorder(IID_PPV_ARGS(&m_commandRecorder)));
    ReturnIfFailed(m_dmlDevice->CreateOperatorInitializer(0, nullptr, IID_PPV_ARGS(&m_operatorInitializer)));
    ReturnIfFailed(m_dmlDevice->CreateBindingTable(nullptr, IID_PPV_ARGS(&m_bindingTable)));
    return S_OK;
}

HRESULT Device::DispatchOperator(
    IDMLCompiledOperator* op,
    const std::vector<pydml::Binding*>& inputs,
    const std::vector<dml::Expression*>& outputs,
    std::vector<pydml::TensorData*>& outputData
    )
{
    std::vector<DmlBufferBinding> inputBindings(inputs.size());
    uint64_t inputsResourceSize = 0;

    for (size_t i = 0; i < inputs.size(); ++i)
    {
        auto input = inputs[i];

        if (!input)
        {
            continue; // null optional tensor
        }

        DmlBufferTensorDesc desc = *input->desc.AsPtr<DML_BUFFER_TENSOR_DESC>();

        // If OWNED_BY_DML is *not* set, this input must be bound at execution
        // fix clang error: logical not is only applied to the left hand side of this bitwise operator [-Werror,-Wlogical-not-parentheses]
        if (!(desc.flags & DML_TENSOR_FLAG_OWNED_BY_DML))
        {
            uint32_t requiredAlignment = std::max(desc.guaranteedBaseOffsetAlignment, DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);

            // Bind to the end of the inputs resource (with appropriate alignment)
            inputBindings[i].offset = RoundUpToMultiple(inputsResourceSize, (uint64_t)requiredAlignment);
            inputBindings[i].sizeInBytes = desc.totalTensorSizeInBytes;

            inputsResourceSize = inputBindings[i].offset + desc.totalTensorSizeInBytes;
        }
    }

    std::vector<DmlBufferBinding> outputBindings(outputs.size());
    uint64_t outputsResourceSize = 0;

    for (size_t i = 0; i < outputs.size(); ++i)
    {
        auto output = outputs[i];

        if (!output)
        {
            continue; // null optional tensor
        }

        dml::TensorDesc desc = output->GetOutputDesc();
        DmlBufferTensorDesc bufferDesc = *desc.AsPtr<DML_BUFFER_TENSOR_DESC>();

        uint32_t requiredAlignment = std::max(bufferDesc.guaranteedBaseOffsetAlignment, DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);

        // Bind to the end of the outputs resource (with appropriate alignment)
        outputBindings[i].offset = RoundUpToMultiple(outputsResourceSize, (uint64_t)requiredAlignment);
        outputBindings[i].sizeInBytes = bufferDesc.totalTensorSizeInBytes;

        outputsResourceSize = outputBindings[i].offset + outputBindings[i].sizeInBytes;
    }

    DML_BINDING_PROPERTIES bindingProps = op->GetBindingProperties();

    ReturnIfFailed(EnsureUploadHeapSize(inputsResourceSize));
    ReturnIfFailed(EnsureCpuOrDefaultBufferSize(inputsResourceSize, m_inputsResource));
    ReturnIfFailed(EnsureReadBackHeapSize(outputsResourceSize));
    ReturnIfFailed(EnsureCpuOrDefaultBufferSize(outputsResourceSize, m_outputsResource));
    ReturnIfFailed(EnsureDefaultBufferSize(bindingProps.TemporaryResourceSize, m_temporaryResource));
    ReturnIfFailed(EnsureDescriptorHeapSize(bindingProps.RequiredDescriptorCount));

    // Set up input and output bindings to point to their respective buffers
    for (auto& binding : inputBindings)
    {
        if (binding.sizeInBytes != 0)
        {
            binding.buffer = m_inputsResource->GetResource();
        }
    }

    for (auto& binding : outputBindings)
    {
        if (binding.sizeInBytes != 0)
        {
            binding.buffer = m_outputsResource->GetResource();
        }
    }

    // The persistent resource should have already been initialized when the operator was initialized
    assert(m_persistentResource->GetResource()->GetDesc().Width >= bindingProps.PersistentResourceSize);

    // Upload inputs for execution
    std::vector<ID3D12Resource*> buffersToClear =
    {
        m_inputsResource->GetResource(),
        m_temporaryResource->GetResource(),
        m_outputsResource->GetResource()
    };
    
    ReturnIfFailed(ClearGpuBuffers(buffersToClear));

    if (inputsResourceSize)
    {
        // Copy the data into the upload heap
        byte* uploadHeapData = nullptr;

        ReturnIfFailed(m_uploadHeap->Map(0, nullptr, reinterpret_cast<void**>(&uploadHeapData)));

        for (size_t i = 0; i < inputs.size(); ++i)
        {
            if (!inputBindings[i].buffer)
            {
                // This input tensor doesn't need to be bound for initialize
                continue;
            }

            DmlBufferTensorDesc bufferDesc = *inputs[i]->desc.AsPtr<DML_BUFFER_TENSOR_DESC>();

            void* dest = uploadHeapData + inputBindings[i].offset;
            const void* src = inputs[i]->data.Get();

            assert(inputs[i]->data.Size() == bufferDesc.totalTensorSizeInBytes);

            memcpy(dest, src, static_cast<size_t>(bufferDesc.totalTensorSizeInBytes));
        }

        m_uploadHeap->Unmap(0, nullptr);

        // Record the copy from the upload heap into the inputs resource
        m_commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                m_inputsResource->GetResource(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_DEST)
            );

        m_commandList->CopyBufferRegion(m_inputsResource->GetResource(), 0, m_uploadHeap->GetResource(), 0, inputsResourceSize);

        m_commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                m_inputsResource->GetResource(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            );
    }

    // Bind for execution
    DmlTypeConverter<1024> converter;

    DML_BINDING_TABLE_DESC bindingTableDesc = {};
    bindingTableDesc.Dispatchable = op;
    bindingTableDesc.CPUDescriptorHandle = m_descriptorHeap->GetDescriptorHeap()->GetCPUDescriptorHandleForHeapStart();
    bindingTableDesc.GPUDescriptorHandle = m_descriptorHeap->GetDescriptorHeap()->GetGPUDescriptorHandleForHeapStart();
    bindingTableDesc.SizeInDescriptors = bindingProps.RequiredDescriptorCount;

    ReturnIfFailed(m_bindingTable->Reset(&bindingTableDesc));

    // Bind inputs
    std::vector<DML_BINDING_DESC> inputBindingDescs(inputBindings.size());
    for (size_t i = 0; i < inputBindings.size(); ++i)
    {
        inputBindingDescs[i] = converter.ToBindingDesc(inputBindings[i]);
    }

    m_bindingTable->BindInputs(static_cast<uint32_t>(inputBindingDescs.size()), inputBindingDescs.data());

    // Bind outputs
    std::vector<DML_BINDING_DESC> outputBindingDescs(outputBindings.size());
    for (size_t i = 0; i < outputBindings.size(); ++i)
    {
        outputBindingDescs[i] = converter.ToBindingDesc(outputBindings[i]);
    }

    m_bindingTable->BindOutputs(static_cast<uint32_t>(outputBindingDescs.size()), outputBindingDescs.data());

    // Bind persistent/temporary resources
    if (bindingProps.PersistentResourceSize != 0)
    {
        DML_BUFFER_BINDING persistentBinding = { m_persistentResource->GetResource(), 0, bindingProps.PersistentResourceSize };
        auto bindingDesc = DML_BINDING_DESC { DML_BINDING_TYPE_BUFFER, &persistentBinding };
        m_bindingTable->BindPersistentResource(&bindingDesc);
    }

    if (bindingProps.TemporaryResourceSize != 0)
    {
        DML_BUFFER_BINDING temporaryBinding = { m_temporaryResource->GetResource(), 0, bindingProps.TemporaryResourceSize };
        auto bindingDesc = DML_BINDING_DESC { DML_BINDING_TYPE_BUFFER, &temporaryBinding };
        m_bindingTable->BindTemporaryResource(&bindingDesc);
    }

    // Record and execute commands, and wait for completion
    ID3D12DescriptorHeap* descriptorHeap = m_descriptorHeap->GetDescriptorHeap();
    m_commandList->SetDescriptorHeaps(1, &descriptorHeap);
    m_commandRecorder->RecordDispatch(m_commandList.Get(), op, m_bindingTable.Get());
    RecordOutputReadBack(outputsResourceSize);
    ReturnIfFailed(ExecuteCommandListAndWait());

    // Read the output data back from the readback heap
    ReturnIfFailed(DownloadFromReadBackHeap(outputsResourceSize, outputs, outputBindings, outputData));
    return S_OK;
}

void Device::RecordOutputReadBack(uint64_t outputsResourceSize)
{
    // Copy output to readback heap
    if (outputsResourceSize != 0)
    {
        m_commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                m_outputsResource->GetResource(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE)
            );

        m_commandList->CopyBufferRegion(m_readbackHeap->GetResource(), 0, m_outputsResource->GetResource(), 0, outputsResourceSize);

        m_commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                m_outputsResource->GetResource(),
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            );
    }
}

 HRESULT Device::DownloadFromReadBackHeap(
    uint64_t outputsResourceSize, 
    const std::vector<dml::Expression*>& outputs,
    const std::vector<DmlBufferBinding>& outputBindings,
    std::vector<pydml::TensorData*>& outputData
    )
{
    if (outputsResourceSize != 0)
    {
        CD3DX12_RANGE readRange(0, static_cast<size_t>(outputsResourceSize));

        byte* readbackHeapData = nullptr;

        ReturnIfFailed(m_readbackHeap->Map(0, &readRange, reinterpret_cast<void**>(&readbackHeapData)));

        for (size_t i = 0; i < outputs.size(); ++i)
        {
            auto output = outputs[i];

            if (!output)
            {
                // This output tensor is optional (and null)
                continue;
            }

            dml::TensorDesc desc = output->GetOutputDesc();
            DmlBufferTensorDesc bufferDesc = *desc.AsPtr<DML_BUFFER_TENSOR_DESC>();

            auto data = new TensorData(&desc);
            void* dest = data->Get();
            const void* src = readbackHeapData + outputBindings[i].offset;

            memcpy(dest, src, static_cast<size_t>(bufferDesc.totalTensorSizeInBytes));

            outputData.push_back(data);
        }

        m_readbackHeap->Unmap(0, nullptr);
    }

    return S_OK;
}

HRESULT Device::InitializeOperator(
    IDMLCompiledOperator* op,
    const std::vector<pydml::Binding*>& inputs
    )
{
    // Allocate resources for initialization
    ReturnIfFailed(m_operatorInitializer->Reset(1, &op));

    DmlBufferArrayBinding inputBinding;
    inputBinding.bindings.resize(inputs.size());

    // Fill in the offsets and sizes for each binding, which will also tell us how big we need to make our buffer
    uint64_t inputsResourceSize = 0;

    for (size_t i = 0; i < inputs.size(); ++i)
    {
        auto input = inputs[i];

        if (!input)
        {
            continue; // null optional tensor
        }

        DmlBufferTensorDesc bufferDesc = *input->desc.AsPtr<DML_BUFFER_TENSOR_DESC>();

        // If OWNED_BY_DML is set, this input must be bound at initialize
        if (bufferDesc.flags & DML_TENSOR_FLAG_OWNED_BY_DML)
        {
            uint32_t requiredAlignment = std::max(bufferDesc.guaranteedBaseOffsetAlignment, DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);

            // Bind to the end of the inputs resource (with appropriate alignment)
            inputBinding.bindings[i].offset = RoundUpToMultiple(inputsResourceSize, (uint64_t)requiredAlignment);
            inputBinding.bindings[i].sizeInBytes = bufferDesc.totalTensorSizeInBytes;

            inputsResourceSize = inputBinding.bindings[i].offset + bufferDesc.totalTensorSizeInBytes;
        }
    }

    uint64_t temporaryResourceSize = m_operatorInitializer->GetBindingProperties().TemporaryResourceSize;
    uint64_t persistentResourceSize = op->GetBindingProperties().PersistentResourceSize;
    uint32_t descriptorHeapSize = m_operatorInitializer->GetBindingProperties().RequiredDescriptorCount;

    ReturnIfFailed(EnsureUploadHeapSize(inputsResourceSize));
    ReturnIfFailed(EnsureCpuOrDefaultBufferSize(inputsResourceSize, m_inputsResource));
    ReturnIfFailed(EnsureDefaultBufferSize(temporaryResourceSize, m_temporaryResource));
    ReturnIfFailed(EnsureDefaultBufferSize(persistentResourceSize, m_persistentResource));
    ReturnIfFailed(EnsureDescriptorHeapSize(descriptorHeapSize));

    // Set up the bindings to point to our input resource
    for (auto& binding : inputBinding.bindings)
    {
        if (binding.sizeInBytes != 0)
        {
            binding.buffer = m_inputsResource->GetResource();
        }
    }

    // Upload inputs for initialization
    std::vector<ID3D12Resource*> buffersToClear =
    {
        m_inputsResource->GetResource(),
        m_temporaryResource->GetResource(),
        m_persistentResource->GetResource()
    };

    ReturnIfFailed(ClearGpuBuffers(buffersToClear));

    if (inputsResourceSize)
    {
        // Copy the data into the upload heap
        byte* uploadHeapData = nullptr;

        ReturnIfFailed(m_uploadHeap->Map(0, nullptr, reinterpret_cast<void**>(&uploadHeapData)));

        for (size_t i = 0; i < inputs.size(); ++i)
        {
            if (!inputBinding.bindings[i].buffer)
            {
                // This input tensor doesn't need to be bound for initialize
                continue;
            }

            DmlBufferTensorDesc bufferDesc = *inputs[i]->desc.AsPtr<DML_BUFFER_TENSOR_DESC>();

            void* dest = uploadHeapData + inputBinding.bindings[i].offset;
            const void* src = inputs[i]->data.Get();

            assert(inputs[i]->data.Size() == bufferDesc.totalTensorSizeInBytes);

            memcpy(dest, src, static_cast<size_t>(bufferDesc.totalTensorSizeInBytes));
        }

        m_uploadHeap->Unmap(0, nullptr);

        // Record the copy from the upload heap into the inputs resource
        m_commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                m_inputsResource->GetResource(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_DEST)
            );

        m_commandList->CopyBufferRegion(m_inputsResource->GetResource(), 0, m_uploadHeap->GetResource(), 0, inputsResourceSize);

        m_commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                m_inputsResource->GetResource(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
                );
    }

    // Bind for initialization
    DmlTypeConverter<1024> converter;

    DML_BINDING_TABLE_DESC bindingTableDesc = {};
    bindingTableDesc.Dispatchable = m_operatorInitializer.Get();
    bindingTableDesc.CPUDescriptorHandle = m_descriptorHeap->GetDescriptorHeap()->GetCPUDescriptorHandleForHeapStart();
    bindingTableDesc.GPUDescriptorHandle = m_descriptorHeap->GetDescriptorHeap()->GetGPUDescriptorHandleForHeapStart();
    bindingTableDesc.SizeInDescriptors = descriptorHeapSize;

    ReturnIfFailed(m_bindingTable->Reset(&bindingTableDesc));

    DML_BINDING_DESC inputBindingDesc = converter.ToBindingDesc(inputBinding);
    m_bindingTable->BindInputs(1, &inputBindingDesc);

    if (persistentResourceSize != 0)
    {
        DML_BUFFER_BINDING outputBinding = { m_persistentResource->GetResource(), 0, persistentResourceSize };
        auto desc = DML_BINDING_DESC { DML_BINDING_TYPE_BUFFER, &outputBinding };
        m_bindingTable->BindOutputs(1, &desc);
    }

    if (temporaryResourceSize != 0)
    {
        DML_BUFFER_BINDING temporaryBinding = { m_temporaryResource->GetResource(), 0, temporaryResourceSize };
        auto desc = DML_BINDING_DESC { DML_BINDING_TYPE_BUFFER, &temporaryBinding };
        m_bindingTable->BindTemporaryResource(&desc);
    }

    // Record and execute commands, and wait for completion
    ID3D12DescriptorHeap* descriptorHeap = m_descriptorHeap->GetDescriptorHeap();
    m_commandList->SetDescriptorHeaps(1, &descriptorHeap);
    m_commandRecorder->RecordDispatch(m_commandList.Get(), m_operatorInitializer.Get(), m_bindingTable.Get());
    ReturnIfFailed(ExecuteCommandListAndWait());
    return S_OK;
}

HRESULT Device::ExecuteCommandListAndWait()
{
    ReturnIfFailed(m_commandList->Close());

    ID3D12CommandList* commandLists[] = { m_commandList.Get() };
    if (m_residencyManager != nullptr){
        gpgmm::d3d12::ResidencySet* residencySets[] = { &m_residencySet };
        m_residencyManager->ExecuteCommandLists(m_commandQueue.Get(), commandLists, residencySets, ARRAYSIZE(commandLists));
    } else {
        m_commandQueue->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);
    }

    WaitForQueueToComplete(m_commandQueue.Get());

    ReturnIfFailed(m_commandAllocator->Reset());
    ReturnIfFailed(m_commandList->Reset(m_commandAllocator.Get(), nullptr));
    ReturnIfFailed(m_residencySet.Reset());
    return S_OK;
}

HRESULT Device::EnsureUploadHeapSize(uint64_t requestedSizeInBytes)
{
    uint64_t existingSize = m_uploadHeap ? m_uploadHeap->GetResource()->GetDesc().Width : 0;
    uint64_t newSize = RoundUpToPow2(requestedSizeInBytes);     // ensures geometric growth
    newSize = std::max(newSize, static_cast<uint64_t>(65536));  // Minimum size of 64k

    if (newSize != existingSize)
    {
        m_uploadHeap = nullptr;
        ReturnIfFailed(CreateResource(
            m_resourceAllocator.Get(),
            CD3DX12_RESOURCE_DESC::Buffer(newSize),
            CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            m_uploadHeap
            ));
    }
    return S_OK;
}

HRESULT Device::EnsureCpuOrDefaultBufferSize(uint64_t requestedSizeInBytes, _Inout_ ComPtr<gpgmm::d3d12::ResourceAllocation>& buffer)
{
    if (m_useCpuCustomHeapResources)
    {
        ReturnIfFailed(EnsureCpuBufferSize(requestedSizeInBytes, buffer));
    }
    else
    {
        ReturnIfFailed(EnsureDefaultBufferSize(requestedSizeInBytes, buffer));
    }
    return S_OK;
}

HRESULT Device::EnsureCpuBufferSize(uint64_t requestedSizeInBytes, _Inout_ ComPtr<gpgmm::d3d12::ResourceAllocation>& buffer)
{
    uint64_t existingSize = buffer ? buffer->GetResource()->GetDesc().Width : 0;
    uint64_t newSize = RoundUpToPow2(requestedSizeInBytes);     // ensures geometric growth
    newSize = std::max(newSize, static_cast<uint64_t>(65536));  // Minimum size of 64k

    if (newSize != existingSize)
    {
        buffer = nullptr;
        ReturnIfFailed(CreateCpuCustomBuffer(m_resourceAllocator.Get(), newSize, buffer));
    }

    UpdateResidencyIfNeeded(buffer.Get(), &m_residencySet);
    
    return S_OK;
}

HRESULT Device::EnsureDefaultBufferSize(uint64_t requestedSizeInBytes, _Inout_ ComPtr<gpgmm::d3d12::ResourceAllocation>& buffer)
{
    uint64_t existingSize = buffer ? buffer->GetResource()->GetDesc().Width : 0;
    uint64_t newSize = RoundUpToPow2(requestedSizeInBytes);     // ensures geometric growth
    newSize = std::max(newSize, static_cast<uint64_t>(65536));  // Minimum size of 64k

    if (newSize != existingSize)
    {
        buffer = nullptr;
        ReturnIfFailed(CreateDefaultBuffer(m_resourceAllocator.Get(), newSize, buffer));
    }

    UpdateResidencyIfNeeded(buffer.Get(), &m_residencySet);
    
    return S_OK;
}

HRESULT Device::EnsureDescriptorHeapSize(uint32_t requestedSizeInDescriptors)
{
    uint32_t existingSize = m_descriptorHeap ? m_descriptorHeap->GetDescriptorHeap()->GetDesc().NumDescriptors : 0;
    uint32_t newSize = RoundUpToPow2(requestedSizeInDescriptors); // ensures geometric growth

    if (newSize != existingSize)
    {
        if (m_descriptorHeap != nullptr && m_residencyManager != nullptr){
            m_residencyManager->UnlockHeap(m_descriptorHeap->m_Heap.Get());
        }

        m_descriptorHeap = nullptr;
        
        auto createHeapFn = [&](ID3D12Pageable** ppPageableOut) -> HRESULT {
            ComPtr<ID3D12DescriptorHeap> d3d12DescriptorHeap;
            D3D12_DESCRIPTOR_HEAP_DESC desc = {};
            desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            desc.NumDescriptors = newSize;
            desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            ReturnIfFailed(m_d3d12Device->CreateDescriptorHeap(
                &desc, IID_PPV_ARGS(&d3d12DescriptorHeap)));
            *ppPageableOut = d3d12DescriptorHeap.Detach();
            return S_OK;
        };

        gpgmm::d3d12::HEAP_DESC heapDesc = {};
        heapDesc.SizeInBytes = newSize * m_d3d12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        heapDesc.MemorySegment = gpgmm::d3d12::RESIDENCY_SEGMENT_LOCAL;

        ComPtr<gpgmm::d3d12::Heap> descriptorHeap;
        ReturnIfFailed(gpgmm::d3d12::Heap::CreateHeap(heapDesc, m_residencyManager.Get(), createHeapFn,
                                            &descriptorHeap));

        if (m_residencyManager != nullptr){
            ReturnIfFailed(m_residencyManager->LockHeap(descriptorHeap.Get()));
        }

        m_descriptorHeap = std::make_unique<SVDescriptorHeap>(std::move(descriptorHeap));
    }
    return S_OK;
}

HRESULT Device::EnsureReadBackHeapSize(uint64_t requestedSizeInBytes)
{
    uint64_t existingSize = m_readbackHeap ? m_readbackHeap->GetResource()->GetDesc().Width : 0;
    uint64_t newSize = RoundUpToPow2(requestedSizeInBytes); // ensures geometric growth
    newSize = std::max(newSize, static_cast<uint64_t>(65536)); // Minimum size of 64k

    if (newSize != existingSize)
    {
        m_readbackHeap = nullptr;
        ReturnIfFailed(CreateReadBackBuffer(m_resourceAllocator.Get(), newSize, m_readbackHeap));
    }

    UpdateResidencyIfNeeded(m_readbackHeap.Get(), &m_residencySet);
    
    return S_OK;
}

HRESULT Device::ClearGpuBuffers(dml::Span<ID3D12Resource*> buffers)
{
    static const uint32_t ClearValue = static_cast<uint32_t>(-1);

    // The number of buffers we can clear at once is limited by the size of our descriptor heap
    assert(static_cast<uint32_t>(buffers.size()) <= m_clearUavDescriptorHeapCpu->GetDesc().NumDescriptors);

    uint32_t descriptorOffset = 0;
    for (ID3D12Resource* buffer : buffers)
    {
        if (!buffer)
        {
            // Nothing to clear; these buffers are lazily-initialized
            continue;
        }

        ReturnIfFailed(FillGpuBuffer(
            m_commandList.Get(),
            m_clearUavDescriptorHeapCpu.Get(),
            m_clearUavDescriptorHeapGpu.Get(),
            descriptorOffset,
            buffer,
            ClearValue));

        ++descriptorOffset;
    }

    m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(nullptr));
    return S_OK;
}

#pragma warning(pop)