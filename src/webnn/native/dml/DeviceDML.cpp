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

#include "DeviceDML.h"

namespace webnn::native::dml {

    HRESULT Device::CloseExecuteResetWait() {
        RETURN_IF_FAILED(mCommandList->Close());
        ID3D12CommandList* mCommandLists[] = {mCommandList.Get()};
        mCommandQueue->ExecuteCommandLists(ARRAYSIZE(mCommandLists), mCommandLists);
        RETURN_IF_FAILED(mCommandQueue.Get()->GetDevice(IID_PPV_ARGS(mD3D12Device.GetAddressOf())));
        ComPtr<ID3D12Fence> fence;
        RETURN_IF_FAILED(mD3D12Device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                                   IID_PPV_ARGS(fence.GetAddressOf())));
        RETURN_IF_FAILED(mCommandQueue.Get()->Signal(fence.Get(), 1));
        RETURN_IF_FAILED(fence->SetEventOnCompletion(1, nullptr));
        RETURN_IF_FAILED(mCommandAllocator->Reset());
        RETURN_IF_FAILED(mCommandList->Reset(mCommandAllocator.Get(), nullptr));
        return S_OK;
    }

    Device::Device(DeviceDescriptor desc) : mDesc(std::move(desc)) {
    }

    std::unique_ptr<Device> Device::Create(DeviceDescriptor desc) {
        std::unique_ptr<Device> device(new Device(desc));
        if (FAILED(device->Init())) {
            dawn::ErrorLog() << "Failed to initialize Device.";
            return nullptr;
        }
        return device;
    }

    HRESULT Device::Init() {
        if (mDesc.useDebugLayer) {
            ComPtr<ID3D12Debug> debug;
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
                debug->EnableDebugLayer();
            }
        }

        ComPtr<IDXGIAdapter1> dxgiAdapter;
        if (mDesc.useGpu) {
            ComPtr<IDXGIFactory6> dxgiFactory;
            RETURN_IF_FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory)));
            UINT i = 0;
            while (dxgiFactory->EnumAdapterByGpuPreference(i++, mDesc.gpuPreference,
                                                           IID_PPV_ARGS(&dxgiAdapter)) !=
                   DXGI_ERROR_NOT_FOUND) {
                if (!IsWarpAdapter(dxgiAdapter.Get())) {
                    break;
                }
            }
        }
        if (!mDesc.useGpu || FAILED(D3D12CreateDevice(dxgiAdapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                                      IID_PPV_ARGS(&mD3D12Device)))) {
            // If a computer's display driver is not functioning or is disabled, the computer's
            // primary (NULL) adapter might also be called "Microsoft Basic Render Driver."
            ComPtr<IDXGIFactory4> dxgiFactory;
            RETURN_IF_FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory)));
            RETURN_IF_FAILED(dxgiFactory->EnumWarpAdapter(IID_PPV_ARGS(&dxgiAdapter)));
            RETURN_IF_FAILED(D3D12CreateDevice(dxgiAdapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                               IID_PPV_ARGS(&mD3D12Device)));
        }

        D3D12_COMMAND_QUEUE_DESC commandQueueDesc{};
        commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        commandQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        RETURN_IF_FAILED(
            mD3D12Device->CreateCommandQueue(&commandQueueDesc, IID_PPV_ARGS(&mCommandQueue)));
        RETURN_IF_FAILED(mD3D12Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                              IID_PPV_ARGS(&mCommandAllocator)));
        RETURN_IF_FAILED(mD3D12Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                         mCommandAllocator.Get(), nullptr,
                                                         IID_PPV_ARGS(&mCommandList)));

        // Create the DirectML device.
        DML_CREATE_DEVICE_FLAGS dmlCreateDeviceFlags = DML_CREATE_DEVICE_FLAG_NONE;
#if defined(_DEBUG)
        dmlCreateDeviceFlags = DML_CREATE_DEVICE_FLAG_DEBUG;
#endif
        if (dmlCreateDeviceFlags == DML_CREATE_DEVICE_FLAG_DEBUG) {
            if (FAILED(DMLCreateDevice(mD3D12Device.Get(), dmlCreateDeviceFlags,
                                       IID_PPV_ARGS(&mDevice)))) {
                dawn::WarningLog() << "Failed to create a DirectML device with debug flag, "
                                      "will fall back to use none flag.";
                RETURN_IF_FAILED(DMLCreateDevice(mD3D12Device.Get(), DML_CREATE_DEVICE_FLAG_NONE,
                                                 IID_PPV_ARGS(&mDevice)));
            }
        } else {
            RETURN_IF_FAILED(
                DMLCreateDevice(mD3D12Device.Get(), dmlCreateDeviceFlags, IID_PPV_ARGS(&mDevice)));
        }
        return S_OK;
    };

    // TODO(mingming): Consider to replace CreateCommittedResource with
    // ResourceAllocator::CreateResource of GPGMM.
    HRESULT Device::CreateResourcesForCompiledOperatorInitializer() {
        if (mConstantInputsResourceSize != 0) {
            RETURN_IF_FAILED(mD3D12Device->CreateCommittedResource(
                &CreateHeapProperties(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE,
                &CreateResourceDesc(mConstantInputsResourceSize), D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr, IID_PPV_ARGS(&mUploadResource)));

            RETURN_IF_FAILED(mD3D12Device->CreateCommittedResource(
                &CreateHeapProperties(), D3D12_HEAP_FLAG_NONE,
                &CreateResourceDesc(mConstantInputsResourceSize,
                                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&mInputResource)));
        }

        if (mTemporaryResourceSize != 0) {
            RETURN_IF_FAILED(mD3D12Device->CreateCommittedResource(
                &CreateHeapProperties(), D3D12_HEAP_FLAG_NONE,
                &CreateResourceDesc(mTemporaryResourceSize,
                                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&mTemporaryResource)));
        }

        if (mPersistentResourceSize != 0) {
            RETURN_IF_FAILED(mD3D12Device->CreateCommittedResource(
                &CreateHeapProperties(), D3D12_HEAP_FLAG_NONE,
                &CreateResourceDesc(mPersistentResourceSize,
                                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                IID_PPV_ARGS(&mPersistentResource)));
        }
        return S_OK;
    };

    // TODO(mingming): Consider to replace CreateCommittedResource with
    // ResourceAllocator::CreateResource of GPGMM.
    HRESULT Device::CreateResourcesForCompiledOperator() {
        if (mNonConstantInputsResourceSize) {
            // Release the upload resource and input resource which has been allocated for
            // initializing constant inputs and then re-allocate them with new size to prepare
            // for initializing common inputs.
            mUploadResource = nullptr;
            mInputResource = nullptr;
            RETURN_IF_FAILED(mD3D12Device->CreateCommittedResource(
                &CreateHeapProperties(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE,
                &CreateResourceDesc(mNonConstantInputsResourceSize),
                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&mUploadResource)));

            RETURN_IF_FAILED(mD3D12Device->CreateCommittedResource(
                &CreateHeapProperties(), D3D12_HEAP_FLAG_NONE,
                &CreateResourceDesc(mNonConstantInputsResourceSize,
                                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&mInputResource)));
        }

        if (mOutputResourceSize) {
            RETURN_IF_FAILED(mD3D12Device->CreateCommittedResource(
                &CreateHeapProperties(), D3D12_HEAP_FLAG_NONE,
                &CreateResourceDesc(mOutputResourceSize,
                                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&mOutputResource)));

            RETURN_IF_FAILED(mD3D12Device->CreateCommittedResource(
                &CreateHeapProperties(D3D12_HEAP_TYPE_READBACK), D3D12_HEAP_FLAG_NONE,
                &CreateResourceDesc(mOutputResourceSize), D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                IID_PPV_ARGS(&mReadBackResource)));
        }
        return S_OK;
    };

    void Device::BindTemporaryResource(bool bindForInitializer) {
        if (mTemporaryResourceSize != 0) {
            if ((bindForInitializer && mInitializedTemporaryResourceSize != 0) ||
                !bindForInitializer) {
                DML_BUFFER_BINDING bufferBinding{mTemporaryResource.Get(), 0,
                                                 mTemporaryResourceSize};
                DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
                mBindingTable->BindTemporaryResource(&bindingDesc);
            }
        }
    };

    void Device::BindPersistentResource(bool bindForInitializer) {
        if (mPersistentResourceSize != 0) {
            DML_BUFFER_BINDING bufferBinding{mPersistentResource.Get(), 0, mPersistentResourceSize};
            DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
            if (bindForInitializer) {
                mBindingTable->BindOutputs(1, &bindingDesc);
            } else {
                mBindingTable->BindPersistentResource(&bindingDesc);
            }
        }
    };

    void Device::CopyBufferRegion(ComPtr<ID3D12Resource> srcResource,
                                  ComPtr<ID3D12Resource> destResource,
                                  UINT64 resourceSize,
                                  D3D12_RESOURCE_STATES state,
                                  bool needBarrierEnd) {
        D3D12_RESOURCE_BARRIER resourceBarrier;
        if (state == D3D12_RESOURCE_STATE_COPY_DEST) {
            resourceBarrier.Transition.pResource = destResource.Get();
        } else if (state == D3D12_RESOURCE_STATE_COPY_SOURCE) {
            resourceBarrier.Transition.pResource = srcResource.Get();
        } else {
            dawn::ErrorLog() << "Unsupported D3D12_RESOURCE_STATES.";
            DAWN_ASSERT(0);
        }
        resourceBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        resourceBarrier.Transition.StateAfter = state;
        resourceBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        resourceBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        resourceBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        mCommandList->ResourceBarrier(1, &resourceBarrier);
        mCommandList->CopyBufferRegion(destResource.Get(), 0, srcResource.Get(), 0, resourceSize);
        if (needBarrierEnd) {
            resourceBarrier.Transition.StateBefore = state;
            resourceBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            mCommandList->ResourceBarrier(1, &resourceBarrier);
        }
    }

    HRESULT Device::FillUploadResourceAndInputBindings(
        std::vector<DML_BUFFER_BINDING>& inputBufferBinding,
        const std::vector<std::shared_ptr<InputNode>>& inputNodes,
        std::unordered_map<std::string, Input> namedInputs) {
        int8_t* uploadBuffer;
        RETURN_IF_FAILED(mUploadResource->Map(0, nullptr, reinterpret_cast<void**>(&uploadBuffer)));
        uint64_t offset = 0;
        for (size_t i = 0; i < inputNodes.size(); ++i) {
            auto inputNode = inputNodes[i];
            if (namedInputs.empty()) {
                if (inputNode->type == NodeType::ConstantInput) {
                    offset = RoundUpToMultiple(
                        offset, static_cast<uint64_t>(DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT));
                    inputBufferBinding[i].Buffer = mInputResource.Get();
                    inputBufferBinding[i].Offset = offset;
                    inputBufferBinding[i].SizeInBytes = inputNode->byteLength;
                    memcpy(uploadBuffer + offset, inputNode->buffer,
                           static_cast<size_t>(inputNode->byteLength));
                    offset = offset + inputNode->byteLength;
                }
            } else {
                if (inputNode->type == NodeType::NonConstantInput) {
                    offset = RoundUpToMultiple(
                        offset, static_cast<uint64_t>(DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT));
                    auto arrayBufferView = namedInputs[inputNode->name].resource.arrayBufferView;
                    inputBufferBinding[i].Buffer = mInputResource.Get();
                    inputBufferBinding[i].Offset = offset;
                    inputBufferBinding[i].SizeInBytes = arrayBufferView.byteLength;
                    memcpy(
                        uploadBuffer + offset,
                        static_cast<int8_t*>(arrayBufferView.buffer) + arrayBufferView.byteOffset,
                        arrayBufferView.byteLength);
                    offset = offset + arrayBufferView.byteLength;
                }
            }
        }
        mUploadResource->Unmap(0, nullptr);
        return S_OK;
    }

    HRESULT Device::InitializeGraph(ComPtr<IDMLCompiledOperator>& compiledOperator,
                                    const std::vector<std::shared_ptr<InputNode>>& inputNodes,
                                    const std::vector<Node>& outputNodes) {
        DAWN_ASSERT(compiledOperator != nullptr);
        IDMLCompiledOperator* compiledOperators[] = {compiledOperator.Get()};
        ComPtr<IDMLOperatorInitializer> compiledOperatorInitializer;
        RETURN_IF_FAILED(
            mDevice->CreateOperatorInitializer(ARRAYSIZE(compiledOperators), compiledOperators,
                                               IID_PPV_ARGS(&compiledOperatorInitializer)));

        DML_BINDING_PROPERTIES initializeBindingProperties =
            compiledOperatorInitializer->GetBindingProperties();
        DML_BINDING_PROPERTIES executeBindingProperties = compiledOperator->GetBindingProperties();
        UINT descriptorCount = std::max(initializeBindingProperties.RequiredDescriptorCount,
                                        executeBindingProperties.RequiredDescriptorCount);
        mInitializedTemporaryResourceSize = initializeBindingProperties.TemporaryResourceSize;
        mTemporaryResourceSize = std::max(mInitializedTemporaryResourceSize,
                                          executeBindingProperties.TemporaryResourceSize);
        mPersistentResourceSize = executeBindingProperties.PersistentResourceSize;

        // Describe and create a constant buffer view (CBV), Shader resource view (SRV), and
        // unordered access view (UAV) descriptor heap.
        D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc{};
        descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        descriptorHeapDesc.NumDescriptors = descriptorCount;
        descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        RETURN_IF_FAILED(mD3D12Device->CreateDescriptorHeap(&descriptorHeapDesc,
                                                            IID_PPV_ARGS(&mDescriptorHeap)));

        // Create a binding table over the descriptor heap we just created.
        mBindingTableDesc.Dispatchable = compiledOperatorInitializer.Get();
        mBindingTableDesc.CPUDescriptorHandle =
            mDescriptorHeap->GetCPUDescriptorHandleForHeapStart();
        mBindingTableDesc.GPUDescriptorHandle =
            mDescriptorHeap->GetGPUDescriptorHandleForHeapStart();
        // The size of the binding table, in descriptors. This is the maximum number of
        // descriptors that DirectML is permitted to write, from the start of both the
        // supplied CPU and GPU descriptor handles.
        mBindingTableDesc.SizeInDescriptors = descriptorCount;
        RETURN_IF_FAILED(
            mDevice->CreateBindingTable(&mBindingTableDesc, IID_PPV_ARGS(&mBindingTable)));

        // Initialize constant inputs.
        for (auto& inputNode : inputNodes) {
            if (inputNode->type == NodeType::ConstantInput) {
                uint64_t offset =
                    RoundUpToMultiple(mConstantInputsResourceSize,
                                      static_cast<uint64_t>(DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT));
                mConstantInputsResourceSize = offset + inputNode->byteLength;
            } else {
                uint64_t offset =
                    RoundUpToMultiple(mNonConstantInputsResourceSize,
                                      static_cast<uint64_t>(DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT));
                mNonConstantInputsResourceSize = offset + inputNode->byteLength;
            }
        }
        // Set the descriptor heap(s).
        ID3D12DescriptorHeap* descriptorHeaps[] = {mDescriptorHeap.Get()};
        mCommandList->SetDescriptorHeaps(ARRAYSIZE(descriptorHeaps), descriptorHeaps);

        RETURN_IF_FAILED(CreateResourcesForCompiledOperatorInitializer());
        BindTemporaryResource();
        BindPersistentResource();

        if (mConstantInputsResourceSize) {
            std::vector<DML_BUFFER_BINDING> inputBufferBinding(inputNodes.size());
            RETURN_IF_FAILED(FillUploadResourceAndInputBindings(inputBufferBinding, inputNodes));
            // Copy buffer from uploadResource to inputResource.
            CopyBufferRegion(mUploadResource, mInputResource, mConstantInputsResourceSize,
                             D3D12_RESOURCE_STATE_COPY_DEST);

            DML_BUFFER_ARRAY_BINDING inputBufferArrayBinding = {};
            inputBufferArrayBinding.BindingCount = inputBufferBinding.size();
            inputBufferArrayBinding.Bindings = inputBufferBinding.data();
            DML_BINDING_DESC inputBindingDesc{DML_BINDING_TYPE_BUFFER_ARRAY,
                                              &inputBufferArrayBinding};
            mBindingTable->BindInputs(1, &inputBindingDesc);
        }

        // Record execution of the operator initializer.
        // The command recorder is a stateless object that records Dispatches into an existing
        // Direct3D 12 command list.
        RETURN_IF_FAILED(mDevice->CreateCommandRecorder(IID_PPV_ARGS(&mCommandRecorder)));
        mCommandRecorder->RecordDispatch(mCommandList.Get(), compiledOperatorInitializer.Get(),
                                         mBindingTable.Get());
        RETURN_IF_FAILED(CloseExecuteResetWait());

        // Bind and execute the operator on the GPU.
        // Reset the binding table to bind for the operator we want to execute (it was
        // previously used to bind for the initializer).
        for (size_t i = 0; i < outputNodes.size(); ++i) {
            uint64_t byteLength = reinterpret_cast<const DML_BUFFER_TENSOR_DESC*>(
                                      outputNodes[i].outputTensorDesc.Desc)
                                      ->TotalTensorSizeInBytes;
            uint64_t offset = RoundUpToMultiple(
                mOutputResourceSize, static_cast<uint64_t>(DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT));
            mOutputResourceSize = offset + byteLength;
        }

        mBindingTableDesc.Dispatchable = compiledOperator.Get();
        mBindingTable->Reset(&mBindingTableDesc);

        RETURN_IF_FAILED(CreateResourcesForCompiledOperator());
        BindTemporaryResource(false);
        BindPersistentResource(false);
        return S_OK;
    };

    HRESULT Device::ExecuteGraph(ComPtr<IDMLCompiledOperator>& compiledOperator,
                                 const std::vector<std::shared_ptr<InputNode>>& inputNodes,
                                 const std::vector<Node>& outputNodes,
                                 std::unordered_map<std::string, Input> namedInputs,
                                 std::unordered_map<std::string, Resource> namedOutputs) {
        DAWN_ASSERT(compiledOperator != nullptr);
        // Initialize non-constant inputs.
        if (mNonConstantInputsResourceSize) {
            std::vector<DML_BUFFER_BINDING> inputBufferBinding(inputNodes.size());
            RETURN_IF_FAILED(
                FillUploadResourceAndInputBindings(inputBufferBinding, inputNodes, namedInputs));
            // Copy buffer from uploadResource to inputResource.
            CopyBufferRegion(mUploadResource, mInputResource, mNonConstantInputsResourceSize,
                             D3D12_RESOURCE_STATE_COPY_DEST);

            std::vector<DML_BINDING_DESC> inputBindingDesc(inputNodes.size());
            for (size_t i = 0; i < inputBufferBinding.size(); ++i) {
                if (inputBufferBinding[i].Buffer != nullptr) {
                    inputBindingDesc[i] = {DML_BINDING_TYPE_BUFFER, &inputBufferBinding[i]};
                }
            }
            mBindingTable->BindInputs(inputBindingDesc.size(), inputBindingDesc.data());
        }

        // Prepare for outputs and read back buffer from Gpu.
        std::vector<ArrayBufferView> outputArrayBufferViews;
        ArrayBufferView outputArrayBufferView;
        for (size_t i = 0; i < outputNodes.size(); ++i) {
            std::string name = outputNodes[i].name;
            if (namedOutputs.find(name) != namedOutputs.end()) {
                outputArrayBufferView = namedOutputs[name].arrayBufferView;
                outputArrayBufferViews.push_back(outputArrayBufferView);
                DAWN_ASSERT(outputArrayBufferView.buffer != nullptr &&
                            outputArrayBufferView.byteLength != 0);
            } else {
                size_t byteLength = reinterpret_cast<const DML_BUFFER_TENSOR_DESC*>(
                                        outputNodes[i].outputTensorDesc.Desc)
                                        ->TotalTensorSizeInBytes;
                // It is an unuseful output of dml graph. We need not read back and copy buffer
                // to it, just reserve it as a placeholder.
                outputArrayBufferView = {nullptr, byteLength, 0};
                outputArrayBufferViews.push_back(outputArrayBufferView);
            }
        }

        std::vector<DML_BINDING_DESC> outputBindingDesc(outputNodes.size());
        std::vector<DML_BUFFER_BINDING> outputBufferBinding(outputNodes.size());

        uint64_t outputOffset = 0;
        for (size_t i = 0; i < outputNodes.size(); ++i) {
            auto output = outputArrayBufferViews[i];
            outputOffset = RoundUpToMultiple(
                outputOffset, static_cast<uint64_t>(DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT));
            outputBufferBinding[i].Buffer = mOutputResource.Get();
            outputBufferBinding[i].Offset = outputOffset;
            outputBufferBinding[i].SizeInBytes = output.byteLength;
            outputBindingDesc[i] = {DML_BINDING_TYPE_BUFFER, &outputBufferBinding[i]};
            outputOffset = outputOffset + output.byteLength;
        }
        mBindingTable->BindOutputs(outputBindingDesc.size(), outputBindingDesc.data());

        // Record execution of the compiled operator.
        ID3D12DescriptorHeap* descriptorHeaps[] = {mDescriptorHeap.Get()};
        mCommandList->SetDescriptorHeaps(ARRAYSIZE(descriptorHeaps), descriptorHeaps);
        mCommandRecorder->RecordDispatch(mCommandList.Get(), compiledOperator.Get(),
                                         mBindingTable.Get());

        // Copy buffer from outputResource to readBackResource.
        CopyBufferRegion(mOutputResource, mReadBackResource, mOutputResourceSize,
                         D3D12_RESOURCE_STATE_COPY_SOURCE, false);
        RETURN_IF_FAILED(CloseExecuteResetWait());

        int8_t* readBackBuffer;
        RETURN_IF_FAILED(
            mReadBackResource->Map(0, nullptr, reinterpret_cast<void**>(&readBackBuffer)));

        uint64_t offset = 0;
        for (size_t i = 0; i < outputNodes.size(); ++i) {
            offset = RoundUpToMultiple(offset,
                                       static_cast<uint64_t>(DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT));
            ArrayBufferView output = outputArrayBufferViews[i];
            if (output.buffer) {
                memcpy(static_cast<int8_t*>(output.buffer) + output.byteOffset,
                       readBackBuffer + offset, output.byteLength);
            }
            offset += output.byteLength;
        }
        mReadBackResource->Unmap(0, nullptr);
        return S_OK;
    }

}  // namespace webnn::native::dml
