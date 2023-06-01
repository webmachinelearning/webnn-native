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

#include "webnn/native/dml/ContextDML.h"

#include "webnn/native/dml/GraphDML.h"

namespace webnn::native::dml {

    HRESULT Context::Initialize() {
        WEBNN_RETURN_IF_FAILED(mCommandRecorder.D3D12Device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&mCommandRecorder.commandAllocator)));
        WEBNN_RETURN_IF_FAILED(mCommandRecorder.D3D12Device->CreateCommandList(
            0, D3D12_COMMAND_LIST_TYPE_DIRECT, mCommandRecorder.commandAllocator.Get(), nullptr,
            IID_PPV_ARGS(&mCommandRecorder.commandList)));
        return S_OK;
    };

    // static
    ContextBase* Context::Create(ComPtr<IDMLDevice> DMLDevice,
                                 ComPtr<ID3D12Device> D3D12Device,
                                 ComPtr<ID3D12CommandQueue> commandQueue) {
        Context* context = new Context(DMLDevice, D3D12Device, commandQueue);
        if (FAILED(context->Initialize())) {
            dawn::ErrorLog() << "Failed to initialize Device.";
            delete context;
            return nullptr;
        }
        return context;
    }

    Context::Context(ComPtr<IDMLDevice> DMLDevice,
                     ComPtr<ID3D12Device> D3D12Device,
                     ComPtr<ID3D12CommandQueue> commandQueue) {
        mCommandRecorder.DMLDevice = DMLDevice;
        mCommandRecorder.D3D12Device = D3D12Device;
        mCommandRecorder.commandQueue = commandQueue;
    }

    GraphBase* Context::CreateGraphImpl() {
        return new Graph(this);
    }

}  // namespace webnn::native::dml
