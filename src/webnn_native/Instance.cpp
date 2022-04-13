// Copyright 2018 The Dawn Authors
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

#include "webnn_native/Instance.h"

#include "common/Assert.h"
#include "common/Log.h"
#include "webnn_native/ErrorData.h"
#include "webnn_native/GraphBuilder.h"
#include "webnn_native/NamedInputs.h"
#include "webnn_native/NamedOperands.h"
#include "webnn_native/NamedOutputs.h"
#include "webnn_native/ValidationUtils_autogen.h"

namespace webnn_native {

    // Forward definitions of each backend's "Connect" function that creates new BackendConnection.
    // Conditionally compiled declarations are used to avoid using static constructors instead.
#if defined(WEBNN_ENABLE_BACKEND_DML)
    namespace dml {
        BackendConnection* Connect(InstanceBase* instance);
    }
#endif  // defined(WEBNN_ENABLE_BACKEND_DML)
#if defined(WEBNN_ENABLE_BACKEND_OPENVINO)
    namespace ie {
        BackendConnection* Connect(InstanceBase* instance);
    }
#endif  // defined(WEBNN_ENABLE_BACKEND_OPENVINO)
#if defined(WEBNN_ENABLE_BACKEND_NULL)
    namespace null {
        BackendConnection* Connect(InstanceBase* instance);
    }
#endif  // defined(WEBNN_ENABLE_BACKEND_NULL)
#if defined(WEBNN_ENABLE_BACKEND_ONEDNN)
    namespace onednn {
        BackendConnection* Connect(InstanceBase* instance);
    }
#endif  // defined(WEBNN_ENABLE_BACKEND_ONEDNN)
#if defined(WEBNN_ENABLE_BACKEND_MLAS)
    namespace mlas {
        BackendConnection* Connect(InstanceBase* instance);
    }
#endif  // defined(WEBNN_ENABLE_BACKEND_MLAS)

    namespace {

        BackendsBitset GetEnabledBackends() {
            BackendsBitset enabledBackends;
#if defined(WEBNN_ENABLE_BACKEND_NULL)
            enabledBackends.set(wnn::BackendType::Null);
#endif  // defined(WEBNN_ENABLE_BACKEND_NULL)
#if defined(WEBNN_ENABLE_BACKEND_DML)
            enabledBackends.set(wnn::BackendType::DirectML);
#endif  // defined(WEBNN_ENABLE_BACKEND_DML)
#if defined(WEBNN_ENABLE_BACKEND_OPENVINO)
            enabledBackends.set(wnn::BackendType::OpenVINO);
#endif  // defined(WEBNN_ENABLE_BACKEND_OPENVINO)
#if defined(WEBNN_ENABLE_BACKEND_ONEDNN)
            enabledBackends.set(wnn::BackendType::OneDNN);
#endif  // defined(WEBNN_ENABLE_BACKEND_ONEDNN)
#if defined(WEBNN_ENABLE_BACKEND_MLAS)
            enabledBackends.set(wnn::BackendType::MLAS);
#endif  // defined(WEBNN_ENABLE_BACKEND_MLAS)
            return enabledBackends;
        }

    }  // anonymous namespace

    // InstanceBase

    // static
    InstanceBase* InstanceBase::Create(const InstanceDescriptor* descriptor) {
        Ref<InstanceBase> instance = AcquireRef(new InstanceBase);
        if (!instance->Initialize(descriptor)) {
            return nullptr;
        }
        return instance.Detach();
    }

    bool InstanceBase::Initialize(const InstanceDescriptor*) {
        for (wnn::BackendType b : IterateBitSet(GetEnabledBackends())) {
            ConnectBackend(b);
        }
        return true;
    }

    void InstanceBase::ConnectBackend(wnn::BackendType backendType) {
        auto Register = [this](BackendConnection* connection, wnn::BackendType expectedType) {
            if (connection != nullptr) {
                ASSERT(connection->GetType() == expectedType);
                ASSERT(connection->GetInstance() == this);
                mBackends.insert(
                    std::make_pair(expectedType, std::unique_ptr<BackendConnection>(connection)));
            }
        };

        switch (backendType) {
#if defined(WEBNN_ENABLE_BACKEND_NULL)
            case wnn::BackendType::Null:
                Register(null::Connect(this), wnn::BackendType::Null);
                break;
#endif  // defined(WEBNN_ENABLE_BACKEND_NULL)

#if defined(WEBNN_ENABLE_BACKEND_DML)
            case wnn::BackendType::DirectML:
                Register(dml::Connect(this), wnn::BackendType::DirectML);
                break;
#endif  // defined(WEBNN_ENABLE_BACKEND_DML)

#if defined(WEBNN_ENABLE_BACKEND_OPENVINO)
            case wnn::BackendType::OpenVINO:
                Register(ie::Connect(this), wnn::BackendType::OpenVINO);
                break;
#endif  // defined(WEBNN_ENABLE_BACKEND_OPENVINO)

#if defined(WEBNN_ENABLE_BACKEND_ONEDNN)
            case wnn::BackendType::OneDNN:
                Register(onednn::Connect(this), wnn::BackendType::OneDNN);
                break;
#endif  // defined(WEBNN_ENABLE_BACKEND_ONEDNN)

#if defined(WEBNN_ENABLE_BACKEND_MLAS)
            case wnn::BackendType::MLAS:
                Register(mlas::Connect(this), wnn::BackendType::MLAS);
                break;
#endif  // defined(WEBNN_ENABLE_BACKEND_MLAS)

            default:
                UNREACHABLE();
        }
    }

    ContextBase* InstanceBase::CreateTestContext(const ContextOptions* options) {
        ASSERT(mBackends.find(wnn::BackendType::Null) != mBackends.end());
        return mBackends[wnn::BackendType::Null]->CreateContext(options);
    }

    ContextBase* InstanceBase::CreateContext(const ContextOptions* options) {
        if (mBackends.find(wnn::BackendType::DirectML) != mBackends.end()) {
            return mBackends[wnn::BackendType::DirectML]->CreateContext(options);
        } else if (mBackends.find(wnn::BackendType::OpenVINO) != mBackends.end()) {
            return mBackends[wnn::BackendType::OpenVINO]->CreateContext(options);
        } else if (mBackends.find(wnn::BackendType::OneDNN) != mBackends.end()) {
            return mBackends[wnn::BackendType::OneDNN]->CreateContext(options);
        } else if (mBackends.find(wnn::BackendType::MLAS) != mBackends.end()) {
            return mBackends[wnn::BackendType::MLAS]->CreateContext(options);
        }
        UNREACHABLE();
        return nullptr;
    }

    ContextBase* InstanceBase::CreateContextWithGpuDevice(const GpuDevice* wnn_device) {
#if defined(WEBNN_ENABLE_GPU_BUFFER)
        WGPUDevice device = reinterpret_cast<WGPUDevice>(wnn_device->device);
        if (mBackends.find(wnn::BackendType::DirectML) != mBackends.end()) {
            return mBackends[wnn::BackendType::DirectML]->CreateContextWithGpuDevice(device);
        } else if (mBackends.find(wnn::BackendType::OpenVINO) != mBackends.end()) {
            return mBackends[wnn::BackendType::OpenVINO]->CreateContextWithGpuDevice(device);
        } else if (mBackends.find(wnn::BackendType::OneDNN) != mBackends.end()) {
            return mBackends[wnn::BackendType::OneDNN]->CreateContextWithGpuDevice(device);
        } else if (mBackends.find(wnn::BackendType::MLAS) != mBackends.end()) {
            return mBackends[wnn::BackendType::MLAS]->CreateContextWithGpuDevice(device);
        }
        UNREACHABLE();
#endif
        return nullptr;
    }

    GraphBuilderBase* InstanceBase::CreateGraphBuilder(ContextBase* context) {
        return new GraphBuilderBase(context);
    }

    NamedInputsBase* InstanceBase::CreateNamedInputs() {
        return new NamedInputsBase();
    }

    NamedOperandsBase* InstanceBase::CreateNamedOperands() {
        return new NamedOperandsBase();
    }

    NamedOutputsBase* InstanceBase::CreateNamedOutputs() {
        return new NamedOutputsBase();
    }

    OperatorArrayBase* InstanceBase::CreateOperatorArray() {
        return new OperatorArrayBase();
    }

    bool InstanceBase::ConsumedError(MaybeError maybeError) {
        if (maybeError.IsError()) {
            std::unique_ptr<ErrorData> error = maybeError.AcquireError();

            ASSERT(error != nullptr);
            // TODO: gpgmm config polluted dawn common.
            dawn::ErrorLog() << error->GetMessage();

            return true;
        }
        return false;
    }

}  // namespace webnn_native
