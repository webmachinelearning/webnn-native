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

#ifndef WEBNN_NATIVE_DML_CONTEXT_DML_H_
#define WEBNN_NATIVE_DML_CONTEXT_DML_H_

#include "webnn/native/Context.h"

#include "common/Log.h"
#include "CommandRecorderDML.h"
#include "dml_platform.h"
#include "webnn/native/Graph.h"

namespace webnn::native::dml {

    class Context : public ContextBase {
      public:
        static ContextBase* Create(ComPtr<IDMLDevice> DMLDevice,
                                   ComPtr<ID3D12Device> D3D12Device,
                                   ComPtr<ID3D12CommandQueue> commandQueue);
        ~Context() override = default;

      private:
        Context(ComPtr<IDMLDevice> DMLDevice,
                ComPtr<ID3D12Device> D3D12Device,
                ComPtr<ID3D12CommandQueue> commandQueue);
        HRESULT Initialize();

        GraphBase* CreateGraphImpl() override;

        CommandRecorder mCommandRecorder;
    };

}  // namespace webnn::native::dml

#endif  // WEBNN_NATIVE_DML_CONTEXT_DML_H_
