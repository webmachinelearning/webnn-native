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

#ifndef WEBNN_NATIVE_DML_MODEL_DML_H_
#define WEBNN_NATIVE_DML_MODEL_DML_H_

#define DML_TARGET_VERSION_USE_LATEST 1

#include <unordered_map>
#include <unordered_set>

#include "DMLUtils.h"
#include "DirectML.h"
#include "webnn/native/Graph.h"
#include "webnn/native/Operand.h"
#include "webnn/native/Operator.h"
#include "webnn/native/dml/ContextDML.h"
#include "webnn/native/ops/BatchNorm.h"
#include "webnn/native/ops/Binary.h"
#include "webnn/native/ops/Clamp.h"
#include "webnn/native/ops/Concat.h"
#include "webnn/native/ops/Constant.h"
#include "webnn/native/ops/Conv2d.h"
#include "webnn/native/ops/Gemm.h"
#include "webnn/native/ops/Gru.h"
#include "webnn/native/ops/Input.h"
#include "webnn/native/ops/InstanceNorm.h"
#include "webnn/native/ops/LeakyRelu.h"
#include "webnn/native/ops/Pad.h"
#include "webnn/native/ops/Pool2d.h"
#include "webnn/native/ops/Reduce.h"
#include "webnn/native/ops/Resample2d.h"
#include "webnn/native/ops/Reshape.h"
#include "webnn/native/ops/Slice.h"
#include "webnn/native/ops/Split.h"
#include "webnn/native/ops/Squeeze.h"
#include "webnn/native/ops/Transpose.h"
#include "webnn/native/ops/Unary.h"
namespace webnn::native::dml {

    using namespace Microsoft::WRL;

    struct CompiledGraph {
        CompiledGraph(ComPtr<ID3D12Device> d3d12Device,
                      ComPtr<IDMLDevice> device,
                      ComPtr<IDMLDevice1> device1,
                      const DML_GRAPH_DESC& graphDesc,
                      DML_EXECUTION_FLAGS flag = DML_EXECUTION_FLAG_NONE)
            : D3D12Device(d3d12Device) {
            WEBNN_CHECK(device.Get()->QueryInterface(IID_PPV_ARGS(&device1)));
            WEBNN_CHECK(device1->CompileGraph(&graphDesc, flag, IID_PPV_ARGS(&compiledOperator)));
            IDMLCompiledOperator* compiledOperators[] = {compiledOperator.Get()};
            WEBNN_CHECK(
                device->CreateOperatorInitializer(ARRAYSIZE(compiledOperators), compiledOperators,
                                                  IID_PPV_ARGS(&compiledOperatorInitializer)));
            DML_BINDING_PROPERTIES initializeBindingProperties =
                compiledOperatorInitializer->GetBindingProperties();
            DML_BINDING_PROPERTIES executeBindingProperties =
                compiledOperator->GetBindingProperties();
            UINT descriptorCount = std::max(initializeBindingProperties.RequiredDescriptorCount,
                                            executeBindingProperties.RequiredDescriptorCount);
            initializedTemporaryResourceSize = initializeBindingProperties.TemporaryResourceSize;
            temporaryResourceSize = std::max(initializedTemporaryResourceSize,
                                             executeBindingProperties.TemporaryResourceSize);
            persistentResourceSize = executeBindingProperties.PersistentResourceSize;

            // Describe and create a constant buffer view (CBV), Shader resource view (SRV), and
            // unordered access view (UAV) descriptor heap.
            D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc{};
            descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            descriptorHeapDesc.NumDescriptors = descriptorCount;
            descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            WEBNN_CHECK(D3D12Device->CreateDescriptorHeap(&descriptorHeapDesc,
                                                          IID_PPV_ARGS(&descriptorHeap)));

            // Create a binding table over the descriptor heap we just created.
            bindingTableDesc.Dispatchable = compiledOperatorInitializer.Get();
            bindingTableDesc.CPUDescriptorHandle =
                descriptorHeap->GetCPUDescriptorHandleForHeapStart();
            bindingTableDesc.GPUDescriptorHandle =
                descriptorHeap->GetGPUDescriptorHandleForHeapStart();
            // The size of the binding table, in descriptors. This is the maximum number of
            // descriptors that DirectML is permitted to write, from the start of both the supplied
            // CPU and GPU descriptor handles.
            bindingTableDesc.SizeInDescriptors = descriptorCount;
            WEBNN_CHECK(device->CreateBindingTable(&bindingTableDesc, IID_PPV_ARGS(&bindingTable)));
        };

        void BindTemporaryResource(bool bindForInitialization = true) {
            if (temporaryResourceSize != 0) {
                if (temporaryResource == nullptr) {
                    D3D12Device->CreateCommittedResource(
                        &CreateHeapProperties(), D3D12_HEAP_FLAG_NONE,
                        &CreateResourceDesc(temporaryResourceSize,
                                            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                        IID_PPV_ARGS(&temporaryResource));
                }

                if ((bindForInitialization && initializedTemporaryResourceSize != 0) ||
                    (!bindForInitialization && temporaryResourceSize != 0)) {
                    DML_BUFFER_BINDING bufferBinding{temporaryResource.Get(), 0,
                                                     temporaryResourceSize};
                    DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
                    bindingTable->BindTemporaryResource(&bindingDesc);
                }
            }
        };

        void BindPersistentResource(bool bindForInitialization = true) {
            if (persistentResourceSize != 0) {
                if (persistentResource == nullptr) {
                    D3D12Device->CreateCommittedResource(
                        &CreateHeapProperties(), D3D12_HEAP_FLAG_NONE,
                        &CreateResourceDesc(persistentResourceSize,
                                            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                        IID_PPV_ARGS(&persistentResource));
                }

                DML_BUFFER_BINDING bufferBinding{persistentResource.Get(), 0,
                                                 persistentResourceSize};
                DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
                if (bindForInitialization) {
                    bindingTable->BindOutputs(1, &bindingDesc);
                } else {
                    bindingTable->BindPersistentResource(&bindingDesc);
                }
            }
        };

        ComPtr<ID3D12Device> D3D12Device;
        // IDMLCompiledOperator represents the DirectML graph's output which need to be initialized
        // by IDMLOperatorInitializer.
        ComPtr<IDMLCompiledOperator> compiledOperator;
        ComPtr<IDMLOperatorInitializer> compiledOperatorInitializer;

        ComPtr<ID3D12DescriptorHeap> descriptorHeap;
        ComPtr<IDMLBindingTable> bindingTable;
        DML_BINDING_TABLE_DESC bindingTableDesc;

        ComPtr<ID3D12Resource> uploadResource;
        ComPtr<ID3D12Resource> inputResource;
        ComPtr<ID3D12Resource> outputResource;
        ComPtr<ID3D12Resource> readBackResource;
        ComPtr<ID3D12Resource> temporaryResource;
        ComPtr<ID3D12Resource> persistentResource;
        uint64_t commonInputsResourceSize = 0;
        uint64_t outputResourceSize = 0;
        UINT64 temporaryResourceSize = 0;
        UINT64 initializedTemporaryResourceSize = 0;
        UINT64 persistentResourceSize = 0;
    };

    class Graph : public GraphBase {
      public:
        explicit Graph(Context* context);
        ~Graph() override = default;

        virtual MaybeError AddConstant(const op::Constant* constant) override;
        virtual MaybeError AddInput(const op::Input* input) override;
        virtual MaybeError AddOutput(std::string_view name, const OperandBase* output) override;
        virtual MaybeError AddBatchNorm(const op::BatchNorm* batchNorm) override;
        virtual MaybeError AddBinary(const op::Binary* binary) override;
        virtual MaybeError AddConv2d(const op::Conv2d* conv2d) override;
        virtual MaybeError AddConvTranspose2d(const op::ConvTranspose2d* convTranspose2d) override;
        virtual MaybeError AddPad(const op::Pad* pad) override;
        virtual MaybeError AddPool2d(const op::Pool2d* pool2d) override;
        virtual MaybeError AddReduce(const op::Reduce* reduce) override;
        virtual MaybeError AddResample2d(const op::Resample2d* resample2d) override;
        virtual MaybeError AddReshape(const op::Reshape* reshape) override;
        virtual MaybeError AddSlice(const op::Slice* slice) override;
        virtual MaybeError AddSplit(const op::Split* split) override;
        virtual MaybeError AddSqueeze(const op::Squeeze* squeeze) override;
        virtual MaybeError AddTranspose(const op::Transpose* transpose) override;
        virtual MaybeError AddUnary(const op::Unary* unary) override;
        virtual MaybeError AddGemm(const op::Gemm* Gemm) override;
        virtual MaybeError AddGru(const op::Gru* Gru) override;
        virtual MaybeError AddConcat(const op::Concat* concat) override;
        virtual MaybeError AddClamp(const op::Clamp* clamp) override;
        virtual MaybeError AddInstanceNorm(const op::InstanceNorm* instanceNorm) override;
        virtual MaybeError Finish() override;

        void FillUploadResourceAndInputBindings(
            uint64_t uploadResourceSize,
            std::vector<DML_BUFFER_BINDING>& inputBufferBinding,
            std::unordered_map<std::string, Input> namedInputs = {});
        MaybeError CreateConstantInput(DML_TENSOR_DESC& inputTensorDESC,
                                       void const* value,
                                       size_t size,
                                       const std::vector<UINT>& dmlTensorDims,
                                       const std::vector<UINT>& strides = {},
                                       DML_TENSOR_DATA_TYPE dataType = DML_TENSOR_DATA_TYPE_FLOAT32,
                                       DML_TENSOR_FLAGS tensorFlag = DML_TENSOR_FLAG_OWNED_BY_DML);
        std::shared_ptr<EdgeInfoBase> Clamp(const op::ClampBase* clamp,
                                            std::shared_ptr<EdgeInfoBase> inputEdge);
        void AppendIdentity(const DML_TENSOR_DESC& inputTensorDesc,
                            DML_TENSOR_DESC& outputTensorDesc,
                            ComPtr<IDMLOperator>& dmlOperator);
        MaybeError HardSwish(std::shared_ptr<EdgeInfoBase>& inputEdge,
                             const std::vector<UINT>& inputDims);
        MaybeError EmulateFusedOperator(FusionOperatorBase* activation,
                                        std::shared_ptr<EdgeInfoBase>& inputEdge,
                                        const std::vector<UINT>& inputDims);
        MaybeError TransposeOutputToNhwc(std::shared_ptr<EdgeInfoBase>& inputEdge,
                                         const std::vector<UINT>& nchwOutputDims);

      private:
        MaybeError CompileImpl() override;
        MaybeError ComputeImpl(NamedInputsBase* inputs, NamedOutputsBase* outputs) override;

        ComPtr<IDMLDevice> mDevice;
        ComPtr<IDMLDevice1> mDevice1;
        ComPtr<ID3D12Device> mD3D12Device;
        ComPtr<IDMLCommandRecorder> mCommandRecorder;
        ComPtr<ID3D12CommandQueue> mCommandQueue;
        ComPtr<ID3D12CommandAllocator> mCommandAllocator;
        ComPtr<ID3D12GraphicsCommandList> mCommandList;

        // Describe a graph of DirectML operators used to compile a combined, optimized operator.
        std::vector<std::shared_ptr<InputEdgeInfo>> mInputs;
        std::vector<EdgeInfo> mOutputs;
        DmlGraphDesc mGraphDesc;
        std::unique_ptr<CompiledGraph> mCompiledGraph;

        std::map<const OperandBase*, std::shared_ptr<EdgeInfoBase>> mGraphEdgesMap;
        // Keep the input tensors description here to avoid releasing too early.
        std::vector<std::shared_ptr<DmlTensorDesc>> mDmlTensorsDesc;
        std::unordered_set<const OperandBase*> mConstantSet;
        std::vector<std::unique_ptr<char>> mConstantsBuffer;
    };

}  // namespace webnn::native::dml

#endif  // WEBNN_NATIVE_DML_MODEL_DML_H_