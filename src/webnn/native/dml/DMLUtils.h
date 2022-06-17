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

#define DML_TARGET_VERSION_USE_LATEST 1

#include <dxgi1_6.h>
#include <webnn/webnn_cpp.h>
#include <wrl\client.h>
#include <map>
#include <vector>

#include "DirectML.h"
#include "common/Assert.h"
#include "common/Log.h"

namespace webnn::native::dml {
#define WEBNN_CHECK(hr)                             \
    if (((HRESULT)(hr)) < 0) {                      \
        dawn::ErrorLog() << "Failed to do " << #hr; \
        DAWN_ASSERT(0);                             \
    }

    using namespace Microsoft::WRL;

    // Represent the DirectML tensor description.
    struct DmlTensorDesc {
        std::vector<UINT> dimensions = {};
        std::vector<UINT> strides = {};
        // Describes a tensor that will be stored in a Direct3D 12 buffer resource.
        DML_BUFFER_TENSOR_DESC bufferDesc = {};
    };

    // Represent the information of the graph's edges.
    struct EdgeInfoBase {
        virtual ~EdgeInfoBase() = default;
        DML_TENSOR_DESC outputTensorDESC = {};
        std::string name = "";
        bool isInputEdge = false;
    };

    // Only represent the information of the input edges.
    struct InputEdgeInfo final : public EdgeInfoBase {
        ~InputEdgeInfo() override = default;
        // Indicate the index of the graph's input.
        size_t inputIndex = 0;
        void const* buffer = nullptr;
        size_t byteLength = 0;
        // Indicate if the input is from constant buffer which need to be
        // uploaded in the stage of initialization.
        bool isConstantInput = false;
    };

    // Represent the information of the intermediate edges and output edges.
    struct EdgeInfo final : public EdgeInfoBase {
        ~EdgeInfo() override = default;
        // Indicate the index of the intermediate node from which this edge was produced.
        uint32_t nodeIndex = 0;
        // Indicate the index of the intermediate node' output from which this edge was produced.
        uint32_t outputNodeIndex = 0;
    };

    // Describe a graph of DirectML operators used to compile a combined, optimized operator.
    class DmlGraphDesc {
      public:
        void AddInputEdge(std::unique_ptr<DML_INPUT_GRAPH_EDGE_DESC>& inputEdgeDesc) {
            mInputEdges.push_back({DML_GRAPH_EDGE_TYPE_INPUT, inputEdgeDesc.get()});
            mInputEdgesDesc.push_back(std::move(inputEdgeDesc));
        };
        void AddIntermediateEdge(
            std::unique_ptr<DML_INTERMEDIATE_GRAPH_EDGE_DESC>& intermediateEdgeDesc) {
            mIntermediateEdges.push_back(
                {DML_GRAPH_EDGE_TYPE_INTERMEDIATE, intermediateEdgeDesc.get()});
            mIntermediateEdgesDesc.push_back(std::move(intermediateEdgeDesc));
        };
        void AddOutputEdge(std::unique_ptr<DML_OUTPUT_GRAPH_EDGE_DESC>& outputEdgeDesc) {
            mOutputEdges.push_back({DML_GRAPH_EDGE_TYPE_OUTPUT, outputEdgeDesc.get()});
            mOutputEdgesDesc.push_back(std::move(outputEdgeDesc));
        };
        void AddIntermediateNode(ComPtr<IDMLOperator> dmlOperator) {
            mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;
            std::unique_ptr<DML_OPERATOR_GRAPH_NODE_DESC> nodeDesc(
                new DML_OPERATOR_GRAPH_NODE_DESC);
            nodeDesc->Operator = mIntermediateNodesMap[mIntermediateNodes.size()].Get();
            mIntermediateNodes.push_back({DML_GRAPH_NODE_TYPE_OPERATOR, nodeDesc.get()});
            mIntermediateNodesDesc.push_back(std::move(nodeDesc));
        }
        size_t NodeCount() {
            return mIntermediateNodes.size();
        };

        DML_GRAPH_DESC ConvertDmlGraphDesc(size_t inputCount, size_t outputCount) {
            DML_GRAPH_DESC graphDesc = {};
            graphDesc.NodeCount = static_cast<UINT>(mIntermediateNodes.size());
            graphDesc.Nodes = mIntermediateNodes.data();
            graphDesc.InputEdgeCount = static_cast<UINT>(mInputEdges.size());
            graphDesc.InputEdges = mInputEdges.data();
            graphDesc.OutputEdgeCount = static_cast<UINT>(mOutputEdges.size());
            graphDesc.OutputEdges = mOutputEdges.data();
            graphDesc.IntermediateEdgeCount = static_cast<UINT>(mIntermediateEdges.size());
            graphDesc.IntermediateEdges = mIntermediateEdges.data();
            graphDesc.InputCount = static_cast<UINT>(inputCount);
            graphDesc.OutputCount = static_cast<UINT>(outputCount);
            return graphDesc;
        };

      private:
        std::vector<DML_GRAPH_NODE_DESC> mIntermediateNodes;
        std::vector<DML_GRAPH_EDGE_DESC> mInputEdges;
        std::vector<DML_GRAPH_EDGE_DESC> mOutputEdges;
        std::vector<DML_GRAPH_EDGE_DESC> mIntermediateEdges;

        // Keep intermediate nodes here to avoid releasing too early.
        std::map<uint32_t, ComPtr<IDMLOperator>> mIntermediateNodesMap;
        // Keep the descriptions of nodes and edges here to avoid releasing too early.
        std::vector<std::unique_ptr<DML_OPERATOR_GRAPH_NODE_DESC>> mIntermediateNodesDesc;
        std::vector<std::unique_ptr<DML_INPUT_GRAPH_EDGE_DESC>> mInputEdgesDesc;
        std::vector<std::unique_ptr<DML_OUTPUT_GRAPH_EDGE_DESC>> mOutputEdgesDesc;
        std::vector<std::unique_ptr<DML_INTERMEDIATE_GRAPH_EDGE_DESC>> mIntermediateEdgesDesc;
    };

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
    HRESULT IsWarpAdapter(IDXGIAdapter1* pAdapter, bool* isWarpAdapter);

    void InitD3D12(ComPtr<ID3D12GraphicsCommandList>& commandList,
                   ComPtr<ID3D12CommandQueue>& commandQueue,
                   ComPtr<ID3D12CommandAllocator>& commandAllocator,
                   ComPtr<ID3D12Device>& D3D12Device,
                   DXGI_GPU_PREFERENCE gpuPreference,
                   bool useGpu);

    void CloseExecuteResetWait(ComPtr<ID3D12GraphicsCommandList> commandList,
                               ComPtr<ID3D12CommandQueue> commandQueue,
                               ComPtr<ID3D12CommandAllocator> commandAllocator,
                               ComPtr<ID3D12Device> D3D12Device);
}  // namespace webnn::native::dml
#endif  // WEBNN_NATIVE_DML_UTILS_H_