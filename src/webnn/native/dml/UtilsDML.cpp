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

#include "UtilsDML.h"

namespace webnn::native::dml {

    bool IsWarpAdapter(IDXGIAdapter1* pAdapter) {
        DXGI_ADAPTER_DESC1 pDesc;
        if (FAILED(pAdapter->GetDesc1(&pDesc))) {
            dawn::ErrorLog() << "Failed to get DXGI_ADAPTER_DESC1.";
            DAWN_ASSERT(0);
        };
        // See here for documentation on filtering WARP adapter:
        // https://docs.microsoft.com/en-us/windows/desktop/direct3ddxgi/d3d10-graphics-programming-guide-dxgi#new-info-about-enumerating-adapters-for-windows-8
        bool isBasicRenderDriverVendorId = pDesc.VendorId == 0x1414;
        bool isBasicRenderDriverDeviceId = pDesc.DeviceId == 0x8c;
        bool isSoftwareAdapter = pDesc.Flags == DXGI_ADAPTER_FLAG_SOFTWARE;
        return isSoftwareAdapter || (isBasicRenderDriverVendorId && isBasicRenderDriverDeviceId);
    }

    uint64_t RoundUpToMultiple(uint64_t value, uint64_t multiple) {
        uint64_t remainder = value % multiple;
        if (remainder != 0) {
            DAWN_ASSERT(multiple <= std::numeric_limits<uint64_t>::max() - value);
            value += multiple - remainder;
        }
        return value;
    }

    HRESULT GraphBuilder::CreateOperator(DML_OPERATOR_TYPE type, const void* desc) {
        ComPtr<IDMLOperator> dmlOperator;
        DML_OPERATOR_DESC dmlOperatorDesc = {};
        dmlOperatorDesc.Type = type;
        dmlOperatorDesc.Desc = desc;
        RETURN_IF_FAILED(mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));
        mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;
        DML_OPERATOR_GRAPH_NODE_DESC nodeDesc;
        nodeDesc.Operator = mIntermediateNodesMap[mIntermediateNodes.size()].Get();
        mIntermediateNodesDesc.push_back(std::move(nodeDesc));
        mIntermediateNodes.push_back(
            {DML_GRAPH_NODE_TYPE_OPERATOR, &mIntermediateNodesDesc.back()});
        return S_OK;
    }

    std::shared_ptr<NodeBase> GraphBuilder::CreateNode(const DML_TENSOR_DESC& outputTensorDesc,
                                                       uint32_t outputNodeIndex) {
        std::shared_ptr<Node> node(new Node());
        node->outputTensorDesc = outputTensorDesc;
        node->nodeIndex = this->GetNodeCount() - 1;
        node->outputNodeIndex = outputNodeIndex;
        std::shared_ptr<NodeBase> nodeBase(node);
        return nodeBase;
    }

    void GraphBuilder::AddNodes(std::vector<std::shared_ptr<NodeBase>> nodes) {
        for (size_t i = 0; i < nodes.size(); ++i) {
            switch (nodes[i]->type) {
                case NodeType::ConstantInput:
                case NodeType::NonConstantInput: {
                    auto node = reinterpret_cast<InputNode*>(nodes[i].get());
                    DML_INPUT_GRAPH_EDGE_DESC inputEdgeDesc;
                    inputEdgeDesc.GraphInputIndex = node->inputIndex;
                    inputEdgeDesc.ToNodeIndex = this->GetNodeCount() - 1;
                    inputEdgeDesc.ToNodeInputIndex = i;
                    this->AddInputEdge(inputEdgeDesc);
                    break;
                }
                case NodeType::Intermediate: {
                    auto node = reinterpret_cast<Node*>(nodes[i].get());
                    DML_INTERMEDIATE_GRAPH_EDGE_DESC intermediateEdgeDesc;
                    intermediateEdgeDesc.FromNodeIndex = node->nodeIndex;
                    intermediateEdgeDesc.FromNodeOutputIndex = node->outputNodeIndex;
                    intermediateEdgeDesc.ToNodeIndex = this->GetNodeCount() - 1;
                    intermediateEdgeDesc.ToNodeInputIndex = i;
                    this->AddIntermediateEdge(intermediateEdgeDesc);
                    break;
                }
                default:
                    dawn::ErrorLog() << "Invalid node type";
                    DAWN_ASSERT(0);
            }
        }
    }

    void GraphBuilder::SetGraphOutput(std::shared_ptr<NodeBase> node, UINT graphOutputIndex) {
        DAWN_ASSERT(node->type == NodeType::Intermediate);
        auto outputNode = reinterpret_cast<Node*>(node.get());
        DML_OUTPUT_GRAPH_EDGE_DESC outputEdgeDesc;
        outputEdgeDesc.FromNodeIndex = outputNode->nodeIndex;
        outputEdgeDesc.FromNodeOutputIndex = outputNode->outputNodeIndex;
        outputEdgeDesc.GraphOutputIndex = graphOutputIndex;
        this->AddOutputEdge(outputEdgeDesc);
    }

    DML_GRAPH_DESC GraphBuilder::GetGraphDesc(size_t inputCount, size_t outputCount) {
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

}  // namespace webnn::native::dml
