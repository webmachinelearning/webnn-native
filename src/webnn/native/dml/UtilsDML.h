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

#ifndef WEBNN_NATIVE_UTILSDML_H_
#define WEBNN_NATIVE_UTILSDML_H_

#include <deque>
#include <map>
#include <vector>

#include "dml_platform.h"
#include "common/Assert.h"
#include "common/Log.h"

namespace webnn::native::dml {

#define RETURN_IF_FAILED(EXPR)                                                                    \
    do {                                                                                          \
        auto HR = EXPR;                                                                           \
        if (FAILED(HR)) {                                                                         \
            dawn::ErrorLog() << "Failed to do " << #EXPR << " Return HRESULT " << std::hex << HR; \
            return HR;                                                                            \
        }                                                                                         \
    } while (0)

    // Represent the DirectML tensor description.
    struct TensorDesc {
        std::vector<UINT> dimensions = {};
        std::vector<UINT> strides = {};
        DML_BUFFER_TENSOR_DESC bufferDesc = {};
    };

    enum NodeType { NonConstantInput, ConstantInput, Intermediate };

    // Represent the information of the graph's nodes.
    struct NodeBase {
        virtual ~NodeBase() = default;
        DML_TENSOR_DESC outputTensorDesc = {};
        std::string name = "";
        NodeType type = NodeType::Intermediate;
    };

    // Only represent the information of the input nodes.
    struct InputNode final : public NodeBase {
        ~InputNode() override = default;
        // Indicate the index of the graph's input.
        size_t inputIndex = 0;
        void const* buffer = nullptr;
        size_t byteLength = 0;
    };

    // Represent the information of the intermediate and output nodes.
    struct Node final : public NodeBase {
        ~Node() override = default;
        uint32_t nodeIndex = 0;
        uint32_t outputNodeIndex = 0;
    };

    // Describe a graph of DirectML operators used to compile a combined, optimized operator.
    class GraphBuilder {
      public:
        GraphBuilder(IDMLDevice* device) : mDevice(device) {
        }

        // Create and insert an IDMLOperator into the graph. Notice that this method will update the
        // graph's node count.
        HRESULT CreateOperator(DML_OPERATOR_TYPE type, const void* desc);
        std::shared_ptr<NodeBase> CreateNode(const DML_TENSOR_DESC& outputTensorDesc,
                                             uint32_t outputNodeIndex = 0);
        // Convert nodes to input edges or intermediate edges and insert them into the graph.
        void AddNodes(std::vector<std::shared_ptr<NodeBase>> nodes);

        // Set the node as the graph's output and convert it as output edge to insert into graph.
        void SetGraphOutput(std::shared_ptr<NodeBase> node, UINT graphOutputIndex = 0);
        DML_GRAPH_DESC GetGraphDesc(size_t inputCount, size_t outputCount);

      private:
        void AddInputEdge(const DML_INPUT_GRAPH_EDGE_DESC& inputEdgeDesc) {
            mInputEdgesDesc.push_back(std::move(inputEdgeDesc));
            mInputEdges.push_back({DML_GRAPH_EDGE_TYPE_INPUT, &mInputEdgesDesc.back()});
        };

        void AddIntermediateEdge(const DML_INTERMEDIATE_GRAPH_EDGE_DESC& intermediateEdgeDesc) {
            mIntermediateEdgesDesc.push_back(std::move(intermediateEdgeDesc));
            mIntermediateEdges.push_back(
                {DML_GRAPH_EDGE_TYPE_INTERMEDIATE, &mIntermediateEdgesDesc.back()});
        };

        void AddOutputEdge(const DML_OUTPUT_GRAPH_EDGE_DESC& outputEdgeDesc) {
            mOutputEdgesDesc.push_back(std::move(outputEdgeDesc));
            mOutputEdges.push_back({DML_GRAPH_EDGE_TYPE_OUTPUT, &mOutputEdgesDesc.back()});
        };

        size_t GetNodeCount() {
            return mIntermediateNodes.size();
        };

        IDMLDevice* mDevice;
        std::vector<DML_GRAPH_NODE_DESC> mIntermediateNodes;
        std::vector<DML_GRAPH_EDGE_DESC> mInputEdges;
        std::vector<DML_GRAPH_EDGE_DESC> mOutputEdges;
        std::vector<DML_GRAPH_EDGE_DESC> mIntermediateEdges;

        // Keep intermediate nodes here to avoid releasing too early.
        std::map<uint32_t, ComPtr<IDMLOperator>> mIntermediateNodesMap;
        // Keep the descriptions of nodes and edges here to avoid releasing too early.
        std::deque<DML_OPERATOR_GRAPH_NODE_DESC> mIntermediateNodesDesc;
        std::deque<DML_INPUT_GRAPH_EDGE_DESC> mInputEdgesDesc;
        std::deque<DML_OUTPUT_GRAPH_EDGE_DESC> mOutputEdgesDesc;
        std::deque<DML_INTERMEDIATE_GRAPH_EDGE_DESC> mIntermediateEdgesDesc;
    };

}  // namespace webnn::native::dml

#endif  // WEBNN_NATIVE_UTILS_DML_H_
