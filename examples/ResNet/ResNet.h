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

#include <webnn/webnn.h>
#include <webnn/webnn_cpp.h>

#include "examples/SampleUtils.h"

class ResNet : public ExampleBase {
  public:
    ResNet();
    ~ResNet() override = default;

    bool ParseAndCheckExampleOptions(int argc, const char* argv[]) override;
    const wnn::Operand LoadNchw(const wnn::GraphBuilder& builder, bool softmax = true);
    const wnn::Operand LoadNhwc(const wnn::GraphBuilder& builder, bool softmax = true);
    const wnn::Operand BuildNchwConv(const wnn::GraphBuilder& builder,
                                     const wnn::Operand& input,
                                     const std::string& name,
                                     const std::string& stageName,
                                     utils::Conv2dOptions* options = nullptr);
    const wnn::Operand BuildNhwcConv(const wnn::GraphBuilder& builder,
                                     const wnn::Operand& input,
                                     const std::vector<std::string> nameIndices,
                                     utils::Conv2dOptions* options = nullptr,
                                     bool relu = true);
    const wnn::Operand BuildBatchNorm(const wnn::GraphBuilder& builder,
                                      const wnn::Operand& input,
                                      const std::string& name,
                                      const std::string& stageName,
                                      bool relu = true);
    const wnn::Operand BuildFusedBatchNorm(const wnn::GraphBuilder& builder,
                                           const wnn::Operand& input,
                                           const std::vector<std::string> nameIndices);
    const wnn::Operand BuildNchwBottlenectV2(const wnn::GraphBuilder& builder,
                                             const wnn::Operand& input,
                                             const std::string& stageName,
                                             const std::vector<std::string> nameIndices,
                                             bool downsample = false,
                                             int32_t stride = 1);
    const wnn::Operand BuildNhwcBottlenectV2(const wnn::GraphBuilder& builder,
                                             const wnn::Operand& input,
                                             std::vector<std::string> nameIndices,
                                             bool downsample = false,
                                             bool shortcut = true);
    const wnn::Operand BuildGemm(const wnn::GraphBuilder& builder,
                                 const wnn::Operand& input,
                                 const std::string& name);
    const wnn::Operand loop(const wnn::GraphBuilder& builder,
                            const wnn::Operand node,
                            uint32_t num);
};
