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

class ResNet {
  public:
    ResNet();
    ~ResNet() = default;

    ml::Graph LoadNCHW(const std::string& weightsPath, bool softmax = true);
    ml::Graph LoadNHWC(const std::string& weightsPath, bool softmax = true);
    const ml::Operand BuildConstantFromNpy(const ml::GraphBuilder& builder,
                                           const std::string& path);
    const ml::Operand BuildNchwConv(const ml::GraphBuilder& builder,
                                    const ml::Operand& input,
                                    const std::string& name,
                                    const std::string& stageName,
                                    utils::Conv2dOptions* options = nullptr);
    const ml::Operand BuildNhwcConv(const ml::GraphBuilder& builder,
                                    const ml::Operand& input,
                                    const std::vector<std::string> nameIndices,
                                    utils::Conv2dOptions* options = nullptr,
                                    bool relu = true);
    const ml::Operand BuildBatchNorm(const ml::GraphBuilder& builder,
                                     const ml::Operand& input,
                                     const std::string& name,
                                     const std::string& stageName,
                                     bool relu = true);
    const ml::Operand BuildFusedBatchNorm(const ml::GraphBuilder& builder,
                                          const ml::Operand& input,
                                          const std::vector<std::string> nameIndices);
    const ml::Operand BuildNchwBottlenectV2(const ml::GraphBuilder& builder,
                                            const ml::Operand& input,
                                            const std::string& stageName,
                                            const std::vector<std::string> nameIndices,
                                            bool downsample = false,
                                            int32_t stride = 1);
    const ml::Operand BuildNhwcBottlenectV2(const ml::GraphBuilder& builder,
                                            const ml::Operand& input,
                                            std::vector<std::string> nameIndices,
                                            bool downsample = false,
                                            bool shortcut = true);
    const ml::Operand BuildGemm(const ml::GraphBuilder& builder,
                                const ml::Operand& input,
                                const std::string& name);
    const ml::Operand loop(const ml::GraphBuilder& builder, const ml::Operand node, uint32_t num);

  private:
    ml::Context mContext;
    std::vector<SHARED_DATA_TYPE> mConstants;
    std::string mDataPath;
};
