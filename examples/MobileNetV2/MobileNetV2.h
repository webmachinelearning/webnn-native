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

class MobileNetV2 : public ExampleBase {
  public:
    MobileNetV2();
    ~MobileNetV2() override = default;

    bool ParseAndCheckExampleOptions(int argc, const char* argv[]) override;
    const wnn::Operand LoadNCHW(const wnn::GraphBuilder& builder, bool softmax = true);
    const wnn::Operand LoadNHWC(const wnn::GraphBuilder& builder, bool softmax = true);
    const wnn::Operand LoadBatchNormNCHW(const wnn::GraphBuilder& builder, bool softmax = true);
    const wnn::Operand BuildConstantFromNpy(const wnn::GraphBuilder& builder,
                                            const std::string& path);
    const wnn::Operand BuildConv(const wnn::GraphBuilder& builder,
                                 const wnn::Operand& input,
                                 int32_t convIndex,
                                 bool relu6,
                                 utils::Conv2dOptions* options = nullptr,
                                 const std::string& biasName = "");
    const wnn::Operand BuildConvBatchNorm(const wnn::GraphBuilder& builder,
                                          const wnn::Operand& input,
                                          int32_t nameIndex,
                                          bool relu,
                                          utils::Conv2dOptions* options = nullptr,
                                          int32_t subNameIndex = -1);
    const wnn::Operand BuildFire(const wnn::GraphBuilder& builder,
                                 const wnn::Operand& input,
                                 const std::vector<int32_t>& convIndexes,
                                 int32_t groups,
                                 bool strides = false,
                                 bool shouldAdd = true);
    const wnn::Operand BuildBatchNormFire(const wnn::GraphBuilder& builder,
                                          const wnn::Operand& input,
                                          int32_t subNameIndex,
                                          utils::Conv2dOptions* options);
    const wnn::Operand BuildLinearBottleneck(const wnn::GraphBuilder& builder,
                                             const wnn::Operand& input,
                                             const std::vector<int32_t>& convIndexes,
                                             int32_t biasIndex,
                                             utils::Conv2dOptions* dwiseOptions,
                                             bool shouldAdd = true);
    const wnn::Operand BuildFireMore(const wnn::GraphBuilder& builder,
                                     const wnn::Operand& input,
                                     const std::vector<int32_t>& convIndexes,
                                     const std::vector<int32_t> groups,
                                     bool strides = true);
    const wnn::Operand BuildGemm(const wnn::GraphBuilder& builder,
                                 const wnn::Operand& input,
                                 int32_t gemmIndex);

  private:
    std::vector<SHARED_DATA_TYPE> mConstants;
};
