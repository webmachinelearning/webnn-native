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

class MobileNetV2 {
  public:
    MobileNetV2(bool nchw);
    ~MobileNetV2() = default;

    bool LoadNCHW(const std::string& weightsPath, bool softmax = true);
    bool LoadBatchNormNCHW(const std::string& weightsPath, bool softmax = true);
    bool LoadNHWC(const std::string& weightsPath, bool softmax = true);
    ml::Result Compute(const void* inputData, size_t inputLength);
    const ml::Operand BuildConstantFromNpy(const ml::GraphBuilder& builder,
                                           const std::string& path);
    const ml::Operand BuildConv(const ml::GraphBuilder& builder,
                                const ml::Operand& input,
                                int32_t convIndex,
                                bool fused,
                                utils::Conv2dOptions* options = nullptr,
                                const std::string& biasName = "");
    const ml::Operand BuildConvBatchNorm(const ml::GraphBuilder& builder,
                                         const ml::Operand& input,
                                         int32_t nameIndex,
                                         utils::Conv2dOptions* options = nullptr,
                                         int32_t subNameIndex = -1);
    const ml::Operand BuildFire(const ml::GraphBuilder& builder,
                                const ml::Operand& input,
                                const std::vector<int32_t>& convIndexes,
                                int32_t groups,
                                bool strides = false,
                                bool shouldAdd = true);
    const ml::Operand BuildBatchNormFire(const ml::GraphBuilder& builder,
                                         const ml::Operand& input,
                                         int32_t subNameIndex,
                                         utils::Conv2dOptions* options);
    const ml::Operand BuildLinearBottleneck(const ml::GraphBuilder& builder,
                                            const ml::Operand& input,
                                            const std::vector<int32_t>& convIndexes,
                                            int32_t biasIndex,
                                            utils::Conv2dOptions* dwiseOptions,
                                            bool shouldAdd = true);
    const ml::Operand BuildFireMore(const ml::GraphBuilder& builder,
                                    const ml::Operand& input,
                                    const std::vector<int32_t>& convIndexes,
                                    const std::vector<int32_t> groups,
                                    bool strides = true);
    const ml::Operand BuildGemm(const ml::GraphBuilder& builder,
                                const ml::Operand& input,
                                int32_t gemmIndex);

  private:
    ml::Context mContext;
    ml::Graph mGraph;
    ml::NamedResults mResults;
    bool mNCHW = true;
    std::vector<SHARED_DATA_TYPE> mConstants;
    std::string mDataPath;
};
