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

class SqueezeNet {
  public:
    SqueezeNet(bool nchw);
    ~SqueezeNet() = default;

    ml::Graph LoadNCHW(const std::string& weightsPath, bool softmax = true);
    ml::Graph LoadNHWC(const std::string& weightsPath, bool softmax = true);
    std::vector<float> Compute(const void* inputData, size_t inputLength);
    const ml::Operand BuildConstantFromNpy(const ml::GraphBuilder& builder,
                                           const std::string& path);
    const ml::Operand BuildConv(const ml::GraphBuilder& builder,
                                const ml::Operand& input,
                                const std::string& name,
                                utils::Conv2dOptions* options = nullptr);
    const ml::Operand BuildFire(const ml::GraphBuilder& builder,
                                const ml::Operand& input,
                                const std::string& convName,
                                const std::string& conv1x1Name,
                                const std::string& conv3x3Name);

  private:
    ml::Context mContext;
    bool mNchw = true;
    std::vector<SHARED_DATA_TYPE> mConstants;
    std::string mDataPath;
};
