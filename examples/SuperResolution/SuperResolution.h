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

#include <webnn/webnn.h>
#include <webnn/webnn_cpp.h>

#include "examples/SampleUtils.h"

class SuperResolution : public ExampleBase {
  public:
    SuperResolution();
    ~SuperResolution() override = default;

    const wnn::Operand LoadNchw(const wnn::GraphBuilder& builder, bool softmax);

  private:
    const wnn::Operand BuildConstantFromNpy(const wnn::GraphBuilder& builder,
                                            const std::string& path);

    const wnn::Operand BuildConv(const wnn::GraphBuilder& builder,
                                 const wnn::Operand& input,
                                 int32_t convIndex,
                                 bool relu6,
                                 utils::Conv2dOptions* options,
                                 const std::string& biasName = "");

    std::vector<SHARED_DATA_TYPE> mConstants;
};