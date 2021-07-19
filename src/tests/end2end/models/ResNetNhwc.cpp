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

#include "examples/ResNet/ResNet.h"
#include "src/tests/WebnnTest.h"

class ResNetNhwcTests : public WebnnTest {
  public:
    void TestResNetNhwc(const std::string& inputFile, const std::string& expectedFile) {
        const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
        ResNet resnet;
        const std::string nhwcPath =
            "node/third_party/webnn-polyfill/test-data/models/resnet101v2_nhwc/";
        ml::Graph graph = resnet.LoadNHWC(nhwcPath + "weights/", false);
        const cnpy::NpyArray inputNpy = cnpy::npy_load(nhwcPath + "test_data_set/" + inputFile);
        const std::vector<float> inputData = inputNpy.as_vec<float>();
        const std::vector<float> result(utils::SizeOfShape({1, 1001}));
        utils::Compute(graph, {{"input", inputData}}, {{"output", result}});
        const cnpy::NpyArray outputNpy = cnpy::npy_load(nhwcPath + "test_data_set/" + expectedFile);
        EXPECT_TRUE(utils::CheckValue(result, outputNpy.as_vec<float>()));
    }
};

TEST_F(ResNetNhwcTests, NhwcTest0) {
    TestResNetNhwc("0/input_0.npy", "0/output_0.npy");
}

TEST_F(ResNetNhwcTests, NhwcTest1) {
    TestResNetNhwc("1/input_0.npy", "1/output_0.npy");
}

TEST_F(ResNetNhwcTests, NhwcTest2) {
    TestResNetNhwc("2/input_0.npy", "2/output_0.npy");
}
