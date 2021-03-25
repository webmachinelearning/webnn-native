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

#include "examples/SqueezeNet/SqueezeNet.h"
#include "src/tests/WebnnTest.h"

class SqueezeNetNhwcTests : public WebnnTest {
  public:
    void TestSqueezeNetNhwc(const std::string& inputFile, const std::string& expectedFile) {
        const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
        SqueezeNet squeezenet(false);
        const std::string nhwcPath =
            "node/third_party/webnn-polyfill/test-data/models/squeezenet1.0_nhwc/";
        squeezenet.LoadNHWC(nhwcPath + "weights/");
        const cnpy::NpyArray inputNpy = cnpy::npy_load(nhwcPath + "test_data_set/" + inputFile);
        const std::vector<float> inputData = inputNpy.as_vec<float>();
        const ml::Result result =
            squeezenet.Compute(inputData.data(), inputData.size() * sizeof(float));
        EXPECT_TRUE(utils::CheckShape(result, {1, 1001}));
        const cnpy::NpyArray outputNpy = cnpy::npy_load(nhwcPath + "test_data_set/" + expectedFile);
        EXPECT_TRUE(utils::CheckValue(result, outputNpy.as_vec<float>()));
    }
};

TEST_F(SqueezeNetNhwcTests, NhwcTest0) {
    TestSqueezeNetNhwc("0/input_0.npy", "0/output_0.npy");
}

TEST_F(SqueezeNetNhwcTests, NhwcTest1) {
    TestSqueezeNetNhwc("1/input_0.npy", "1/output_0.npy");
}

TEST_F(SqueezeNetNhwcTests, NhwcTest2) {
    TestSqueezeNetNhwc("2/input_0.npy", "2/output_0.npy");
}
