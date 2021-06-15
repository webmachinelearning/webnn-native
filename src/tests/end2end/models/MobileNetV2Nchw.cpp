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

#include "examples/MobileNetV2/MobileNetV2.h"
#include "src/tests/WebnnTest.h"

class MobileNetV2NchwTests : public WebnnTest {
  public:
    void TestMobileNetV2Nchw(const std::string& inputFile, const std::string& expectedFile) {
        const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
        MobileNetV2 mobilemetv2(true);
        const std::string nchwPath =
            "node/third_party/webnn-polyfill/test-data/models/mobilenetv2_nchw/";
        mobilemetv2.LoadNCHW(nchwPath + "weights/", false);
        const cnpy::NpyArray inputNpy = cnpy::npy_load(nchwPath + "test_data_set/" + inputFile);
        const std::vector<float> inputData = inputNpy.as_vec<float>();
        const ml::Result result =
            mobilemetv2.Compute(inputData.data(), inputData.size() * sizeof(float));
        EXPECT_TRUE(utils::CheckShape(result, {1, 1000}));
        const cnpy::NpyArray outputNpy = cnpy::npy_load(nchwPath + "test_data_set/" + expectedFile);
        EXPECT_TRUE(utils::CheckValue(result, outputNpy.as_vec<float>()));
    }
};

TEST_F(MobileNetV2NchwTests, NchwTest0) {
    TestMobileNetV2Nchw("0/input_0.npy", "0/output_0.npy");
}

TEST_F(MobileNetV2NchwTests, NchwTest1) {
    TestMobileNetV2Nchw("1/input_0.npy", "1/output_0.npy");
}

TEST_F(MobileNetV2NchwTests, NchwTest2) {
    TestMobileNetV2Nchw("2/input_0.npy", "2/output_0.npy");
}