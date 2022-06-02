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
#include "webnn/tests/WebnnTest.h"

static const std::string kModelPath = WEBNN_END2END_TEST_MODEL_PATH;

class MobileNetV2NhwcTests : public WebnnTest {
  public:
    void TestMobileNetV2Nhwc(const std::string& inputFile,
                             const std::string& expectedFile,
                             bool fused = true) {
        MobileNetV2 mobilenetv2;
        mobilenetv2.mFused = fused;
        const std::string nhwcPath = kModelPath + "/mobilenetv2_nhwc/";
        mobilenetv2.mWeightsPath = nhwcPath + "weights/";
        mobilenetv2.mLayout = "nhwc";
        const wnn::GraphBuilder builder = utils::CreateGraphBuilder(GetContext());
        wnn::Operand output = mobilenetv2.LoadNhwc(builder);
        wnn::Graph graph = utils::Build(builder, {{"output", output}});
        const cnpy::NpyArray inputNpy = cnpy::npy_load(nhwcPath + "test_data_set/" + inputFile);
        const std::vector<float> inputData = inputNpy.as_vec<float>();
        std::vector<float> result(utils::SizeOfShape({1, 1001}));
        utils::Compute(graph, {{"input", inputData}}, {{"output", result}});
        const cnpy::NpyArray outputNpy = cnpy::npy_load(nhwcPath + "test_data_set/" + expectedFile);
        EXPECT_TRUE(utils::CheckValue(result, outputNpy.as_vec<float>()));
    }
};

TEST_F(MobileNetV2NhwcTests, NhwcTest0) {
    TestMobileNetV2Nhwc("0/input_0.npy", "0/output_0.npy", false);
}

TEST_F(MobileNetV2NhwcTests, NhwcTest1) {
    TestMobileNetV2Nhwc("1/input_0.npy", "1/output_0.npy", false);
}

TEST_F(MobileNetV2NhwcTests, NhwcTest2) {
    TestMobileNetV2Nhwc("2/input_0.npy", "2/output_0.npy", false);
}

TEST_F(MobileNetV2NhwcTests, FusedNhwcTest0) {
    TestMobileNetV2Nhwc("0/input_0.npy", "0/output_0.npy");
}

TEST_F(MobileNetV2NhwcTests, FusedNhwcTest1) {
    TestMobileNetV2Nhwc("1/input_0.npy", "1/output_0.npy");
}

TEST_F(MobileNetV2NhwcTests, FusedNhwcTest2) {
    TestMobileNetV2Nhwc("2/input_0.npy", "2/output_0.npy");
}
