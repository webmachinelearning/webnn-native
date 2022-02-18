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

static const std::string kModelPath = WEBNN_END2END_TEST_MODEL_PATH;

class MobileNetV2BatchNormNchwTests : public WebnnTest {
  public:
    void TestMobileNetV2Nchw(const std::string& inputFile,
                             const std::string& expectedFile,
                             bool fused = true) {
        MobileNetV2 mobilenetv2;
        mobilenetv2.mFused = fused;
        const std::string nchwPath = kModelPath + "/mobilenetv2_batchnorm_nchw/";
        mobilenetv2.mWeightsPath = nchwPath + "weights/";
        const wnn::GraphBuilder builder = wnn::CreateGraphBuilder(GetContext());
        wnn::Operand output = mobilenetv2.LoadBatchNormNCHW(builder, false);
        wnn::Graph graph = utils::Build(builder, {{"output", output}});
        const cnpy::NpyArray inputNpy = cnpy::npy_load(nchwPath + "test_data_set/" + inputFile);
        const std::vector<float> inputData = inputNpy.as_vec<float>();
        std::vector<float> result(utils::SizeOfShape({1, 1000}));
        utils::Compute(graph, {{"input", inputData}}, {{"output", result}});
        const cnpy::NpyArray outputNpy = cnpy::npy_load(nchwPath + "test_data_set/" + expectedFile);
        EXPECT_TRUE(utils::CheckValue(result, outputNpy.as_vec<float>()));
    }
};

TEST_F(MobileNetV2BatchNormNchwTests, NchwTest0) {
    TestMobileNetV2Nchw("0/input_0.npy", "0/output_0.npy", false);
}

TEST_F(MobileNetV2BatchNormNchwTests, NchwTest1) {
    TestMobileNetV2Nchw("1/input_0.npy", "1/output_0.npy", false);
}

TEST_F(MobileNetV2BatchNormNchwTests, NchwTest2) {
    TestMobileNetV2Nchw("2/input_0.npy", "2/output_0.npy", false);
}

TEST_F(MobileNetV2BatchNormNchwTests, FusedNchwTest0) {
    TestMobileNetV2Nchw("0/input_0.npy", "0/output_0.npy");
}

TEST_F(MobileNetV2BatchNormNchwTests, FusedNchwTest1) {
    TestMobileNetV2Nchw("1/input_0.npy", "1/output_0.npy");
}

TEST_F(MobileNetV2BatchNormNchwTests, FusedNchwTest2) {
    TestMobileNetV2Nchw("2/input_0.npy", "2/output_0.npy");
}