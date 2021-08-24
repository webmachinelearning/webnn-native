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

class SqueezeNetNchwTests : public WebnnTest {
  public:
    void TestSqueezeNetNchw(const std::string& inputFile,
                            const std::string& expectedFile,
                            bool fused = false) {
        SqueezeNet squeezenet;
        squeezenet.mFused = fused;
        const std::string nchwPath =
            "node/third_party/webnn-polyfill/test-data/models/squeezenet1.1_nchw/";
        squeezenet.mWeightsPath = nchwPath + "weights/";
        const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
        ml::Operand output = squeezenet.LoadNCHW(builder, false);
        ml::Graph graph = utils::Build(builder, {{"output", output}});
        const cnpy::NpyArray inputNpy = cnpy::npy_load(nchwPath + "test_data_set/" + inputFile);
        const std::vector<float> inputData = inputNpy.as_vec<float>();
        std::vector<float> result(utils::SizeOfShape({1, 1000}));
        utils::Compute(graph, {{"input", inputData}}, {{"output", result}});
        const cnpy::NpyArray outputNpy = cnpy::npy_load(nchwPath + "test_data_set/" + expectedFile);
        EXPECT_TRUE(utils::CheckValue(result, outputNpy.as_vec<float>()));
    }
};

TEST_F(SqueezeNetNchwTests, NchwTest0) {
    TestSqueezeNetNchw("0/input_0.npy", "0/output_0.npy");
}

TEST_F(SqueezeNetNchwTests, NchwTest1) {
    TestSqueezeNetNchw("1/input_0.npy", "1/output_0.npy");
}

TEST_F(SqueezeNetNchwTests, NchwTest2) {
    TestSqueezeNetNchw("2/input_0.npy", "2/output_0.npy");
}

TEST_F(SqueezeNetNchwTests, FusedNchwTest0) {
    TestSqueezeNetNchw("0/input_0.npy", "0/output_0.npy", true);
}

TEST_F(SqueezeNetNchwTests, FusedNchwTest1) {
    TestSqueezeNetNchw("1/input_0.npy", "1/output_0.npy", true);
}

TEST_F(SqueezeNetNchwTests, FusedNchwTest2) {
    TestSqueezeNetNchw("2/input_0.npy", "2/output_0.npy", true);
}
