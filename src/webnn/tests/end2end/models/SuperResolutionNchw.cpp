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

#include "examples/SuperResolution/SuperResolution.h"
#include "webnn/tests/WebnnTest.h"

static const std::string kModelPath = WEBNN_END2END_TEST_MODEL_PATH;

class SuperResolutionNchwTests : public WebnnTest {
  public:
    void TestSuperResolutionNchw(const std::string& inputFile,
                                 const std::string& expectedFile,
                                 bool fused = true) {
        SuperResolution superresolution;
        superresolution.mFused = true;
        const std::string nchwPath = kModelPath + "/super_resolution_nchw/";
        superresolution.mWeightsPath = nchwPath + "weights/";
        const wnn::GraphBuilder builder = wnn::CreateGraphBuilder(GetContext());
        wnn::Operand output = superresolution.LoadNchw(builder, false);
        wnn::Graph graph = utils::Build(builder, {{"output", output}});
        const cnpy::NpyArray inputNpy = cnpy::npy_load(nchwPath + "test_data_set/" + inputFile);
        const std::vector<float> inputData = inputNpy.as_vec<float>();
        std::vector<float> result(utils::SizeOfShape({/*TODO: batchSize?*/ 1, 1, 672, 672}));
        utils::Compute(GetContext(), graph, {{"input", inputData}}, {{"output", result}});
        const cnpy::NpyArray outputNpy = cnpy::npy_load(nchwPath + "test_data_set/" + expectedFile);
        EXPECT_TRUE(utils::CheckValue(result, outputNpy.as_vec<float>()));
    }
};

TEST_F(SuperResolutionNchwTests, NchwTest0) {
    TestSuperResolutionNchw("0/input_0.npy", "0/output_0.npy", false);
}
