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

#include "src/tests/WebnnTest.h"

class ResampleTests : public WebnnTest {
  public:
    void TestResample(const std::vector<int32_t>& inputShape,
                      const std::vector<float>& inputData,
                      const std::vector<int32_t>& expectedShape,
                      const std::vector<float>& expectedValue,
                      const ml::ResampleOptions* options = nullptr) {
        const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
        const ml::Operand inputOperand = utils::BuildInput(builder, "input", inputShape);
        const ml::Operand output = builder.Resample(inputOperand, options);
        const ml::Graph graph = utils::Build(builder, {{"output", output}});
        ASSERT_TRUE(graph);
        std::vector<float> result(utils::SizeOfShape(expectedShape));
        utils::Compute(graph, {{"input", inputData}}, {{"output", result}});
        EXPECT_TRUE(utils::CheckValue(result, expectedValue));
    }
};

TEST_F(ResampleTests, UpsampleLinear) {
    const std::vector<int32_t> inputShape = {1, 1, 2, 2};
    const std::vector<float> inputData = {1, 2, 3, 4};
    const std::vector<int32_t> expectedShape = {1, 1, 4, 4};
    const std::vector<float> expectedValue = {
        1., 1.25, 1.75, 2., 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3., 3.25, 3.75, 4.,
    };

    ml::ResampleOptions options;
    options.mode = ml::InterpolationMode::Linear;
    std::vector<float> scales = {1.0, 1.0, 2.0, 2.0};
    options.scalesCount = scales.size();
    options.scales = scales.data();
    TestResample(inputShape, inputData, expectedShape, expectedValue, &options);

    options = {};
    options.mode = ml::InterpolationMode::Linear;
    std::vector<int32_t> sizes = {1, 1, 4, 4};
    options.sizesCount = sizes.size();
    options.sizes = sizes.data();
    TestResample(inputShape, inputData, expectedShape, expectedValue, &options);
}

TEST_F(ResampleTests, UpsampleNearest) {
    const std::vector<int32_t> inputShape = {1, 1, 2, 2};
    const std::vector<float> inputData = {1, 2, 3, 4};
    const std::vector<int32_t> expectedShape = {1, 1, 4, 6};
    const std::vector<float> expectedValue = {
        1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 3, 3, 3, 4, 4, 4,
    };

    ml::ResampleOptions options;
    options.mode = ml::InterpolationMode::NearestNeighbor;
    std::vector<float> scales = {1.0, 1.0, 2.0, 3.0};
    options.scalesCount = scales.size();
    options.scales = scales.data();
    TestResample(inputShape, inputData, expectedShape, expectedValue, &options);

    options = {};
    options.mode = ml::InterpolationMode::NearestNeighbor;
    std::vector<int32_t> sizes = {1, 1, 4, 6};
    options.sizesCount = sizes.size();
    options.sizes = sizes.data();
    TestResample(inputShape, inputData, expectedShape, expectedValue, &options);
}
