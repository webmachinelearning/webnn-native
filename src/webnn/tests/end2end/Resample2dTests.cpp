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

#include "webnn/tests/WebnnTest.h"

class Resample2dTests : public WebnnTest {
  public:
    void TestResample2d(const std::vector<int32_t>& inputShape,
                        const std::vector<float>& inputData,
                        const std::vector<int32_t>& expectedShape,
                        const std::vector<float>& expectedValue,
                        const wnn::Resample2dOptions* options = nullptr) {
        const wnn::GraphBuilder builder = utils::CreateGraphBuilder(GetContext());
        const wnn::Operand inputOperand = utils::BuildInput(builder, "input", inputShape);
        const wnn::Operand output = builder.Resample2d(inputOperand, options);
        const wnn::Graph graph = utils::Build(builder, {{"output", output}});
        ASSERT_TRUE(graph);
        std::vector<float> result(utils::SizeOfShape(expectedShape));
        utils::Compute(graph, {{"input", inputData}}, {{"output", result}});
        EXPECT_TRUE(utils::CheckValue(result, expectedValue));
    }
};

TEST_F(Resample2dTests, UpsampleLinear) {
    const std::vector<int32_t> inputShape = {1, 1, 2, 2};
    const std::vector<float> inputData = {1, 2, 3, 4};
    const std::vector<int32_t> expectedShape = {1, 1, 4, 4};
    const std::vector<float> expectedValue = {
        1., 1.25, 1.75, 2., 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3., 3.25, 3.75, 4.,
    };

    wnn::Resample2dOptions options;
    options.mode = wnn::InterpolationMode::Linear;
    std::vector<float> scales = {2.0, 2.0};
    options.scalesCount = scales.size();
    options.scales = scales.data();
    TestResample2d(inputShape, inputData, expectedShape, expectedValue, &options);

    options = {};
    options.mode = wnn::InterpolationMode::Linear;
    std::vector<int32_t> sizes = {4, 4};
    options.sizesCount = sizes.size();
    options.sizes = sizes.data();
    TestResample2d(inputShape, inputData, expectedShape, expectedValue, &options);
}

TEST_F(Resample2dTests, UpsampleLinearWithAxes01) {
    const std::vector<int32_t> inputShape = {2, 2, 1, 1};
    const std::vector<float> inputData = {1, 2, 3, 4};
    const std::vector<int32_t> expectedShape = {4, 4, 1, 1};
    const std::vector<float> expectedValue = {
        1., 1.25, 1.75, 2., 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3., 3.25, 3.75, 4.,
    };

    wnn::Resample2dOptions options;
    options.mode = wnn::InterpolationMode::Linear;
    std::vector<float> scales = {2.0, 2.0};
    options.scalesCount = scales.size();
    options.scales = scales.data();
    std::vector<int32_t> axes = {0, 1};
    options.axesCount = axes.size();
    options.axes = axes.data();
    TestResample2d(inputShape, inputData, expectedShape, expectedValue, &options);

    options = {};
    options.mode = wnn::InterpolationMode::Linear;
    std::vector<int32_t> sizes = {4, 4};
    options.sizesCount = sizes.size();
    options.sizes = sizes.data();
    options.axesCount = axes.size();
    options.axes = axes.data();
    TestResample2d(inputShape, inputData, expectedShape, expectedValue, &options);
}

TEST_F(Resample2dTests, UpsampleLinearWithAxes12) {
    const std::vector<int32_t> inputShape = {1, 2, 2, 1};
    const std::vector<float> inputData = {1, 2, 3, 4};
    const std::vector<int32_t> expectedShape = {1, 4, 4, 1};
    const std::vector<float> expectedValue = {
        1., 1.25, 1.75, 2., 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3., 3.25, 3.75, 4.,
    };

    wnn::Resample2dOptions options;
    options.mode = wnn::InterpolationMode::Linear;
    std::vector<float> scales = {2.0, 2.0};
    options.scalesCount = scales.size();
    options.scales = scales.data();
    std::vector<int32_t> axes = {1, 2};
    options.axesCount = axes.size();
    options.axes = axes.data();
    TestResample2d(inputShape, inputData, expectedShape, expectedValue, &options);

    options = {};
    options.mode = wnn::InterpolationMode::Linear;
    std::vector<int32_t> sizes = {4, 4};
    options.sizesCount = sizes.size();
    options.sizes = sizes.data();
    options.axesCount = axes.size();
    options.axes = axes.data();
    TestResample2d(inputShape, inputData, expectedShape, expectedValue, &options);
}

TEST_F(Resample2dTests, UpsampleLinearWithAxes23) {
    const std::vector<int32_t> inputShape = {1, 1, 2, 2};
    const std::vector<float> inputData = {1, 2, 3, 4};
    const std::vector<int32_t> expectedShape = {1, 1, 4, 4};
    const std::vector<float> expectedValue = {
        1., 1.25, 1.75, 2., 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3., 3.25, 3.75, 4.,
    };

    wnn::Resample2dOptions options;
    options.mode = wnn::InterpolationMode::Linear;
    std::vector<float> scales = {2.0, 2.0};
    options.scalesCount = scales.size();
    options.scales = scales.data();
    std::vector<int32_t> axes = {2, 3};
    options.axesCount = axes.size();
    options.axes = axes.data();
    TestResample2d(inputShape, inputData, expectedShape, expectedValue, &options);

    options = {};
    options.mode = wnn::InterpolationMode::Linear;
    std::vector<int32_t> sizes = {4, 4};
    options.sizesCount = sizes.size();
    options.sizes = sizes.data();
    options.axesCount = axes.size();
    options.axes = axes.data();
    TestResample2d(inputShape, inputData, expectedShape, expectedValue, &options);
}

TEST_F(Resample2dTests, UpsampleSizeLinearIgnoredScales) {
    const std::vector<int32_t> inputShape = {1, 1, 2, 2};
    const std::vector<float> inputData = {1, 2, 3, 4};
    const std::vector<int32_t> expectedShape = {1, 1, 4, 4};
    const std::vector<float> expectedValue = {
        1., 1.25, 1.75, 2., 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3., 3.25, 3.75, 4.,
    };

    wnn::Resample2dOptions options;
    options.mode = wnn::InterpolationMode::Linear;
    std::vector<float> scales = {3.0, 4.0};
    options.scalesCount = scales.size();
    options.scales = scales.data();
    std::vector<int32_t> sizes = {4, 4};
    options.sizesCount = sizes.size();
    options.sizes = sizes.data();
    TestResample2d(inputShape, inputData, expectedShape, expectedValue, &options);
}

TEST_F(Resample2dTests, UpsampleNearest) {
    const std::vector<int32_t> inputShape = {1, 1, 2, 2};
    const std::vector<float> inputData = {1, 2, 3, 4};
    const std::vector<int32_t> expectedShape = {1, 1, 4, 6};
    const std::vector<float> expectedValue = {
        1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 3, 3, 3, 4, 4, 4,
    };

    wnn::Resample2dOptions options;
    options.mode = wnn::InterpolationMode::NearestNeighbor;
    std::vector<float> scales = {2.0, 3.0};
    options.scalesCount = scales.size();
    options.scales = scales.data();
    TestResample2d(inputShape, inputData, expectedShape, expectedValue, &options);

    options = {};
    options.mode = wnn::InterpolationMode::NearestNeighbor;
    std::vector<int32_t> sizes = {4, 6};
    options.sizesCount = sizes.size();
    options.sizes = sizes.data();
    TestResample2d(inputShape, inputData, expectedShape, expectedValue, &options);
}
