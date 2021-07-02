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

class ReduceMeanTests : public WebnnTest {
  protected:
    void CheckReduceMean(const std::vector<int32_t>& inputShape,
                         const std::vector<float>& inputData,
                         const std::vector<int32_t>& expectedShape,
                         const std::vector<float>& expectedValue,
                         const std::vector<int32_t>& axes = {},
                         bool keepDimensions = false) {
        const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
        const ml::Operand a = utils::BuildInput(builder, "a", inputShape);
        ml::ReduceMeanOptions options;
        if (!axes.empty()) {
            options.axes = axes.data();
            options.axesCount = axes.size();
        }
        if (keepDimensions) {
            options.keepDimensions = keepDimensions;
        }
        const ml::Operand b = builder.ReduceMean(a, &options);
        const ml::Graph graph = utils::AwaitBuild(builder, {{"b", b}});
        ASSERT_TRUE(graph);
        const ml::Input input = {inputData.data(), inputData.size() * sizeof(float)};
        const ml::Result result = utils::AwaitCompute(graph, {{"a", input}}).Get("b");
        EXPECT_TRUE(utils::CheckShape(result, expectedShape));
        EXPECT_TRUE(utils::CheckValue(result, expectedValue));
    }
};

TEST_F(ReduceMeanTests, ReduceMeanDefault) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.};
    const std::vector<int32_t> expectedShape = {};
    const std::vector<float> expectedValue = {18.25};
    CheckReduceMean(inputShape, inputData, expectedShape, expectedValue, {});
}

TEST_F(ReduceMeanTests, ReduceMeanDefaultAxesKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.};
    const std::vector<int32_t> expectedShape = {1, 1, 1};
    const std::vector<float> expectedValue = {18.25};
    CheckReduceMean(inputShape, inputData, expectedShape, expectedValue, {}, true);
}

TEST_F(ReduceMeanTests, ReduceMeanAxes0NotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.};
    const std::vector<int32_t> expectedShape = {2, 2};
    const std::vector<float> expectedValue = {30., 1., 40., 2.};
    CheckReduceMean(inputShape, inputData, expectedShape, expectedValue, {0});
}

TEST_F(ReduceMeanTests, ReduceMeanAxes1NotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.};
    const std::vector<int32_t> expectedShape = {3, 2};
    const std::vector<float> expectedValue = {12.5, 1.5, 35., 1.5, 57.5, 1.5};
    CheckReduceMean(inputShape, inputData, expectedShape, expectedValue, {1});
}

TEST_F(ReduceMeanTests, ReduceMeanAxes2NotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.};
    const std::vector<int32_t> expectedShape = {3, 2};
    const std::vector<float> expectedValue = {3., 11., 15.5, 21., 28., 31.};
    CheckReduceMean(inputShape, inputData, expectedShape, expectedValue, {2});
}

TEST_F(ReduceMeanTests, ReduceMeanNegativeAxesNotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.};
    const std::vector<int32_t> expectedShape = {3, 2};
    const std::vector<float> expectedValue = {3., 11., 15.5, 21., 28., 31.};
    CheckReduceMean(inputShape, inputData, expectedShape, expectedValue, {-1});
}

TEST_F(ReduceMeanTests, ReduceMeanAxes0KeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.};
    const std::vector<int32_t> expectedShape = {1, 2, 2};
    const std::vector<float> expectedValue = {30., 1., 40., 2.};
    CheckReduceMean(inputShape, inputData, expectedShape, expectedValue, {0}, true);
}

TEST_F(ReduceMeanTests, ReduceMeanAxes1KeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.};
    const std::vector<int32_t> expectedShape = {3, 1, 2};
    const std::vector<float> expectedValue = {12.5, 1.5, 35., 1.5, 57.5, 1.5};
    CheckReduceMean(inputShape, inputData, expectedShape, expectedValue, {1}, true);
}

TEST_F(ReduceMeanTests, ReduceMeanAxes2KeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.};
    const std::vector<int32_t> expectedShape = {3, 2, 1};
    const std::vector<float> expectedValue = {3., 11., 15.5, 21., 28., 31.};
    CheckReduceMean(inputShape, inputData, expectedShape, expectedValue, {2}, true);
}

TEST_F(ReduceMeanTests, ReduceMeanNegativeAxesKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.};
    const std::vector<int32_t> expectedShape = {3, 2, 1};
    const std::vector<float> expectedValue = {3., 11., 15.5, 21., 28., 31.};
    CheckReduceMean(inputShape, inputData, expectedShape, expectedValue, {-1}, true);
}