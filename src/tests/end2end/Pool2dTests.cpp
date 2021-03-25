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

class Pool2dTests : public WebnnTest {};

TEST_F(Pool2dTests, MaxPool2dDefault) {
    const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
    const ml::Operand x = utils::BuildInput(builder, "x", {1, 1, 4, 4});
    utils::Pool2dOptions options;
    options.windowDimensions = {3, 3};
    const ml::Operand y = builder.MaxPool2d(x, options.AsPtr());
    const ml::Graph graph = utils::AwaitBuild(builder, {{"y", y}});
    ASSERT_TRUE(graph);
    const std::vector<float> dataX = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    const ml::Input inputX = {dataX.data(), dataX.size() * sizeof(float)};
    const ml::Result result = utils::AwaitCompute(graph, {{"x", inputX}}).Get("y");
    EXPECT_TRUE(utils::CheckShape(result, {1, 1, 2, 2}));
    const std::vector<float> expectedValue({11, 12, 15, 16});
    EXPECT_TRUE(utils::CheckValue(result, expectedValue));
}

TEST_F(Pool2dTests, MaxPool2dNhwc) {
    const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
    const ml::Operand x = utils::BuildInput(builder, "x", {1, 4, 4, 1});
    utils::Pool2dOptions options;
    options.windowDimensions = {3, 3};
    options.layout = ml::InputOperandLayout::Nhwc;
    const ml::Operand y = builder.MaxPool2d(x, options.AsPtr());
    const ml::Graph graph = utils::AwaitBuild(builder, {{"y", y}});
    ASSERT_TRUE(graph);
    const std::vector<float> dataX = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    const ml::Input inputX = {dataX.data(), dataX.size() * sizeof(float)};
    const ml::Result result = utils::AwaitCompute(graph, {{"x", inputX}}).Get("y");
    EXPECT_TRUE(utils::CheckShape(result, {1, 2, 2, 1}));
    const std::vector<float> expectedValue({11, 12, 15, 16});
    EXPECT_TRUE(utils::CheckValue(result, expectedValue));
}

TEST_F(Pool2dTests, MaxPool2dDilationsDefault) {
    const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
    const ml::Operand x = utils::BuildInput(builder, "x", {1, 1, 4, 4});
    utils::Pool2dOptions options;
    options.windowDimensions = {2, 2};
    options.dilations = {2, 2};
    const ml::Operand y = builder.MaxPool2d(x, options.AsPtr());
    const ml::Graph graph = utils::AwaitBuild(builder, {{"y", y}});
    ASSERT_TRUE(graph);
    const std::vector<float> dataX = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    const ml::Input inputX = {dataX.data(), dataX.size() * sizeof(float)};
    const ml::Result result = utils::AwaitCompute(graph, {{"x", inputX}}).Get("y");
    EXPECT_TRUE(utils::CheckShape(result, {1, 1, 2, 2}));
    const std::vector<float> expectedValue({11, 12, 15, 16});
    EXPECT_TRUE(utils::CheckValue(result, expectedValue));
}

TEST_F(Pool2dTests, MaxPool2dDilationsNhwc) {
    const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
    const ml::Operand x = utils::BuildInput(builder, "x", {1, 4, 4, 1});
    utils::Pool2dOptions options;
    options.windowDimensions = {2, 2};
    options.dilations = {2, 2};
    options.layout = ml::InputOperandLayout::Nhwc;
    const ml::Operand y = builder.MaxPool2d(x, options.AsPtr());
    const ml::Graph graph = utils::AwaitBuild(builder, {{"y", y}});
    ASSERT_TRUE(graph);
    const std::vector<float> dataX = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    const ml::Input inputX = {dataX.data(), dataX.size() * sizeof(float)};
    const ml::Result result = utils::AwaitCompute(graph, {{"x", inputX}}).Get("y");
    EXPECT_TRUE(utils::CheckShape(result, {1, 2, 2, 1}));
    const std::vector<float> expectedValue({11, 12, 15, 16});
    EXPECT_TRUE(utils::CheckValue(result, expectedValue));
}

TEST_F(Pool2dTests, MaxPool2dPadsDefault) {
    const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
    const ml::Operand x = utils::BuildInput(builder, "x", {1, 1, 5, 5});
    utils::Pool2dOptions options;
    options.windowDimensions = {5, 5};
    options.padding = {2, 2, 2, 2};
    const ml::Operand y = builder.MaxPool2d(x, options.AsPtr());
    const ml::Graph graph = utils::AwaitBuild(builder, {{"y", y}});
    ASSERT_TRUE(graph);
    const std::vector<float> dataX = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
    const ml::Input inputX = {dataX.data(), dataX.size() * sizeof(float)};
    const ml::Result result = utils::AwaitCompute(graph, {{"x", inputX}}).Get("y");
    EXPECT_TRUE(utils::CheckShape(result, {1, 1, 5, 5}));
    const std::vector<float> expectedValue({13, 14, 15, 15, 15, 18, 19, 20, 20, 20, 23, 24, 25,
                                            25, 25, 23, 24, 25, 25, 25, 23, 24, 25, 25, 25});
    EXPECT_TRUE(utils::CheckValue(result, expectedValue));
}

TEST_F(Pool2dTests, MaxPool2dPadsNhwc) {
    const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
    const ml::Operand x = utils::BuildInput(builder, "x", {1, 5, 5, 1});
    utils::Pool2dOptions options;
    options.windowDimensions = {5, 5};
    options.padding = {2, 2, 2, 2};
    options.layout = ml::InputOperandLayout::Nhwc;
    const ml::Operand y = builder.MaxPool2d(x, options.AsPtr());
    const ml::Graph graph = utils::AwaitBuild(builder, {{"y", y}});
    ASSERT_TRUE(graph);
    const std::vector<float> dataX = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
    const ml::Input inputX = {dataX.data(), dataX.size() * sizeof(float)};
    const ml::Result result = utils::AwaitCompute(graph, {{"x", inputX}}).Get("y");
    EXPECT_TRUE(utils::CheckShape(result, {1, 5, 5, 1}));
    const std::vector<float> expectedValue({13, 14, 15, 15, 15, 18, 19, 20, 20, 20, 23, 24, 25,
                                            25, 25, 23, 24, 25, 25, 25, 23, 24, 25, 25, 25});
    EXPECT_TRUE(utils::CheckValue(result, expectedValue));
}

TEST_F(Pool2dTests, MaxPool2dAutoPadSameUpperDefault) {
    const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
    const ml::Operand x = utils::BuildInput(builder, "x", {1, 1, 5, 5});
    utils::Pool2dOptions options;
    options.windowDimensions = {5, 5};
    options.autoPad = ml::AutoPad::SameUpper;
    const ml::Operand y = builder.MaxPool2d(x, options.AsPtr());
    const ml::Graph graph = utils::AwaitBuild(builder, {{"y", y}});
    ASSERT_TRUE(graph);
    const std::vector<float> dataX = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
    const ml::Input inputX = {dataX.data(), dataX.size() * sizeof(float)};
    const ml::Result result = utils::AwaitCompute(graph, {{"x", inputX}}).Get("y");
    EXPECT_TRUE(utils::CheckShape(result, {1, 1, 5, 5}));
    const std::vector<float> expectedValue({13, 14, 15, 15, 15, 18, 19, 20, 20, 20, 23, 24, 25,
                                            25, 25, 23, 24, 25, 25, 25, 23, 24, 25, 25, 25});
    EXPECT_TRUE(utils::CheckValue(result, expectedValue));
}

TEST_F(Pool2dTests, MaxPool2dAutoPadSameUpperNhwc) {
    const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
    const ml::Operand x = utils::BuildInput(builder, "x", {1, 5, 5, 1});
    utils::Pool2dOptions options;
    options.windowDimensions = {5, 5};
    options.autoPad = ml::AutoPad::SameUpper;
    options.layout = ml::InputOperandLayout::Nhwc;
    const ml::Operand y = builder.MaxPool2d(x, options.AsPtr());
    const ml::Graph graph = utils::AwaitBuild(builder, {{"y", y}});
    ASSERT_TRUE(graph);
    const std::vector<float> dataX = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
    const ml::Input inputX = {dataX.data(), dataX.size() * sizeof(float)};
    const ml::Result result = utils::AwaitCompute(graph, {{"x", inputX}}).Get("y");
    EXPECT_TRUE(utils::CheckShape(result, {1, 5, 5, 1}));
    const std::vector<float> expectedValue({13, 14, 15, 15, 15, 18, 19, 20, 20, 20, 23, 24, 25,
                                            25, 25, 23, 24, 25, 25, 25, 23, 24, 25, 25, 25});
    EXPECT_TRUE(utils::CheckValue(result, expectedValue));
}

TEST_F(Pool2dTests, MaxPool2dStridesDefault) {
    const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
    const ml::Operand x = utils::BuildInput(builder, "x", {1, 1, 5, 5});
    utils::Pool2dOptions options;
    options.windowDimensions = {2, 2};
    options.strides = {2, 2};
    const ml::Operand y = builder.MaxPool2d(x, options.AsPtr());
    const ml::Graph graph = utils::AwaitBuild(builder, {{"y", y}});
    ASSERT_TRUE(graph);
    const std::vector<float> dataX = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
    const ml::Input inputX = {dataX.data(), dataX.size() * sizeof(float)};
    const ml::Result result = utils::AwaitCompute(graph, {{"x", inputX}}).Get("y");
    EXPECT_TRUE(utils::CheckShape(result, {1, 1, 2, 2}));
    const std::vector<float> expectedValue({7, 9, 17, 19});
    EXPECT_TRUE(utils::CheckValue(result, expectedValue));
}

TEST_F(Pool2dTests, MaxPool2dStridesNhwc) {
    const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
    const ml::Operand x = utils::BuildInput(builder, "x", {1, 5, 5, 1});
    utils::Pool2dOptions options;
    options.windowDimensions = {2, 2};
    options.strides = {2, 2};
    options.layout = ml::InputOperandLayout::Nhwc;
    const ml::Operand y = builder.MaxPool2d(x, options.AsPtr());
    const ml::Graph graph = utils::AwaitBuild(builder, {{"y", y}});
    ASSERT_TRUE(graph);
    const std::vector<float> dataX = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
    const ml::Input inputX = {dataX.data(), dataX.size() * sizeof(float)};
    const ml::Result result = utils::AwaitCompute(graph, {{"x", inputX}}).Get("y");
    EXPECT_TRUE(utils::CheckShape(result, {1, 2, 2, 1}));
    const std::vector<float> expectedValue({7, 9, 17, 19});
    EXPECT_TRUE(utils::CheckValue(result, expectedValue));
}

TEST_F(Pool2dTests, AveragePool2dDefault) {
    const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
    const ml::Operand x = utils::BuildInput(builder, "x", {1, 1, 4, 4});
    utils::Pool2dOptions options;
    options.windowDimensions = {3, 3};
    const ml::Operand y = builder.AveragePool2d(x, options.AsPtr());
    const ml::Graph graph = utils::AwaitBuild(builder, {{"y", y}});
    ASSERT_TRUE(graph);
    const std::vector<float> dataX = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    const ml::Input inputX = {dataX.data(), dataX.size() * sizeof(float)};
    const ml::Result result = utils::AwaitCompute(graph, {{"x", inputX}}).Get("y");
    EXPECT_TRUE(utils::CheckShape(result, {1, 1, 2, 2}));
    const std::vector<float> expectedValue({6, 7, 10, 11});
    EXPECT_TRUE(utils::CheckValue(result, expectedValue));
}

TEST_F(Pool2dTests, AveragePool2dNhwc) {
    const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
    const ml::Operand x = utils::BuildInput(builder, "x", {1, 4, 4, 1});
    utils::Pool2dOptions options;
    options.windowDimensions = {3, 3};
    options.layout = ml::InputOperandLayout::Nhwc;
    const ml::Operand y = builder.AveragePool2d(x, options.AsPtr());
    const ml::Graph graph = utils::AwaitBuild(builder, {{"y", y}});
    ASSERT_TRUE(graph);
    const std::vector<float> dataX = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    const ml::Input inputX = {dataX.data(), dataX.size() * sizeof(float)};
    const ml::Result result = utils::AwaitCompute(graph, {{"x", inputX}}).Get("y");
    EXPECT_TRUE(utils::CheckShape(result, {1, 2, 2, 1}));
    const std::vector<float> expectedValue({6, 7, 10, 11});
    EXPECT_TRUE(utils::CheckValue(result, expectedValue));
}

TEST_F(Pool2dTests, AveragePool2dPadsDefault) {
    const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
    const ml::Operand x = utils::BuildInput(builder, "x", {1, 1, 5, 5});
    utils::Pool2dOptions options;
    options.windowDimensions = {5, 5};
    options.padding = {2, 2, 2, 2};
    const ml::Operand y = builder.AveragePool2d(x, options.AsPtr());
    const ml::Graph graph = utils::AwaitBuild(builder, {{"y", y}});
    ASSERT_TRUE(graph);
    const std::vector<float> dataX = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
    const ml::Input inputX = {dataX.data(), dataX.size() * sizeof(float)};
    const ml::Result result = utils::AwaitCompute(graph, {{"x", inputX}}).Get("y");
    EXPECT_TRUE(utils::CheckShape(result, {1, 1, 5, 5}));
    const std::vector<float> expectedValue({7,    7.5,  8,    8.5,  9,    9.5,  10,   10.5, 11,
                                            11.5, 12,   12.5, 13,   13.5, 14,   14.5, 15,   15.5,
                                            16,   16.5, 17,   17.5, 18,   18.5, 19});
    EXPECT_TRUE(utils::CheckValue(result, expectedValue));
}

TEST_F(Pool2dTests, AveragePool2dPadsNhwc) {
    const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
    const ml::Operand x = utils::BuildInput(builder, "x", {1, 5, 5, 1});
    utils::Pool2dOptions options;
    options.windowDimensions = {5, 5};
    options.padding = {2, 2, 2, 2};
    options.layout = ml::InputOperandLayout::Nhwc;
    const ml::Operand y = builder.AveragePool2d(x, options.AsPtr());
    const ml::Graph graph = utils::AwaitBuild(builder, {{"y", y}});
    ASSERT_TRUE(graph);
    const std::vector<float> dataX = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
    const ml::Input inputX = {dataX.data(), dataX.size() * sizeof(float)};
    const ml::Result result = utils::AwaitCompute(graph, {{"x", inputX}}).Get("y");
    EXPECT_TRUE(utils::CheckShape(result, {1, 5, 5, 1}));
    const std::vector<float> expectedValue({7,    7.5,  8,    8.5,  9,    9.5,  10,   10.5, 11,
                                            11.5, 12,   12.5, 13,   13.5, 14,   14.5, 15,   15.5,
                                            16,   16.5, 17,   17.5, 18,   18.5, 19});
    EXPECT_TRUE(utils::CheckValue(result, expectedValue));
}

TEST_F(Pool2dTests, AveragePool2dAutoPadSameUpperDefault) {
    const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
    const ml::Operand x = utils::BuildInput(builder, "x", {1, 1, 5, 5});
    utils::Pool2dOptions options;
    options.windowDimensions = {5, 5};
    options.autoPad = ml::AutoPad::SameUpper;
    const ml::Operand y = builder.AveragePool2d(x, options.AsPtr());
    const ml::Graph graph = utils::AwaitBuild(builder, {{"y", y}});
    ASSERT_TRUE(graph);
    const std::vector<float> dataX = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
    const ml::Input inputX = {dataX.data(), dataX.size() * sizeof(float)};
    const ml::Result result = utils::AwaitCompute(graph, {{"x", inputX}}).Get("y");
    EXPECT_TRUE(utils::CheckShape(result, {1, 1, 5, 5}));
    const std::vector<float> expectedValue({7,    7.5,  8,    8.5,  9,    9.5,  10,   10.5, 11,
                                            11.5, 12,   12.5, 13,   13.5, 14,   14.5, 15,   15.5,
                                            16,   16.5, 17,   17.5, 18,   18.5, 19});
    EXPECT_TRUE(utils::CheckValue(result, expectedValue));
}

TEST_F(Pool2dTests, AveragePool2dAutoPadSameUpperNhwc) {
    const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
    const ml::Operand x = utils::BuildInput(builder, "x", {1, 5, 5, 1});
    utils::Pool2dOptions options;
    options.windowDimensions = {5, 5};
    options.autoPad = ml::AutoPad::SameUpper;
    options.layout = ml::InputOperandLayout::Nhwc;
    const ml::Operand y = builder.AveragePool2d(x, options.AsPtr());
    const ml::Graph graph = utils::AwaitBuild(builder, {{"y", y}});
    ASSERT_TRUE(graph);
    const std::vector<float> dataX = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
    const ml::Input inputX = {dataX.data(), dataX.size() * sizeof(float)};
    const ml::Result result = utils::AwaitCompute(graph, {{"x", inputX}}).Get("y");
    EXPECT_TRUE(utils::CheckShape(result, {1, 5, 5, 1}));
    const std::vector<float> expectedValue({7,    7.5,  8,    8.5,  9,    9.5,  10,   10.5, 11,
                                            11.5, 12,   12.5, 13,   13.5, 14,   14.5, 15,   15.5,
                                            16,   16.5, 17,   17.5, 18,   18.5, 19});
    EXPECT_TRUE(utils::CheckValue(result, expectedValue));
}

TEST_F(Pool2dTests, AveragePool2dStridesDefault) {
    const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
    const ml::Operand x = utils::BuildInput(builder, "x", {1, 1, 5, 5});
    utils::Pool2dOptions options;
    options.windowDimensions = {2, 2};
    options.strides = {2, 2};
    const ml::Operand y = builder.AveragePool2d(x, options.AsPtr());
    const ml::Graph graph = utils::AwaitBuild(builder, {{"y", y}});
    ASSERT_TRUE(graph);
    const std::vector<float> dataX = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
    const ml::Input inputX = {dataX.data(), dataX.size() * sizeof(float)};
    const ml::Result result = utils::AwaitCompute(graph, {{"x", inputX}}).Get("y");
    EXPECT_TRUE(utils::CheckShape(result, {1, 1, 2, 2}));
    const std::vector<float> expectedValue({4, 6, 14, 16});
    EXPECT_TRUE(utils::CheckValue(result, expectedValue));
}

TEST_F(Pool2dTests, AveragePool2dStridesNhwc) {
    const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
    const ml::Operand x = utils::BuildInput(builder, "x", {1, 5, 5, 1});
    utils::Pool2dOptions options;
    options.windowDimensions = {2, 2};
    options.strides = {2, 2};
    options.layout = ml::InputOperandLayout::Nhwc;
    const ml::Operand y = builder.AveragePool2d(x, options.AsPtr());
    const ml::Graph graph = utils::AwaitBuild(builder, {{"y", y}});
    ASSERT_TRUE(graph);
    const std::vector<float> dataX = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
    const ml::Input inputX = {dataX.data(), dataX.size() * sizeof(float)};
    const ml::Result result = utils::AwaitCompute(graph, {{"x", inputX}}).Get("y");
    EXPECT_TRUE(utils::CheckShape(result, {1, 2, 2, 1}));
    const std::vector<float> expectedValue({4, 6, 14, 16});
    EXPECT_TRUE(utils::CheckValue(result, expectedValue));
}

TEST_F(Pool2dTests, GlobalAveragePool2dDefault) {
    const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
    const ml::Operand x = utils::BuildInput(builder, "x", {1, 3, 5, 5});
    const ml::Operand y = builder.AveragePool2d(x);
    const ml::Graph graph = utils::AwaitBuild(builder, {{"y", y}});
    ASSERT_TRUE(graph);
    const std::vector<float> dataX = {
        -1.1289884,  0.34016284,  0.497431,    2.1915932,   0.42038894,  -0.18261199, -0.15769927,
        -0.26465914, 0.03877424,  0.39492005,  -0.33410737, 0.74918455,  -1.3542547,  -0.0222946,
        0.7094626,   -0.09399617, 0.790736,    -0.75826526, 0.27656242,  0.46543223,  -1.2342638,
        1.1549494,   0.24823844,  0.75670505,  -1.7108902,  -1.4767597,  -1.4969662,  -0.31936142,
        0.5327554,   -0.06070877, 0.31212643,  2.2274113,   1.2775147,   0.59886885,  -1.5765078,
        0.18522178,  0.22655599,  0.88869494,  0.38609484,  -0.05860576, -0.72732115, -0.0046324,
        -1.3593693,  -0.6295078,  1.384531,    0.06825881,  0.19907428,  0.20298219,  -0.8399954,
        1.3583295,   0.02117888,  -1.0636739,  -0.30460566, -0.92678875, -0.09120782, -0.88333017,
        -0.9641269,  0.6065926,   -0.5830042,  -0.81138134, 1.3569402,   1.2891295,   0.2508177,
        0.20211531,  0.8832168,   -0.19886094, -0.61088,    0.682026,    -0.5253442,  1.5022339,
        1.0256356,   1.0642492,   -0.4169051,  -0.8740329,  1.1494869};
    const ml::Input inputX = {dataX.data(), dataX.size() * sizeof(float)};
    const ml::Result result = utils::AwaitCompute(graph, {{"x", inputX}}).Get("y");
    EXPECT_TRUE(utils::CheckShape(result, {1, 3, 1, 1}));
    const std::vector<float> expectedValue({0.07170041, 0.05194739, 0.07117923});
    EXPECT_TRUE(utils::CheckValue(result, expectedValue));
}

TEST_F(Pool2dTests, GlobalAveragePool2dNhwc) {
    const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
    const ml::Operand x = utils::BuildInput(builder, "x", {1, 5, 5, 3});
    utils::Pool2dOptions options;
    options.layout = ml::InputOperandLayout::Nhwc;
    const ml::Operand y = builder.AveragePool2d(x, options.AsPtr());
    const ml::Graph graph = utils::AwaitBuild(builder, {{"y", y}});
    ASSERT_TRUE(graph);
    const std::vector<float> dataX = {
        -1.1289884,  -1.4767597,  0.02117888,  0.34016284,  -1.4969662,  -1.0636739,  0.497431,
        -0.31936142, -0.30460566, 2.1915932,   0.5327554,   -0.92678875, 0.42038894,  -0.06070877,
        -0.09120782, -0.18261199, 0.31212643,  -0.88333017, -0.15769927, 2.2274113,   -0.9641269,
        -0.26465914, 1.2775147,   0.6065926,   0.03877424,  0.59886885,  -0.5830042,  0.39492005,
        -1.5765078,  -0.81138134, -0.33410737, 0.18522178,  1.3569402,   0.74918455,  0.22655599,
        1.2891295,   -1.3542547,  0.88869494,  0.2508177,   -0.0222946,  0.38609484,  0.20211531,
        0.7094626,   -0.05860576, 0.8832168,   -0.09399617, -0.72732115, -0.19886094, 0.790736,
        -0.0046324,  -0.61088,    -0.75826526, -1.3593693,  0.682026,    0.27656242,  -0.6295078,
        -0.5253442,  0.46543223,  1.384531,    1.5022339,   -1.2342638,  0.06825881,  1.0256356,
        1.1549494,   0.19907428,  1.0642492,   0.24823844,  0.20298219,  -0.4169051,  0.75670505,
        -0.8399954,  -0.8740329,  -1.7108902,  1.3583295,   1.1494869};
    const ml::Input inputX = {dataX.data(), dataX.size() * sizeof(float)};
    const ml::Result result = utils::AwaitCompute(graph, {{"x", inputX}}).Get("y");
    EXPECT_TRUE(utils::CheckShape(result, {1, 1, 1, 3}));
    const std::vector<float> expectedValue({0.07170041, 0.05194739, 0.07117923});
    EXPECT_TRUE(utils::CheckValue(result, expectedValue));
}
