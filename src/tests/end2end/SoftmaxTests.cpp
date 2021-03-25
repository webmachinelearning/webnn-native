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

class SoftmaxTests : public WebnnTest {};

TEST_F(SoftmaxTests, Softmax) {
    const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
    const ml::Operand a = utils::BuildInput(builder, "a", {3, 4});
    const ml::Operand b = builder.Softmax(a);
    const ml::Graph graph = utils::AwaitBuild(builder, {{"b", b}});
    ASSERT_TRUE(graph);
    const std::vector<float> inputData = {0.4301911,   0.54719144,  -1.1637765, 0.18390046,
                                          0.58390397,  0.1735679,   0.539724,   -0.953514,
                                          -0.59202826, -0.17344485, 0.14395015, -0.37920907};
    const ml::Input input = {inputData.data(), inputData.size() * sizeof(float)};
    const ml::Result result = utils::AwaitCompute(graph, {{"a", input}}).Get("b");
    EXPECT_TRUE(utils::CheckShape(result, {3, 4}));
    const std::vector<float> expectedData = {0.32165375, 0.36157736, 0.0653337,  0.25143513,
                                             0.35271573, 0.23400122, 0.33747196, 0.07581109,
                                             0.17110129, 0.26004094, 0.35717794, 0.21167983};
    EXPECT_TRUE(utils::CheckValue(result, expectedData));
}