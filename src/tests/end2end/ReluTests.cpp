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

class ReluTests : public WebnnTest {};

TEST_F(ReluTests, Relu) {
    const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
    const ml::Operand a = utils::BuildInput(builder, "a", {3, 4, 5});
    const ml::Operand b = builder.Relu(a);
    const ml::Graph graph = utils::AwaitBuild(builder, {{"b", b}});
    ASSERT_TRUE(graph);
    const std::vector<float> inputData = {
        -1.483762,   0.6447428,   -1.2266507,  -1.7132527,  0.9777725,   -0.34438756, -0.99921757,
        -1.2882805,  1.3725083,   -0.06386258, -0.44738683, -0.6776338,  0.5027815,   -1.0428967,
        -1.4220539,  0.00880813,  -1.2053454,  1.1644533,   -1.6577007,  -0.33448243, 0.69386536,
        0.06171616,  -0.20644434, 1.0620342,   -0.8824057,  -0.7676657,  0.7517342,   1.4035656,
        -0.29105335, 0.18367627,  1.3628657,   -0.39770076, -0.1550809,  -1.2575449,  0.5797014,
        -0.02414344, 0.9181723,   -1.1963434,  0.56652546,  -0.25052008, -0.02097719, -2.6274924,
        0.7993208,   -0.31359985, 0.9019325,   -0.02042965, 0.5222995,   1.3394557,   -1.0482218,
        1.1774449,   0.8999488,   -1.1143959,  1.0122099,   -0.48604885, -0.06009902, -0.1766853,
        1.4515465,   -0.7182982,  2.0361354,   0.7899623};
    const ml::Input input = {inputData.data(), inputData.size() * sizeof(float)};
    const ml::Result result = utils::AwaitCompute(graph, {{"a", input}}).Get("b");
    EXPECT_TRUE(utils::CheckShape(result, {3, 4, 5}));
    const std::vector<float> expectedData(
        {0.,        0.6447428, 0.,        0.,        0.9777725,  0.,         0.,         0.,
         1.3725083, 0.,        0.,        0.,        0.5027815,  0.,         0.,         0.00880813,
         0.,        1.1644533, 0.,        0.,        0.69386536, 0.06171616, 0.,         1.0620342,
         0.,        0.,        0.7517342, 1.4035656, 0.,         0.18367627, 1.3628657,  0.,
         0.,        0.,        0.5797014, 0.,        0.9181723,  0.,         0.56652546, 0.,
         0.,        0.,        0.7993208, 0.,        0.9019325,  0.,         0.5222995,  1.3394557,
         0.,        1.1774449, 0.8999488, 0.,        1.0122099,  0.,         0.,         0.,
         1.4515465, 0.,        2.0361354, 0.7899623});
    EXPECT_TRUE(utils::CheckValue(result, expectedData));
}

TEST_F(ReluTests, ReluWithPreallocatedBuffer) {
    const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
    const ml::Operand a = utils::BuildInput(builder, "a", {3, 4, 5});
    const ml::Operand b = builder.Relu(a);
    const ml::Graph graph = utils::AwaitBuild(builder, {{"b", b}});
    ASSERT_TRUE(graph);
    const std::vector<float> inputData = {
        -1.483762,   0.6447428,   -1.2266507,  -1.7132527,  0.9777725,   -0.34438756, -0.99921757,
        -1.2882805,  1.3725083,   -0.06386258, -0.44738683, -0.6776338,  0.5027815,   -1.0428967,
        -1.4220539,  0.00880813,  -1.2053454,  1.1644533,   -1.6577007,  -0.33448243, 0.69386536,
        0.06171616,  -0.20644434, 1.0620342,   -0.8824057,  -0.7676657,  0.7517342,   1.4035656,
        -0.29105335, 0.18367627,  1.3628657,   -0.39770076, -0.1550809,  -1.2575449,  0.5797014,
        -0.02414344, 0.9181723,   -1.1963434,  0.56652546,  -0.25052008, -0.02097719, -2.6274924,
        0.7993208,   -0.31359985, 0.9019325,   -0.02042965, 0.5222995,   1.3394557,   -1.0482218,
        1.1774449,   0.8999488,   -1.1143959,  1.0122099,   -0.48604885, -0.06009902, -0.1766853,
        1.4515465,   -0.7182982,  2.0361354,   0.7899623};
    const ml::Input input = {inputData.data(), inputData.size() * sizeof(float)};
    std::vector<float> outputBuffer(3 * 4 * 5);
    const ml::Output output = {outputBuffer.data(), outputBuffer.size() * sizeof(float)};
    const ml::Result result = utils::AwaitCompute(graph, {{"a", input}}, {{"b", output}}).Get("b");
    EXPECT_TRUE(utils::CheckShape(result, {3, 4, 5}));
    const std::vector<float> expectedData(
        {0.,        0.6447428, 0.,        0.,        0.9777725,  0.,         0.,         0.,
         1.3725083, 0.,        0.,        0.,        0.5027815,  0.,         0.,         0.00880813,
         0.,        1.1644533, 0.,        0.,        0.69386536, 0.06171616, 0.,         1.0620342,
         0.,        0.,        0.7517342, 1.4035656, 0.,         0.18367627, 1.3628657,  0.,
         0.,        0.,        0.5797014, 0.,        0.9181723,  0.,         0.56652546, 0.,
         0.,        0.,        0.7993208, 0.,        0.9019325,  0.,         0.5222995,  1.3394557,
         0.,        1.1774449, 0.8999488, 0.,        1.0122099,  0.,         0.,         0.,
         1.4515465, 0.,        2.0361354, 0.7899623});
    EXPECT_TRUE(utils::CheckValue(result, expectedData));
    EXPECT_TRUE(utils::CheckValue(outputBuffer, expectedData));
}