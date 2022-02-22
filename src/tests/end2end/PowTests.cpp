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

class PowTests : public WebnnTest {};

TEST_F(PowTests, Sqrt1d) {
    const wnn::GraphBuilder builder = wnn::CreateGraphBuilder(GetContext());
    const wnn::Operand a = utils::BuildInput(builder, "a", {3});
    const std::vector<float> bData = {0.5};
    const wnn::Operand b =
        utils::BuildConstant(builder, {1}, bData.data(), bData.size() * sizeof(float));
    const wnn::Operand c = builder.Pow(a, b);
    const wnn::Graph graph = utils::Build(builder, {{"c", c}});
    ASSERT_TRUE(graph);
    const std::vector<float> dataA = {1, 4, 9};
    std::vector<float> result(utils::SizeOfShape({3}));
    utils::Compute(graph, {{"a", dataA}}, {{"c", result}});
    const std::vector<float> expectedValue({1, 2, 3});
    EXPECT_TRUE(utils::CheckValue(result, expectedValue));
}

TEST_F(PowTests, Sqrt3d) {
    const wnn::GraphBuilder builder = wnn::CreateGraphBuilder(GetContext());
    const wnn::Operand a = utils::BuildInput(builder, "a", {3, 4, 5});
    const std::vector<float> bData = {0.5};
    const wnn::Operand b =
        utils::BuildConstant(builder, {1}, bData.data(), bData.size() * sizeof(float));
    const wnn::Operand c = builder.Pow(a, b);
    const wnn::Graph graph = utils::Build(builder, {{"c", c}});
    ASSERT_TRUE(graph);
    const std::vector<float> dataA = {
        0.33435354, 0.57139647, 0.03689031, 0.7820907,  0.7718887,  0.17709309, 1.05624,
        2.2693596,  1.0328789,  1.6043026,  2.0692635,  1.7839943,  1.4888871,  0.57544494,
        0.2760935,  0.25916228, 0.24607088, 0.75507194, 0.9365655,  0.66641825, 0.1919839,
        0.42336762, 1.1776822,  1.8486708,  0.7361624,  0.28052628, 0.261271,   1.0593715,
        0.54762685, 0.61064255, 0.6917134,  0.3692974,  0.01287235, 0.6559981,  0.32968605,
        1.9361054,  1.5982035,  0.49353063, 0.28142217, 0.55740887, 0.43017766, 2.6145968,
        0.4801058,  0.7487864,  1.0473998,  0.11505236, 0.24899477, 0.21978393, 0.21973193,
        0.6550839,  0.7919175,  0.21990986, 0.2881369,  0.5660939,  0.54675615, 0.70638055,
        0.82219034, 0.6266006,  0.89149487, 0.36557788};
    std::vector<float> result(utils::SizeOfShape({3, 4, 5}));
    utils::Compute(graph, {{"a", dataA}}, {{"c", result}});
    const std::vector<float> expectedValue(
        {0.5782331,  0.7559077,  0.1920685,  0.88435894, 0.8785719,  0.4208243,  1.0277354,
         1.5064393,  1.0163065,  1.2666107,  1.4384935,  1.3356625,  1.2201996,  0.75858086,
         0.525446,   0.5090798,  0.4960553,  0.86894876, 0.9677631,  0.81634444, 0.43815967,
         0.6506671,  1.0852107,  1.3596584,  0.8579991,  0.5296473,  0.5111467,  1.0292578,
         0.7400181,  0.7814362,  0.8316931,  0.60769844, 0.11345637, 0.8099371,  0.5741829,
         1.39144,    1.2642008,  0.70251733, 0.53049237, 0.7465982,  0.6558793,  1.6169715,
         0.69289666, 0.86532444, 1.0234255,  0.3391937,  0.49899375, 0.46881118, 0.46875572,
         0.80937254, 0.88989747, 0.46894547, 0.5367839,  0.7523921,  0.7394296,  0.8404645,
         0.9067471,  0.7915811,  0.9441901,  0.60463035});
    EXPECT_TRUE(utils::CheckValue(result, expectedValue));
}

TEST_F(PowTests, Pow1d) {
    const wnn::GraphBuilder builder = wnn::CreateGraphBuilder(GetContext());
    const wnn::Operand a = utils::BuildInput(builder, "a", {3});
    const std::vector<float> bData = {2};
    const wnn::Operand b =
        utils::BuildConstant(builder, {1}, bData.data(), bData.size() * sizeof(float));
    const wnn::Operand c = builder.Pow(a, b);
    const wnn::Graph graph = utils::Build(builder, {{"c", c}});
    ASSERT_TRUE(graph);
    const std::vector<float> dataA = {1, 2, 3};
    std::vector<float> result(utils::SizeOfShape({3}));
    utils::Compute(graph, {{"a", dataA}}, {{"c", result}});
    const std::vector<float> expectedValue({1, 4, 9});
    EXPECT_TRUE(utils::CheckValue(result, expectedValue));
}

TEST_F(PowTests, PowBroadcastScalar) {
    const wnn::GraphBuilder builder = wnn::CreateGraphBuilder(GetContext());
    const wnn::Operand a = utils::BuildInput(builder, "a", {2, 3});
    const std::vector<float> bData = {2};
    const wnn::Operand b =
        utils::BuildConstant(builder, {1}, bData.data(), bData.size() * sizeof(float));
    const wnn::Operand c = builder.Pow(a, b);
    const wnn::Graph graph = utils::Build(builder, {{"c", c}});
    ASSERT_TRUE(graph);
    const std::vector<float> dataA = {1, 2, 3, 4, 5, 6};
    std::vector<float> result(utils::SizeOfShape({2, 3}));
    utils::Compute(graph, {{"a", dataA}}, {{"c", result}});
    const std::vector<float> expectedValue({1, 4, 9, 16, 25, 36});
    EXPECT_TRUE(utils::CheckValue(result, expectedValue));
}

TEST_F(PowTests, PowBroadcast1d) {
    const wnn::GraphBuilder builder = wnn::CreateGraphBuilder(GetContext());
    const wnn::Operand a = utils::BuildInput(builder, "a", {2, 3});
    const std::vector<float> bData = {1, 2, 3};
    const wnn::Operand b =
        utils::BuildConstant(builder, {3}, bData.data(), bData.size() * sizeof(float));
    const wnn::Operand c = builder.Pow(a, b);
    const wnn::Graph graph = utils::Build(builder, {{"c", c}});
    ASSERT_TRUE(graph);
    const std::vector<float> dataA = {1, 2, 3, 4, 5, 6};
    std::vector<float> result(utils::SizeOfShape({2, 3}));
    utils::Compute(graph, {{"a", dataA}}, {{"c", result}});
    const std::vector<float> expectedValue({1, 4, 27, 4, 25, 216});
    EXPECT_TRUE(utils::CheckValue(result, expectedValue));
}