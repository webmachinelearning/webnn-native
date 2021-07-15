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

class SigmoidTests : public WebnnTest {};

TEST_F(SigmoidTests, SigmoidWith1DTensor) {
    const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
    const ml::Operand a = utils::BuildInput(builder, "a", {3});
    const ml::Operand b = builder.Sigmoid(a);
    const ml::Graph graph = utils::AwaitBuild(builder, {{"b", b}});
    ASSERT_TRUE(graph);
    const std::vector<float> inputData = {-1, 0, 1};
    const ml::Input input = {inputData.data(), inputData.size() * sizeof(float)};
    const ml::Result result = utils::AwaitCompute(graph, {{"a", input}}).Get("b");
    EXPECT_TRUE(utils::CheckShape(result, {3}));
    const std::vector<float> expectedData = {0.26894143, 0.5, 0.7310586};
    EXPECT_TRUE(utils::CheckValue(result, expectedData));
}

TEST_F(SigmoidTests, SigmoidWith3DTensor) {
    const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
    const ml::Operand a = utils::BuildInput(builder, "a", {3, 4, 5});
    const ml::Operand b = builder.Sigmoid(a);
    const ml::Graph graph = utils::AwaitBuild(builder, {{"b", b}});
    ASSERT_TRUE(graph);
    const std::vector<float> inputData = {
        -0.18371736, 0.4805392,   2.7183356,   0.03039639,  0.04197176,
        -1.1536852,  -2.0124357,  -0.885673,   -0.25776535, 1.0151213,
        -0.22013742, 0.13626824,  0.8574488,   -0.15987602, 0.7025059,
        -0.8209337,  1.2621661,   0.4055987,   -0.65470445, 0.14290208,
        1.6874043,   -0.7997532,  -1.0582826,  1.0813274,   -1.9656292,
        -0.13285251, 0.87344545,  -0.07760263, 1.0503976,   -0.23713546,
        0.21536243,  0.59599924,  -0.8221842,  0.10256762,  -0.67856175,
        1.1891315,   -0.6567207,  -0.2958169,  -1.9581499,  -0.9223802,
        -0.32011083, -0.31802705, 0.7264381,   1.0234208,   0.673269,
        0.96394795,  0.6152301,   -0.4362364,  -1.2325221,  -0.11140272,
        -0.43866253, 0.5770897,   0.42372307,  -0.33066413, -0.46210232,
        -0.6456375,  2.0984166,   -1.2020895,  1.5637838,   -0.7114222,};
    const ml::Input input = {inputData.data(), inputData.size() * sizeof(float)};
    const ml::Result result = utils::AwaitCompute(graph, {{"a", input}}).Get("b");
    EXPECT_TRUE(utils::CheckShape(result, {3, 4, 5}));
    const std::vector<float> expectedData = {
        0.4541994,  0.61787516, 0.9381,     0.50759846, 0.5104914,
        0.23981662, 0.11790343, 0.29200357, 0.43591312, 0.7340212,
        0.44518682, 0.53401446, 0.7021274,  0.4601159,  0.66874313,
        0.3055655,  0.77939874, 0.6000321,  0.34193018, 0.53566486,
        0.8438825,  0.31007832, 0.2576378,  0.7467451,  0.12285913,
        0.46683565, 0.70546216, 0.48060906, 0.7408512,  0.44099236,
        0.55363345, 0.64474046, 0.3053002,  0.52561945, 0.33658236,
        0.7665857,  0.34147665, 0.4265804,  0.12366741, 0.28447315,
        0.42064875, 0.42115664, 0.67402315, 0.7356384,  0.6622347,
        0.7239115,  0.64913297, 0.39263815, 0.2257403,  0.47217807,
        0.39205968, 0.6403975,  0.6043738,  0.41807905, 0.38648725,
        0.34397328, 0.89074916, 0.2311037,  0.8268956,  0.32928467,
    };
    EXPECT_TRUE(utils::CheckValue(result, expectedData));
}
