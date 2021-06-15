// Copyright 2021 The WebNN-native Authors

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

class LeakyReluTests : public WebnnTest {
  public:
    void TestLeakyRelu(const std::vector<int32_t>& inputShape,
                       const std::vector<float>& inputData,
                       const std::vector<float>& expectedValue,
                       float alpha = 0.01) {
        const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
        const ml::Operand a = utils::BuildInput(builder, "a", inputShape);
        ml::LeakyReluOptions options;
        options.alpha = alpha;
        const ml::Operand b = builder.LeakyRelu(a, &options);
        const ml::Graph graph = utils::AwaitBuild(builder, {{"b", b}});
        ASSERT_TRUE(graph);
        const ml::Input input = {inputData.data(), inputData.size() * sizeof(float)};
        const ml::Result result = utils::AwaitCompute(graph, {{"a", input}}).Get("b");

        // Expect output shape is the same as input shape.
        EXPECT_TRUE(utils::CheckShape(result, inputShape));
        EXPECT_TRUE(utils::CheckValue(result, expectedValue));
    }
};

TEST_F(LeakyReluTests, LeakyRelu) {
    const std::vector<std::vector<int32_t>> inputShape = {{3}, {3, 4, 5}};
    const std::vector<std::vector<float>> inputData = {
        {-1, 0, 1},
        {
            0.5945598,   -0.735546,   0.9624621,   0.7178781,   -2.2841945,  1.4461595,
            0.13227068,  -0.05931347, 0.25514695,  0.83969593,  3.4556108,   1.6048287,
            0.30937293,  -0.11302311, -0.55214405, 0.15766327,  0.40505877,  0.7130178,
            -0.53093743, 0.77193236,  -1.6821449,  -0.8352944,  0.08011059,  0.53667474,
            0.11023884,  -0.61316216, 0.53726774,  -0.7437747,  -0.5286507,  1.2811732,
            -0.19160618, -0.5079444,  0.33344734,  1.4179748,   -0.09760198, 1.0317479,
            0.7191149,   0.9713708,   -0.32984316, 0.15518457,  0.16741018,  -0.8231882,
            0.24937603,  -1.1336567,  2.3608718,   1.2201307,   -0.09541762, -0.61066127,
            0.91480494,  0.9309983,   -0.08354045, -0.44542325, 3.088639,    -0.90056187,
            0.25742382,  1.3762826,   0.39736032,  0.49137968,  -0.5622506,  1.1100211,
        }};
    const std::vector<std::vector<float>> expectedValue = {
        {-0.1, 0., 1.},
        {
            0.5945598,   -0.0735546,  0.9624621,   0.7178781,   -0.22841945, 1.4461595,
            0.13227068,  -0.00593135, 0.25514695,  0.83969593,  3.4556108,   1.6048287,
            0.30937293,  -0.01130231, -0.05521441, 0.15766327,  0.40505877,  0.7130178,
            -0.05309374, 0.77193236,  -0.16821449, -0.08352944, 0.08011059,  0.53667474,
            0.11023884,  -0.06131622, 0.53726774,  -0.07437747, -0.05286507, 1.2811732,
            -0.01916062, -0.05079444, 0.33344734,  1.4179748,   -0.0097602,  1.0317479,
            0.7191149,   0.9713708,   -0.03298432, 0.15518457,  0.16741018,  -0.08231882,
            0.24937603,  -0.11336567, 2.3608718,   1.2201307,   -0.00954176, -0.06106613,
            0.91480494,  0.9309983,   -0.00835405, -0.04454232, 3.088639,    -0.09005619,
            0.25742382,  1.3762826,   0.39736032,  0.49137968,  -0.05622506, 1.1100211,
        }};

    for (size_t i = 0; i < inputShape.size(); ++i) {
        TestLeakyRelu(inputShape[i], inputData[i], expectedValue[i], 0.1);
    }
}

TEST_F(LeakyReluTests, LeakyReluDefault) {
    const std::vector<int32_t> inputShape = {3, 4, 5};
    const std::vector<float> inputData = {
        1.2178663,   0.08626969,  -0.25983566, 0.03568677,  -1.5386598,  0.2786136,   0.1057941,
        -0.5374242,  -0.11235637, 0.07136911,  1.1007954,   -0.3993358,  -1.5691061,  0.7312798,
        0.7960611,   0.6767248,   -0.30511293, 0.85154665,  -0.97270423, 0.33083355,  -0.96259284,
        1.0446007,   1.2399997,   -0.4430618,  -0.88743573, -1.1777387,  0.4861841,   1.0564232,
        -0.92164683, -1.7308608,  0.08230155,  -0.7713891,  -0.77213866, -1.0124619,  -1.2846667,
        1.0307417,   0.9004573,   -0.593318,   0.29095086,  -0.50655633, -0.6983193,  0.69927245,
        -1.1014417,  -0.36207023, 1.1648387,   0.0049276,   -0.12467039, 2.7892349,   0.8076212,
        2.2155113,   1.5295383,   -2.2338881,  -1.7535976,  -1.1389159,  -0.16080397, 0.4859151,
        0.34155434,  0.91066486,  0.65148973,  0.13155791,
    };
    const std::vector<float> expectedValue = {
        1.2178663e+00,  8.6269692e-02,  -2.5983565e-03, 3.5686769e-02,  -1.5386598e-02,
        2.7861360e-01,  1.0579410e-01,  -5.3742421e-03, -1.1235637e-03, 7.1369112e-02,
        1.1007954e+00,  -3.9933580e-03, -1.5691061e-02, 7.3127979e-01,  7.9606110e-01,
        6.7672479e-01,  -3.0511292e-03, 8.5154665e-01,  -9.7270422e-03, 3.3083355e-01,
        -9.6259285e-03, 1.0446007e+00,  1.2399997e+00,  -4.4306177e-03, -8.8743567e-03,
        -1.1777386e-02, 4.8618409e-01,  1.0564232e+00,  -9.2164678e-03, -1.7308608e-02,
        8.2301550e-02,  -7.7138911e-03, -7.7213864e-03, -1.0124619e-02, -1.2846666e-02,
        1.0307417e+00,  9.0045732e-01,  -5.9331795e-03, 2.9095086e-01,  -5.0655631e-03,
        -6.9831931e-03, 6.9927245e-01,  -1.1014417e-02, -3.6207023e-03, 1.1648387e+00,
        4.9275993e-03,  -1.2467038e-03, 2.7892349e+00,  8.0762118e-01,  2.2155113e+00,
        1.5295383e+00,  -2.2338880e-02, -1.7535975e-02, -1.1389159e-02, -1.6080397e-03,
        4.8591509e-01,  3.4155434e-01,  9.1066486e-01,  6.5148973e-01,  1.3155791e-01,
    };
    TestLeakyRelu(inputShape, inputData, expectedValue);
}