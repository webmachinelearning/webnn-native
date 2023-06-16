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

#include "webnn/tests/WebnnTest.h"

class InstanceNormTests : public WebnnTest {
    void SetUp() override {
        builder = wnn::CreateGraphBuilder(GetContext());
    }

  protected:
    void CheckInstanceNorm(const std::vector<int32_t>& inputShape,
                           const std::vector<float>& inputData,
                           const std::vector<float>& expectedValue,
                           const wnn::InstanceNormOptions* options = nullptr) {
        const wnn::Operand a = utils::BuildInput(builder, "a", inputShape);
        const wnn::Operand b = builder.InstanceNorm(a, options);
        const wnn::Graph graph = utils::Build(builder, {{"b", b}});
        ASSERT_TRUE(graph);
        std::vector<float> result(utils::SizeOfShape(inputShape));
        utils::Compute(GetContext(), graph, {{"a", inputData}}, {{"b", result}});
        EXPECT_TRUE(utils::CheckValue(result, expectedValue));
    }
    wnn::GraphBuilder builder;
};

TEST_F(InstanceNormTests, InstanceNormNchw) {
    const std::vector<int32_t> inputShape = {1, 2, 1, 3};
    const std::vector<float> inputData = {-1, 0, 1, 2, 3, 4};
    const std::vector<float> scaleData = {1.0, 1.5};
    const wnn::Operand scale =
        utils::BuildConstant(builder, {2}, scaleData.data(), scaleData.size() * sizeof(float));
    const std::vector<float> biasData = {0, 1};
    const wnn::Operand bias =
        utils::BuildConstant(builder, {2}, biasData.data(), biasData.size() * sizeof(float));
    wnn::InstanceNormOptions options;
    options.scale = scale;
    options.bias = bias;
    std::vector<float> expectedValue = {-1.2247356, 0., 1.2247356, -0.8371035, 1., 2.8371034};
    CheckInstanceNorm(inputShape, inputData, expectedValue, &options);

    options = {};
    options.scale = scale;
    expectedValue = {-1.2247356, 0., 1.2247356, -1.8371035, 0., 1.8371034};
    CheckInstanceNorm(inputShape, inputData, expectedValue, &options);

    options = {};
    options.bias = bias;
    expectedValue = {-1.2247356, 0., 1.2247356, -0.2247356, 1., 2.2247356};
    CheckInstanceNorm(inputShape, inputData, expectedValue, &options);

    options = {};
    expectedValue = {-1.2247356, 0., 1.2247356, -1.2247356, 0., 1.2247356};
    CheckInstanceNorm(inputShape, inputData, expectedValue, &options);
}

TEST_F(InstanceNormTests, InstanceNormNhwc) {
    const std::vector<int32_t> inputShape = {1, 1, 3, 2};
    const std::vector<float> inputData = {-1, 2, 0, 3, 1, 4};
    const std::vector<float> scaleData = {1.0, 1.5};
    const wnn::Operand scale =
        utils::BuildConstant(builder, {2}, scaleData.data(), scaleData.size() * sizeof(float));
    const std::vector<float> biasData = {0, 1};
    const wnn::Operand bias =
        utils::BuildConstant(builder, {2}, biasData.data(), biasData.size() * sizeof(float));
    wnn::InstanceNormOptions options;
    options.scale = scale;
    options.bias = bias;
    options.layout = wnn::InputOperandLayout::Nhwc;
    std::vector<float> expectedValue = {-1.2247356, -0.8371035, 0., 1., 1.2247356, 2.8371034};
    CheckInstanceNorm(inputShape, inputData, expectedValue, &options);

    options = {};
    options.scale = scale;
    options.layout = wnn::InputOperandLayout::Nhwc;
    expectedValue = {-1.2247356, -1.8371035, 0, 0, 1.2247356, 1.8371034};
    CheckInstanceNorm(inputShape, inputData, expectedValue, &options);

    options = {};
    options.bias = bias;
    options.layout = wnn::InputOperandLayout::Nhwc;
    expectedValue = {-1.2247356, -0.2247356, 0, 1., 1.2247356, 2.2247356};
    CheckInstanceNorm(inputShape, inputData, expectedValue, &options);

    options = {};
    options.layout = wnn::InputOperandLayout::Nhwc;
    expectedValue = {-1.2247356, -1.2247356, 0, 0., 1.2247356, 1.2247356};
    CheckInstanceNorm(inputShape, inputData, expectedValue, &options);
}

TEST_F(InstanceNormTests, InstanceNormWithEpsilon) {
    const std::vector<int32_t> inputShape = {2, 3, 4, 5};
    const std::vector<float> inputData = {
        0.23991525,  -1.3108366,  -0.8056796,  -0.20892623, 0.4869082,   0.68151075,  0.5888935,
        0.45948547,  -0.88501406, 1.9609013,   0.21517155,  0.7427738,   0.853326,    -1.4071878,
        -0.29106674, -0.11532243, -0.6716852,  -0.30156505, 0.45744178,  0.25426418,  -2.0593688,
        -0.6538451,  -0.2512263,  0.406628,    0.60806894,  0.44109958,  0.92832196,  -0.64059025,
        -1.123297,   0.39360833,  0.25650138,  -1.0688477,  -1.746481,   -1.4138917,  -0.75429744,
        -0.08158732, 0.38385203,  -0.14610007, 1.0802624,   -0.6336054,  -0.21925186, -0.28690416,
        2.0004241,   -0.4133636,  -0.68942326, -0.9513434,  -0.14104685, -0.74617624, 1.1912495,
        -1.8508383,  1.4370863,   1.4129546,   2.4901733,   1.1550623,   -0.6818049,  0.21176137,
        -0.44371676, -0.26306337, -0.01728541, 0.31616235,  1.4783081,   -0.49589762, 1.5187141,
        -0.63279223, -0.7817453,  0.5844729,   0.35824686,  0.95782155,  -0.5790926,  -0.11348185,
        0.07356142,  -0.46691516, 0.00941111,  -0.23862253, -0.38974136, 1.1558224,   -0.51724064,
        0.27272934,  -0.798853,   0.29123458,  2.239732,    0.11339404,  1.0436004,   -0.38251045,
        0.5698435,   -1.3686458,  -0.04936051, -0.6490325,  -0.58417344, -0.03446375, -0.7549325,
        0.3405552,   0.3986468,   0.69191414,  -2.348451,   1.2103504,   -0.33432662, 0.33831024,
        0.11177547,  -1.9477838,  1.9487013,   0.13771565,  0.32841948,  -0.6585632,  -0.19066776,
        -1.1357359,  1.3015537,   -0.50085896, -0.14418271, -0.2672549,  1.2022284,   2.2585037,
        1.3386828,   0.4864522,   -0.17040975, -0.5450272,  -0.0906434,  -0.23216948, -1.6074374,
        -0.33925202,
    };
    const std::vector<float> scaleData = {0.55290383, -1.1786512, -0.12353817};
    const wnn::Operand scale =
        utils::BuildConstant(builder, {3}, scaleData.data(), scaleData.size() * sizeof(float));
    const std::vector<float> biasData = {0.36079535, 2.3073995, -0.12267359};
    const wnn::Operand bias =
        utils::BuildConstant(builder, {3}, biasData.data(), biasData.size() * sizeof(float));

    wnn::InstanceNormOptions options;
    options.scale = scale;
    options.bias = bias;
    options.epsilon = 1e-2;
    std::vector<float> expectedValue = {
        4.94363964e-01,  -5.80250263e-01, -2.30195075e-01, 1.83333233e-01,  6.65521026e-01,
        8.00373435e-01,  7.36193061e-01,  6.46518111e-01,  -2.85170883e-01, 1.68694437e+00,
        4.77217466e-01,  8.42826486e-01,  9.19435143e-01,  -6.47018194e-01, 1.26412854e-01,
        2.48197228e-01,  -1.37341797e-01, 1.19137913e-01,  6.45101905e-01,  5.04307210e-01,
        4.69082165e+00,  2.78269839e+00,  2.23610783e+00,  1.34301233e+00,  1.06953847e+00,
        1.29621410e+00,  6.34766579e-01,  2.76470399e+00,  3.42002106e+00,  1.36068761e+00,
        1.54682255e+00,  3.34610128e+00,  4.26604843e+00,  3.81452894e+00,  2.91907144e+00,
        2.00580788e+00,  1.37393284e+00,  2.09338975e+00,  4.28493977e-01,  2.75522137e+00,
        -7.73530304e-02, -6.95866644e-02, -3.32167834e-01, -5.50693423e-02, -2.33782008e-02,
        6.68977946e-03,  -8.63308161e-02, -1.68630555e-02, -2.39276052e-01, 1.09950148e-01,
        -2.67497659e-01, -2.64727384e-01, -3.88390154e-01, -2.35121816e-01, -2.42527649e-02,
        -1.26832575e-01, -5.15848622e-02, -7.23235458e-02, -1.00538410e-01, -1.38817608e-01,
        1.43087399e+00,  -8.45769346e-02, 1.46189058e+00,  -1.89660758e-01, -3.04000944e-01,
        7.44743168e-01,  5.71086287e-01,  1.03133523e+00,  -1.48439497e-01, 2.08975226e-01,
        3.52554440e-01,  -6.23292625e-02, 3.03311020e-01,  1.12914041e-01,  -3.08865309e-03,
        1.18332565e+00,  -1.00960344e-01, 5.05440831e-01,  -3.17133218e-01, 5.19645929e-01,
        -3.07856321e-01, 2.09997821e+00,  1.04662585e+00,  2.66153336e+00,  1.58310151e+00,
        3.77821898e+00,  2.28427911e+00,  2.96333909e+00,  2.88989353e+00,  2.26741028e+00,
        3.08325863e+00,  1.84274423e+00,  1.77696204e+00,  1.44487035e+00,  4.88773632e+00,
        8.57800603e-01,  2.60697079e+00,  1.84528637e+00,  2.10181117e+00,  4.43402672e+00,
        -3.49317908e-01, -1.20361626e-01, -1.44471616e-01, -1.96910128e-02, -7.88453147e-02,
        4.06361893e-02,  -2.67501414e-01, -3.96289825e-02, -8.47222507e-02, -6.91626817e-02,
        -2.54944086e-01, -3.88485104e-01, -2.72195488e-01, -1.64451107e-01, -8.14064592e-02,
        -3.40449512e-02, -9.14910287e-02, -7.35983998e-02, 1.00271679e-01,  -6.00603521e-02,
    };
    CheckInstanceNorm(inputShape, inputData, expectedValue, &options);
}
