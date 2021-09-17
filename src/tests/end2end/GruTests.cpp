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

class GruTests : public WebnnTest {
    void SetUp() override {
        builder = ml::CreateGraphBuilder(GetContext());
    }

  protected:
    struct Tensor {
        std::vector<int32_t> shape;
        std::vector<float> value;
    };

  public:
    void TestGru(const Tensor& input,
                 const Tensor& weight,
                 const Tensor& recurrentWeight,
                 const int32_t steps,
                 const int32_t hiddenSize,
                 const Tensor& expected,
                 const ml::GruOptions* options = nullptr) {
        const ml::Operand W = utils::BuildConstant(builder, weight.shape, weight.value.data(),
                                                   weight.value.size() * sizeof(float));
        const ml::Operand R =
            utils::BuildConstant(builder, recurrentWeight.shape, recurrentWeight.value.data(),
                                 recurrentWeight.value.size() * sizeof(float));
        const ml::Operand X = utils::BuildInput(builder, "a", input.shape);
        const ml::OperandArray Y = builder.Gru(X, W, R, steps, hiddenSize, options);
        const size_t outputSize = Y.Size();
        std::vector<utils::NamedOperand> namedOperands;
        for (size_t i = 0; i < outputSize; ++i) {
            namedOperands.push_back({"gru" + std::to_string(i), Y.Get(i)});
        }
        const ml::Graph graph = utils::Build(builder, namedOperands);
        ASSERT_TRUE(graph);

        std::vector<utils::NamedOutput<float>> namedOutputs;
        std::vector<std::vector<float>> results;
        results.reserve(outputSize);
        for (size_t i = 0; i < outputSize; ++i) {
            results.push_back(std::vector<float>(utils::SizeOfShape(expected.shape)));
            namedOutputs.push_back({"gru" + std::to_string(i), results.back()});
        }
        utils::Compute(graph, {{"a", input.value}}, namedOutputs);

        EXPECT_TRUE(utils::CheckValue(namedOutputs[0].resource, expected.value));
    }
    ml::GraphBuilder builder;
};

TEST_F(GruTests, GruWith3BatchSize) {
    const int32_t steps = 2;
    const int32_t batchSize = 3;
    const int32_t inputSize = 3;
    const int32_t hiddenSize = 5;
    const int32_t numDirections = 1;

    const std::vector<int32_t> inputShape = {steps, batchSize, inputSize};
    const std::vector<float> inputData = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                                          10, 11, 12, 13, 14, 15, 16, 17, 18};
    Tensor input = {inputShape, inputData};
    const std::vector<int32_t> weightShape = {numDirections, 3 * hiddenSize, inputSize};
    const std::vector<float> weightData(numDirections * 3 * hiddenSize * inputSize, 0.1);
    Tensor weight = {weightShape, weightData};
    const std::vector<int32_t> recurrentWeightShape = {numDirections, 3 * hiddenSize, hiddenSize};
    const std::vector<float> recurrentWeightData(numDirections * 3 * hiddenSize * hiddenSize, 0.1);
    Tensor recurrentWeight = {recurrentWeightShape, recurrentWeightData};
    const std::vector<int32_t> biasShape = {numDirections, 3 * hiddenSize};
    const std::vector<float> biasData(numDirections * 3 * hiddenSize, 0.1);
    const ml::Operand bias =
        utils::BuildConstant(builder, biasShape, biasData.data(), biasData.size() * sizeof(float));
    const std::vector<int32_t> recurrentBiasShape = {numDirections, 3 * hiddenSize};
    const std::vector<float> recurrentBiasData(numDirections * 3 * hiddenSize, 0);
    const ml::Operand recurrentBias =
        utils::BuildConstant(builder, recurrentBiasShape, recurrentBiasData.data(),
                             recurrentBiasData.size() * sizeof(float));
    const std::vector<int32_t> initialHiddenStateShape = {numDirections, batchSize, hiddenSize};
    const std::vector<float> initialHiddenStateData(numDirections * batchSize * hiddenSize, 0);
    const ml::Operand initialHiddenState =
        utils::BuildConstant(builder, initialHiddenStateShape, initialHiddenStateData.data(),
                             initialHiddenStateData.size() * sizeof(float));

    ml::GruOptions options = {};
    options.bias = bias;
    options.recurrentBias = recurrentBias;
    options.initialHiddenState = initialHiddenState;
    options.resetAfter = false;

    const std::vector<int32_t> expectedShape = {numDirections, batchSize, hiddenSize};
    const std::vector<float> expectedValue = {
        0.22391089, 0.22391089, 0.22391089, 0.22391089, 0.22391089, 0.1653014, 0.1653014, 0.1653014,
        0.1653014,  0.1653014,  0.0797327,  0.0797327,  0.0797327,  0.0797327, 0.0797327};
    Tensor expected = {expectedShape, expectedValue};

    TestGru(input, weight, recurrentWeight, steps, hiddenSize, expected, &options);
}

TEST_F(GruTests, GruWithoutInitialHiddenState) {
    const int32_t steps = 2;
    const int32_t batchSize = 3;
    const int32_t inputSize = 3;
    const int32_t hiddenSize = 5;
    const int32_t numDirections = 1;

    const std::vector<int32_t> inputShape = {steps, batchSize, inputSize};
    const std::vector<float> inputData = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                                          10, 11, 12, 13, 14, 15, 16, 17, 18};
    Tensor input = {inputShape, inputData};
    const std::vector<int32_t> weightShape = {numDirections, 3 * hiddenSize, inputSize};
    const std::vector<float> weightData(numDirections * 3 * hiddenSize * inputSize, 0.1);
    Tensor weight = {weightShape, weightData};
    const std::vector<int32_t> recurrentWeightShape = {numDirections, 3 * hiddenSize, hiddenSize};
    const std::vector<float> recurrentWeightData(numDirections * 3 * hiddenSize * hiddenSize, 0.1);
    Tensor recurrentWeight = {recurrentWeightShape, recurrentWeightData};
    const std::vector<int32_t> biasShape = {numDirections, 3 * hiddenSize};
    const std::vector<float> biasData(numDirections * 3 * hiddenSize, 0.1);
    const ml::Operand bias =
        utils::BuildConstant(builder, biasShape, biasData.data(), biasData.size() * sizeof(float));
    const std::vector<int32_t> recurrentBiasShape = {numDirections, 3 * hiddenSize};
    const std::vector<float> recurrentBiasData(numDirections * 3 * hiddenSize, 0);
    const ml::Operand recurrentBias =
        utils::BuildConstant(builder, recurrentBiasShape, recurrentBiasData.data(),
                             recurrentBiasData.size() * sizeof(float));

    ml::GruOptions options = {};
    options.bias = bias;
    options.recurrentBias = recurrentBias;
    options.resetAfter = false;

    const std::vector<int32_t> expectedShape = {numDirections, batchSize, hiddenSize};
    const std::vector<float> expectedValue = {
        0.22391089, 0.22391089, 0.22391089, 0.22391089, 0.22391089, 0.1653014, 0.1653014, 0.1653014,
        0.1653014,  0.1653014,  0.0797327,  0.0797327,  0.0797327,  0.0797327, 0.0797327};
    Tensor expected = {expectedShape, expectedValue};

    TestGru(input, weight, recurrentWeight, steps, hiddenSize, expected, &options);
}
