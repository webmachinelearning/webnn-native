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

  public:
    void TestGru(const std::vector<int32_t>& inputShape,
                 const std::vector<float>& inputData,
                 const std::vector<int32_t>& weightShape,
                 const std::vector<float>& weightData,
                 const std::vector<int32_t>& recurrentWeightShape,
                 const std::vector<float>& recurrentWeightData,
                 const int32_t steps,
                 const int32_t hiddenSize,
                 const std::vector<std::vector<int32_t>>& expectedShape,
                 const std::vector<float>& expectedValue,
                 const ml::GruOptions* options = nullptr) {
        const ml::Operand weight = utils::BuildConstant(builder, weightShape, weightData.data(),
                                                        weightData.size() * sizeof(float));
        const ml::Operand recurrentWeight =
            utils::BuildConstant(builder, recurrentWeightShape, recurrentWeightData.data(),
                                 recurrentWeightData.size() * sizeof(float));
        const ml::Operand a = utils::BuildInput(builder, "a", inputShape);
        const ml::OperandArray b = builder.Gru(a, weight, recurrentWeight, steps, hiddenSize, options);
        const size_t outputSize = b.Size();
        std::vector<utils::NamedOperand> namedOperands;
        for (size_t i = 0; i < outputSize; ++i) {
            namedOperands.push_back({"gru" + std::to_string(i), b.Get(i)});
        }
        const ml::Graph graph = utils::Build(builder, namedOperands);
        ASSERT_TRUE(graph);

        std::vector<utils::NamedOutput<float>> namedOutputs;
        std::vector<std::vector<float>> results;
        results.reserve(outputSize);
        for (size_t i = 0; i < outputSize; ++i) {
            results.push_back(std::vector<float>(utils::SizeOfShape(expectedShape[i])));
            namedOutputs.push_back({"gru" + std::to_string(i), results.back()});
        }
        utils::Compute(graph, {{"a", inputData}}, namedOutputs);

        EXPECT_TRUE(utils::CheckValue(namedOutputs[1].resource, expectedValue));
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
    const std::vector<int32_t> weightShape = {numDirections, 3 * hiddenSize, inputSize};
    const std::vector<float> weightData(numDirections * 3 * hiddenSize * inputSize, 0.1);
    const std::vector<int32_t> recurrentWeightShape = {numDirections, 3 * hiddenSize, hiddenSize};
    const std::vector<float> recurrentWeightData(numDirections * 3 * hiddenSize * hiddenSize, 0.1);
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

    const ml::RecurrentNetworkDirection direction = ml::RecurrentNetworkDirection::Forward;

    ml::GruOptions options = {};
    options.bias = bias;
    options.recurrentBias = recurrentBias;
    options.initialHiddenState = initialHiddenState;
    options.resetAfter = false;
    options.returnSequence = true;
    options.direction = direction;

    const std::vector<int32_t> expectedShape0 = {numDirections, batchSize, hiddenSize};
    const std::vector<int32_t> expectedShape1 = {steps, numDirections, batchSize, hiddenSize};
    auto expectedShape = {expectedShape0, expectedShape0};
    const std::vector<float> expectedValue = {
        0.22391089, 0.22391089, 0.22391089, 0.22391089, 0.22391089, 0.1653014, 0.1653014, 0.1653014,
        0.1653014,  0.1653014,  0.0797327,  0.0797327,  0.0797327,  0.0797327, 0.0797327};

    TestGru(inputShape, inputData, weightShape, weightData, recurrentWeightShape,
            recurrentWeightData, steps, hiddenSize, expectedShape, expectedValue, &options);
}
