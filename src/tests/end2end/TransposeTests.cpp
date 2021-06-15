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

class TransposeTests : public WebnnTest {
  protected:
    void CheckTranspose(const std::vector<int32_t>& inputShape,
                        const std::vector<float>& inputData,
                        const std::vector<int32_t>& expectedShape,
                        const std::vector<float>& expectedValue,
                        const std::vector<int32_t>& permutation = {}) {
        const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
        const ml::Operand a = utils::BuildInput(builder, "a", inputShape);
        ml::TransposeOptions options;
        options.permutation = permutation.data();
        options.permutationCount = permutation.size();
        const ml::Operand b = builder.Transpose(a, &options);
        const ml::Graph graph = utils::AwaitBuild(builder, {{"b", b}});
        ASSERT_TRUE(graph);
        const ml::Input input = {inputData.data(), inputData.size() * sizeof(float)};
        const ml::Result result = utils::AwaitCompute(graph, {{"a", input}}).Get("b");
        EXPECT_TRUE(utils::CheckShape(result, expectedShape));
        EXPECT_TRUE(utils::CheckValue(result, expectedValue));
    }
};

TEST_F(TransposeTests, TransposeDefault) {
    const std::vector<int32_t> inputShape = {2, 3, 4};
    const std::vector<float> inputData = {
        0.43376675, 0.264609,   0.26321858, 0.04260185, 0.6862414,  0.26150206,
        0.04169406, 0.24857993, 0.14914423, 0.19905873, 0.33851373, 0.74131566,
        0.91501445, 0.21852633, 0.02267954, 0.22069663, 0.95799077, 0.17188412,
        0.09732241, 0.03296741, 0.04709655, 0.50648814, 0.13075736, 0.82511896,
    };
    const std::vector<int32_t> expectedShape = {4, 3, 2};
    const std::vector<float> expectedValue = {
        0.43376675, 0.91501445, 0.6862414,  0.95799077, 0.14914423, 0.04709655,
        0.264609,   0.21852633, 0.26150206, 0.17188412, 0.19905873, 0.50648814,
        0.26321858, 0.02267954, 0.04169406, 0.09732241, 0.33851373, 0.13075736,
        0.04260185, 0.22069663, 0.24857993, 0.03296741, 0.74131566, 0.82511896,
    };
    CheckTranspose(inputShape, inputData, expectedShape, expectedValue);
}

TEST_F(TransposeTests, TransposePermutations) {
    const std::vector<int32_t> inputShape = {2, 3, 4};
    const std::vector<float> inputData = {
        0.7760998,  0.8363521,  0.10145967, 0.00533229, 0.8190919,  0.83241564,
        0.39479077, 0.5622921,  0.9306249,  0.00480607, 0.39600816, 0.35415828,
        0.43689877, 0.7603583,  0.14368972, 0.11940759, 0.4834097,  0.6982117,
        0.7195266,  0.72893023, 0.896649,   0.13060148, 0.07824122, 0.33766487,
    };
    const std::vector<std::vector<int32_t>> permutations = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2},
                                                            {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
    const std::vector<std::vector<int32_t>> expectedShapes = {{2, 3, 4}, {2, 4, 3}, {3, 2, 4},
                                                              {3, 4, 2}, {4, 2, 3}, {4, 3, 2}};
    const std::vector<std::vector<float>> expectedValues = {
        {
            0.7760998,  0.8363521,  0.10145967, 0.00533229, 0.8190919,  0.83241564,
            0.39479077, 0.5622921,  0.9306249,  0.00480607, 0.39600816, 0.35415828,
            0.43689877, 0.7603583,  0.14368972, 0.11940759, 0.4834097,  0.6982117,
            0.7195266,  0.72893023, 0.896649,   0.13060148, 0.07824122, 0.33766487,
        },
        {
            0.7760998,  0.8190919,  0.9306249,  0.8363521,  0.83241564, 0.00480607,
            0.10145967, 0.39479077, 0.39600816, 0.00533229, 0.5622921,  0.35415828,
            0.43689877, 0.4834097,  0.896649,   0.7603583,  0.6982117,  0.13060148,
            0.14368972, 0.7195266,  0.07824122, 0.11940759, 0.72893023, 0.33766487,
        },
        {
            0.7760998,  0.8363521,  0.10145967, 0.00533229, 0.43689877, 0.7603583,
            0.14368972, 0.11940759, 0.8190919,  0.83241564, 0.39479077, 0.5622921,
            0.4834097,  0.6982117,  0.7195266,  0.72893023, 0.9306249,  0.00480607,
            0.39600816, 0.35415828, 0.896649,   0.13060148, 0.07824122, 0.33766487,
        },
        {
            0.7760998,  0.43689877, 0.8363521,  0.7603583,  0.10145967, 0.14368972,
            0.00533229, 0.11940759, 0.8190919,  0.4834097,  0.83241564, 0.6982117,
            0.39479077, 0.7195266,  0.5622921,  0.72893023, 0.9306249,  0.896649,
            0.00480607, 0.13060148, 0.39600816, 0.07824122, 0.35415828, 0.33766487,
        },
        {
            0.7760998,  0.8190919,  0.9306249,  0.43689877, 0.4834097,  0.896649,
            0.8363521,  0.83241564, 0.00480607, 0.7603583,  0.6982117,  0.13060148,
            0.10145967, 0.39479077, 0.39600816, 0.14368972, 0.7195266,  0.07824122,
            0.00533229, 0.5622921,  0.35415828, 0.11940759, 0.72893023, 0.33766487,
        },
        {
            0.7760998,  0.43689877, 0.8190919,  0.4834097,  0.9306249,  0.896649,
            0.8363521,  0.7603583,  0.83241564, 0.6982117,  0.00480607, 0.13060148,
            0.10145967, 0.14368972, 0.39479077, 0.7195266,  0.39600816, 0.07824122,
            0.00533229, 0.11940759, 0.5622921,  0.72893023, 0.35415828, 0.33766487,
        },
    };
    for (size_t i = 0; i < permutations.size(); ++i) {
        CheckTranspose(inputShape, inputData, expectedShapes[i], expectedValues[i],
                       permutations[i]);
    }
}
