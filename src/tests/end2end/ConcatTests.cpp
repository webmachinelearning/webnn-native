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

class ConcatTests : public WebnnTest {
  protected:
    typedef struct {
        std::vector<int32_t> shape;
        std::vector<float> value;
    } TensorDescriptor;

    void CheckConcat(const std::vector<TensorDescriptor>& inputs,
                     uint32_t axis,
                     const std::vector<int32_t>& expectedShape,
                     const std::vector<float>& expectedValue,
                     bool inputsDefined = true) {
        const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
        std::vector<ml::Operand> inputsOperand;
        inputsOperand.reserve(inputs.size());
        size_t index = 0;
        std::vector<utils::NamedInput> namedInputs;
        for (auto& input : inputs) {
            std::string inputName = std::to_string(index);
            inputsDefined
                ? inputsOperand.push_back(utils::BuildInput(builder, inputName, input.shape))
                : inputsOperand.push_back(utils::BuildConstant(builder, input.shape,
                                                               input.value.data(),
                                                               input.value.size() * sizeof(float)));
            namedInputs.push_back(
                {inputName, {input.value.data(), input.value.size() * sizeof(float)}});
            ++index;
        }
        const ml::Operand output = builder.Concat(inputsOperand.size(), inputsOperand.data(), axis);
        std::string outputName = std::to_string(inputs.size());
        const ml::Graph graph = utils::AwaitBuild(builder, {{outputName, output}});
        ASSERT_TRUE(graph);
        const ml::Result result = utils::AwaitCompute(graph, namedInputs).Get(outputName.data());
        EXPECT_TRUE(utils::CheckShape(result, expectedShape));
        EXPECT_TRUE(utils::CheckValue(result, expectedValue));
    }
};

TEST_F(ConcatTests, ConcatTwo1DIuputs) {
    const std::vector<TensorDescriptor> inputs = {{{2}, {1, 2}}, {{2}, {3, 4}}};
    const std::vector<int32_t> expectedShape = {4};
    const std::vector<float> expectedValue = {1, 2, 3, 4};
    CheckConcat(inputs, 0, expectedShape, expectedValue);
}

TEST_F(ConcatTests, ConcatTwo2DIuputs) {
    const std::vector<TensorDescriptor> inputs = {{{2, 2}, {1, 2, 3, 4}}, {{2, 2}, {5, 6, 7, 8}}};
    const std::vector<std::vector<int32_t>> expectedShape = {{4, 2}, {2, 4}};
    const std::vector<std::vector<float>> expectedValue = {{1, 2, 3, 4, 5, 6, 7, 8},
                                                           {1, 2, 5, 6, 3, 4, 7, 8}};
    const std::vector<int32_t> axes = {0, 1};
    for (size_t i = 0; i < axes.size(); ++i) {
        CheckConcat(inputs, axes[i], expectedShape[i], expectedValue[i]);
    }
}

TEST_F(ConcatTests, ConcatTwo3DIuputs) {
    const std::vector<TensorDescriptor> inputs = {{{2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}},
                                                  {{2, 2, 2}, {9, 10, 11, 12, 13, 14, 15, 16}}};
    const std::vector<std::vector<int32_t>> expectedShape = {{4, 2, 2}, {2, 4, 2}, {2, 2, 4}};
    const std::vector<std::vector<float>> expectedValue = {
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
        {1, 2, 3, 4, 9, 10, 11, 12, 5, 6, 7, 8, 13, 14, 15, 16},
        {1, 2, 9, 10, 3, 4, 11, 12, 5, 6, 13, 14, 7, 8, 15, 16}};
    const std::vector<int32_t> axes = {0, 1, 2};
    for (size_t i = 0; i < axes.size(); ++i) {
        CheckConcat(inputs, axes[i], expectedShape[i], expectedValue[i]);
    }
}

TEST_F(ConcatTests, ConcatTwo1DConstants) {
    const std::vector<TensorDescriptor> inputs = {{{2}, {1, 2}}, {{2}, {3, 4}}};
    const std::vector<int32_t> expectedShape = {4};
    const std::vector<float> expectedValue = {1, 2, 3, 4};
    CheckConcat(inputs, 0, expectedShape, expectedValue, false);
}

TEST_F(ConcatTests, ConcatTwo2DConstants) {
    const std::vector<TensorDescriptor> inputs = {{{2, 2}, {1, 2, 3, 4}}, {{2, 2}, {5, 6, 7, 8}}};
    const std::vector<std::vector<int32_t>> expectedShape = {{4, 2}, {2, 4}};
    const std::vector<std::vector<float>> expectedValue = {{1, 2, 3, 4, 5, 6, 7, 8},
                                                           {1, 2, 5, 6, 3, 4, 7, 8}};
    const std::vector<int32_t> axes = {0, 1};
    for (size_t i = 0; i < axes.size(); ++i) {
        CheckConcat(inputs, axes[i], expectedShape[i], expectedValue[i], false);
    }
}

TEST_F(ConcatTests, ConcatTwo3DConstants) {
    const std::vector<TensorDescriptor> inputs = {{{2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}},
                                                  {{2, 2, 2}, {9, 10, 11, 12, 13, 14, 15, 16}}};
    const std::vector<std::vector<int32_t>> expectedShape = {{4, 2, 2}, {2, 4, 2}, {2, 2, 4}};
    const std::vector<std::vector<float>> expectedValue = {
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
        {1, 2, 3, 4, 9, 10, 11, 12, 5, 6, 7, 8, 13, 14, 15, 16},
        {1, 2, 9, 10, 3, 4, 11, 12, 5, 6, 13, 14, 7, 8, 15, 16}};
    const std::vector<int32_t> axes = {0, 1, 2};
    for (size_t i = 0; i < axes.size(); ++i) {
        CheckConcat(inputs, axes[i], expectedShape[i], expectedValue[i], false);
    }
}