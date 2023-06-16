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

#include "webnn/tests/WebnnTest.h"

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
        const wnn::GraphBuilder builder = wnn::CreateGraphBuilder(GetContext());
        std::vector<wnn::Operand> inputsOperand;
        inputsOperand.reserve(inputs.size());
        size_t index = 0;
        std::vector<utils::NamedInput<float>> namedInputs;
        for (auto& input : inputs) {
            std::string inputName = std::to_string(index);
            inputsDefined
                ? inputsOperand.push_back(utils::BuildInput(builder, inputName, input.shape))
                : inputsOperand.push_back(utils::BuildConstant(builder, input.shape,
                                                               input.value.data(),
                                                               input.value.size() * sizeof(float)));
            namedInputs.push_back({inputName, input.value});
            ++index;
        }
        const wnn::Operand output =
            builder.Concat(inputsOperand.size(), inputsOperand.data(), axis);
        std::string outputName = std::to_string(inputs.size());
        const wnn::Graph graph = utils::Build(builder, {{outputName, output}});
        ASSERT_TRUE(graph);
        std::vector<float> result(utils::SizeOfShape(expectedShape));
        utils::Compute(GetContext(), graph, namedInputs, {{outputName, result}});
        EXPECT_TRUE(utils::CheckValue(result, expectedValue));
    }
};

TEST_F(ConcatTests, ConcatTwo1DIuputs) {
    const std::vector<TensorDescriptor> inputs = {{{2}, {1, 2}}, {{2}, {3, 4}}};
    const std::vector<int32_t> expectedShape = {4};
    const std::vector<float> expectedValue = {1, 2, 3, 4};
    CheckConcat(inputs, 0, expectedShape, expectedValue);
}

TEST_F(ConcatTests, ConcatThree1DIuputs) {
    const std::vector<TensorDescriptor> inputs = {{{2}, {1, 2}}, {{2}, {3, 4}}, {{2}, {5, 6}}};
    const std::vector<int32_t> expectedShape = {6};
    const std::vector<float> expectedValue = {1, 2, 3, 4, 5, 6};
    CheckConcat(inputs, 0, expectedShape, expectedValue);
}

TEST_F(ConcatTests, ConcatFour1DIuputs) {
    const std::vector<TensorDescriptor> inputs = {
        {{2}, {1, 2}}, {{2}, {3, 4}}, {{2}, {5, 6}}, {{2}, {7, 8}}};
    const std::vector<int32_t> expectedShape = {8};
    const std::vector<float> expectedValue = {1, 2, 3, 4, 5, 6, 7, 8};
    CheckConcat(inputs, 0, expectedShape, expectedValue);
}

TEST_F(ConcatTests, ConcatFive1DIuputs) {
    const std::vector<TensorDescriptor> inputs = {
        {{2}, {1, 2}}, {{2}, {3, 4}}, {{2}, {5, 6}}, {{2}, {7, 8}}, {{2}, {9, 10}}};
    const std::vector<int32_t> expectedShape = {10};
    const std::vector<float> expectedValue = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
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

TEST_F(ConcatTests, ConcatTwo2DIuputsWithAxis0) {
    const std::vector<TensorDescriptor> inputs = {{{1, 2}, {1, 2}}, {{2, 2}, {3, 4, 5, 6}}};
    const std::vector<int32_t> expectedShape = {3, 2};
    const std::vector<float> expectedValue = {1, 2, 3, 4, 5, 6};
    CheckConcat(inputs, 0, expectedShape, expectedValue);
}

TEST_F(ConcatTests, ConcatThree2DIuputsWithAxis0) {
    const std::vector<TensorDescriptor> inputs = {
        {{1, 2}, {1, 2}}, {{2, 2}, {3, 4, 5, 6}}, {{3, 2}, {7, 8, 9, 10, 11, 12}}};
    const std::vector<int32_t> expectedShape = {6, 2};
    const std::vector<float> expectedValue = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    CheckConcat(inputs, 0, expectedShape, expectedValue);
}

TEST_F(ConcatTests, ConcatFour2DIuputsWithAxis0) {
    const std::vector<TensorDescriptor> inputs = {{{1, 2}, {1, 2}},
                                                  {{2, 2}, {3, 4, 5, 6}},
                                                  {{3, 2}, {7, 8, 9, 10, 11, 12}},
                                                  {{2, 2}, {13, 14, 15, 16}}};
    const std::vector<int32_t> expectedShape = {8, 2};
    const std::vector<float> expectedValue = {1, 2,  3,  4,  5,  6,  7,  8,
                                              9, 10, 11, 12, 13, 14, 15, 16};
    CheckConcat(inputs, 0, expectedShape, expectedValue);
}

TEST_F(ConcatTests, ConcatFive2DIuputsWithAxis0) {
    const std::vector<TensorDescriptor> inputs = {{{1, 2}, {1, 2}},
                                                  {{2, 2}, {3, 4, 5, 6}},
                                                  {{3, 2}, {7, 8, 9, 10, 11, 12}},
                                                  {{2, 2}, {13, 14, 15, 16}},
                                                  {{1, 2}, {17, 18}}};
    const std::vector<int32_t> expectedShape = {{9, 2}};
    const std::vector<float> expectedValue = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                                              10, 11, 12, 13, 14, 15, 16, 17, 18};
    CheckConcat(inputs, 0, expectedShape, expectedValue);
}

TEST_F(ConcatTests, ConcatTwo2DIuputsWithAxis1) {
    const std::vector<TensorDescriptor> inputs = {{{2, 1}, {1, 2}}, {{2, 2}, {3, 4, 5, 6}}};
    const std::vector<int32_t> expectedShape = {2, 3};
    const std::vector<float> expectedValue = {1, 3, 4, 2, 5, 6};
    CheckConcat(inputs, 1, expectedShape, expectedValue);
}

TEST_F(ConcatTests, ConcatThree2DIuputsWithAxis1) {
    const std::vector<TensorDescriptor> inputs = {
        {{2, 1}, {1, 2}}, {{2, 2}, {3, 4, 5, 6}}, {{2, 1}, {7, 8}}};
    const std::vector<int32_t> expectedShape = {2, 4};
    const std::vector<float> expectedValue = {1, 3, 4, 7, 2, 5, 6, 8};
    CheckConcat(inputs, 1, expectedShape, expectedValue);
}

TEST_F(ConcatTests, ConcatFour2DIuputsWithAxis1) {
    const std::vector<TensorDescriptor> inputs = {
        {{2, 1}, {1, 2}}, {{2, 2}, {3, 4, 5, 6}}, {{2, 1}, {7, 8}}, {{2, 2}, {9, 10, 11, 12}}};
    const std::vector<int32_t> expectedShape = {2, 6};
    const std::vector<float> expectedValue = {1, 3, 4, 7, 9, 10, 2, 5, 6, 8, 11, 12};
    CheckConcat(inputs, 1, expectedShape, expectedValue);
}

TEST_F(ConcatTests, ConcatFive2DIuputsWithAxis1) {
    const std::vector<TensorDescriptor> inputs = {{{2, 1}, {1, 2}},
                                                  {{2, 2}, {3, 4, 5, 6}},
                                                  {{2, 1}, {7, 8}},
                                                  {{2, 2}, {9, 10, 11, 12}},
                                                  {{2, 1}, {13, 14}}};
    const std::vector<int32_t> expectedShape = {2, 7};
    const std::vector<float> expectedValue = {1, 3, 4, 7, 9, 10, 13, 2, 5, 6, 8, 11, 12, 14};
    CheckConcat(inputs, 1, expectedShape, expectedValue);
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

TEST_F(ConcatTests, ConcatThree3DIuputs) {
    const std::vector<TensorDescriptor> inputs = {{{2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}},
                                                  {{2, 2, 2}, {9, 10, 11, 12, 13, 14, 15, 16}},
                                                  {{2, 2, 2}, {17, 18, 19, 20, 21, 22, 23, 24}}};
    const std::vector<std::vector<int32_t>> expectedShape = {{6, 2, 2}, {2, 6, 2}, {2, 2, 6}};
    const std::vector<std::vector<float>> expectedValue = {
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
        {1, 2, 3, 4, 9, 10, 11, 12, 17, 18, 19, 20, 5, 6, 7, 8, 13, 14, 15, 16, 21, 22, 23, 24},
        {1, 2, 9, 10, 17, 18, 3, 4, 11, 12, 19, 20, 5, 6, 13, 14, 21, 22, 7, 8, 15, 16, 23, 24}};
    const std::vector<int32_t> axes = {0, 1, 2};
    for (size_t i = 0; i < axes.size(); ++i) {
        CheckConcat(inputs, axes[i], expectedShape[i], expectedValue[i]);
    }
}

TEST_F(ConcatTests, ConcatFour3DIuputs) {
    const std::vector<TensorDescriptor> inputs = {{{2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}},
                                                  {{2, 2, 2}, {9, 10, 11, 12, 13, 14, 15, 16}},
                                                  {{2, 2, 2}, {17, 18, 19, 20, 21, 22, 23, 24}},
                                                  {{2, 2, 2}, {25, 26, 27, 28, 29, 30, 31, 32}}};
    const std::vector<std::vector<int32_t>> expectedShape = {{8, 2, 2}, {2, 8, 2}, {2, 2, 8}};
    const std::vector<std::vector<float>> expectedValue = {
        {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
         17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
        {1, 2, 3, 4, 9,  10, 11, 12, 17, 18, 19, 20, 25, 26, 27, 28,
         5, 6, 7, 8, 13, 14, 15, 16, 21, 22, 23, 24, 29, 30, 31, 32},
        {1, 2, 9,  10, 17, 18, 25, 26, 3, 4, 11, 12, 19, 20, 27, 28,
         5, 6, 13, 14, 21, 22, 29, 30, 7, 8, 15, 16, 23, 24, 31, 32}};
    const std::vector<int32_t> axes = {0, 1, 2};
    for (size_t i = 0; i < axes.size(); ++i) {
        CheckConcat(inputs, axes[i], expectedShape[i], expectedValue[i]);
    }
}

TEST_F(ConcatTests, ConcatFive3DIuputs) {
    const std::vector<TensorDescriptor> inputs = {{{2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}},
                                                  {{2, 2, 2}, {9, 10, 11, 12, 13, 14, 15, 16}},
                                                  {{2, 2, 2}, {17, 18, 19, 20, 21, 22, 23, 24}},
                                                  {{2, 2, 2}, {25, 26, 27, 28, 29, 30, 31, 32}},
                                                  {{2, 2, 2}, {33, 34, 35, 36, 37, 38, 39, 40}}};
    const std::vector<std::vector<int32_t>> expectedShape = {{10, 2, 2}, {2, 10, 2}, {2, 2, 10}};
    const std::vector<std::vector<float>> expectedValue = {
        {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
         21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40},
        {1, 2, 3, 4, 9,  10, 11, 12, 17, 18, 19, 20, 25, 26, 27, 28, 33, 34, 35, 36,
         5, 6, 7, 8, 13, 14, 15, 16, 21, 22, 23, 24, 29, 30, 31, 32, 37, 38, 39, 40},
        {1, 2, 9,  10, 17, 18, 25, 26, 33, 34, 3, 4, 11, 12, 19, 20, 27, 28, 35, 36,
         5, 6, 13, 14, 21, 22, 29, 30, 37, 38, 7, 8, 15, 16, 23, 24, 31, 32, 39, 40}};
    const std::vector<int32_t> axes = {0, 1, 2};
    for (size_t i = 0; i < axes.size(); ++i) {
        CheckConcat(inputs, axes[i], expectedShape[i], expectedValue[i]);
    }
}

TEST_F(ConcatTests, ConcatTwo4DIuputs) {
    const std::vector<TensorDescriptor> inputs = {
        {{2, 2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
        {{2, 2, 2, 2}, {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}}};
    const std::vector<std::vector<int32_t>> expectedShape = {
        {4, 2, 2, 2}, {2, 4, 2, 2}, {2, 2, 4, 2}, {2, 2, 2, 4}};
    const std::vector<std::vector<float>> expectedValue = {
        {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
         17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
        {1, 2,  3,  4,  5,  6,  7,  8,  17, 18, 19, 20, 21, 22, 23, 24,
         9, 10, 11, 12, 13, 14, 15, 16, 25, 26, 27, 28, 29, 30, 31, 32},
        {1, 2,  3,  4,  17, 18, 19, 20, 5,  6,  7,  8,  21, 22, 23, 24,
         9, 10, 11, 12, 25, 26, 27, 28, 13, 14, 15, 16, 29, 30, 31, 32},
        {1, 2,  17, 18, 3,  4,  19, 20, 5,  6,  21, 22, 7,  8,  23, 24,
         9, 10, 25, 26, 11, 12, 27, 28, 13, 14, 29, 30, 15, 16, 31, 32}};
    const std::vector<int32_t> axes = {0, 1, 2, 3};
    for (size_t i = 0; i < axes.size(); ++i) {
        CheckConcat(inputs, axes[i], expectedShape[i], expectedValue[i]);
    }
}

TEST_F(ConcatTests, ConcatThree4DIuputs) {
    const std::vector<TensorDescriptor> inputs = {
        {{2, 2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
        {{2, 2, 2, 2}, {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}},
        {{2, 2, 2, 2}, {33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48}}};
    const std::vector<std::vector<int32_t>> expectedShape = {
        {6, 2, 2, 2}, {2, 6, 2, 2}, {2, 2, 6, 2}, {2, 2, 2, 6}};
    const std::vector<std::vector<float>> expectedValue = {
        {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
         17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
         33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48},
        {1,  2,  3,  4,  5,  6,  7,  8,  17, 18, 19, 20, 21, 22, 23, 24,
         33, 34, 35, 36, 37, 38, 39, 40, 9,  10, 11, 12, 13, 14, 15, 16,
         25, 26, 27, 28, 29, 30, 31, 32, 41, 42, 43, 44, 45, 46, 47, 48},
        {1,  2,  3,  4,  17, 18, 19, 20, 33, 34, 35, 36, 5,  6,  7,  8,
         21, 22, 23, 24, 37, 38, 39, 40, 9,  10, 11, 12, 25, 26, 27, 28,
         41, 42, 43, 44, 13, 14, 15, 16, 29, 30, 31, 32, 45, 46, 47, 48},
        {1,  2,  17, 18, 33, 34, 3,  4,  19, 20, 35, 36, 5,  6,  21, 22,
         37, 38, 7,  8,  23, 24, 39, 40, 9,  10, 25, 26, 41, 42, 11, 12,
         27, 28, 43, 44, 13, 14, 29, 30, 45, 46, 15, 16, 31, 32, 47, 48}};
    const std::vector<int32_t> axes = {0, 1, 2, 3};
    for (size_t i = 0; i < axes.size(); ++i) {
        CheckConcat(inputs, axes[i], expectedShape[i], expectedValue[i]);
    }
}

TEST_F(ConcatTests, ConcatFour4DIuputs) {
    const std::vector<TensorDescriptor> inputs = {
        {{2, 2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
        {{2, 2, 2, 2}, {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}},
        {{2, 2, 2, 2}, {33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48}},
        {{2, 2, 2, 2}, {49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64}}};
    const std::vector<std::vector<int32_t>> expectedShape = {
        {8, 2, 2, 2}, {2, 8, 2, 2}, {2, 2, 8, 2}, {2, 2, 2, 8}};
    const std::vector<std::vector<float>> expectedValue = {
        {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
         23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
         45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64},
        {1,  2,  3,  4,  5,  6,  7,  8,  17, 18, 19, 20, 21, 22, 23, 24, 33, 34, 35, 36, 37, 38,
         39, 40, 49, 50, 51, 52, 53, 54, 55, 56, 9,  10, 11, 12, 13, 14, 15, 16, 25, 26, 27, 28,
         29, 30, 31, 32, 41, 42, 43, 44, 45, 46, 47, 48, 57, 58, 59, 60, 61, 62, 63, 64},
        {1,  2,  3,  4,  17, 18, 19, 20, 33, 34, 35, 36, 49, 50, 51, 52, 5,  6,  7,  8,  21, 22,
         23, 24, 37, 38, 39, 40, 53, 54, 55, 56, 9,  10, 11, 12, 25, 26, 27, 28, 41, 42, 43, 44,
         57, 58, 59, 60, 13, 14, 15, 16, 29, 30, 31, 32, 45, 46, 47, 48, 61, 62, 63, 64},
        {1,  2,  17, 18, 33, 34, 49, 50, 3,  4,  19, 20, 35, 36, 51, 52, 5,  6,  21, 22, 37, 38,
         53, 54, 7,  8,  23, 24, 39, 40, 55, 56, 9,  10, 25, 26, 41, 42, 57, 58, 11, 12, 27, 28,
         43, 44, 59, 60, 13, 14, 29, 30, 45, 46, 61, 62, 15, 16, 31, 32, 47, 48, 63, 64}};
    const std::vector<int32_t> axes = {0, 1, 2, 3};
    for (size_t i = 0; i < axes.size(); ++i) {
        CheckConcat(inputs, axes[i], expectedShape[i], expectedValue[i]);
    }
}

TEST_F(ConcatTests, ConcatFive4DIuputs) {
    const std::vector<TensorDescriptor> inputs = {
        {{2, 2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
        {{2, 2, 2, 2}, {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}},
        {{2, 2, 2, 2}, {33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48}},
        {{2, 2, 2, 2}, {49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64}},
        {{2, 2, 2, 2}, {65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80}}};
    const std::vector<std::vector<int32_t>> expectedShape = {
        {10, 2, 2, 2}, {2, 10, 2, 2}, {2, 2, 10, 2}, {2, 2, 2, 10}};
    const std::vector<std::vector<float>> expectedValue = {
        {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
         21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
         41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
         61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80},
        {1,  2,  3,  4,  5,  6,  7,  8,  17, 18, 19, 20, 21, 22, 23, 24, 33, 34, 35, 36,
         37, 38, 39, 40, 49, 50, 51, 52, 53, 54, 55, 56, 65, 66, 67, 68, 69, 70, 71, 72,
         9,  10, 11, 12, 13, 14, 15, 16, 25, 26, 27, 28, 29, 30, 31, 32, 41, 42, 43, 44,
         45, 46, 47, 48, 57, 58, 59, 60, 61, 62, 63, 64, 73, 74, 75, 76, 77, 78, 79, 80},
        {1,  2,  3,  4,  17, 18, 19, 20, 33, 34, 35, 36, 49, 50, 51, 52, 65, 66, 67, 68,
         5,  6,  7,  8,  21, 22, 23, 24, 37, 38, 39, 40, 53, 54, 55, 56, 69, 70, 71, 72,
         9,  10, 11, 12, 25, 26, 27, 28, 41, 42, 43, 44, 57, 58, 59, 60, 73, 74, 75, 76,
         13, 14, 15, 16, 29, 30, 31, 32, 45, 46, 47, 48, 61, 62, 63, 64, 77, 78, 79, 80},
        {1,  2,  17, 18, 33, 34, 49, 50, 65, 66, 3,  4,  19, 20, 35, 36, 51, 52, 67, 68,
         5,  6,  21, 22, 37, 38, 53, 54, 69, 70, 7,  8,  23, 24, 39, 40, 55, 56, 71, 72,
         9,  10, 25, 26, 41, 42, 57, 58, 73, 74, 11, 12, 27, 28, 43, 44, 59, 60, 75, 76,
         13, 14, 29, 30, 45, 46, 61, 62, 77, 78, 15, 16, 31, 32, 47, 48, 63, 64, 79, 80}};
    const std::vector<int32_t> axes = {0, 1, 2, 3};
    for (size_t i = 0; i < axes.size(); ++i) {
        CheckConcat(inputs, axes[i], expectedShape[i], expectedValue[i]);
    }
}

TEST_F(ConcatTests, DISABLED_ConcatTwo1DConstants) {
    const std::vector<TensorDescriptor> inputs = {{{2}, {1, 2}}, {{2}, {3, 4}}};
    const std::vector<int32_t> expectedShape = {4};
    const std::vector<float> expectedValue = {1, 2, 3, 4};
    CheckConcat(inputs, 0, expectedShape, expectedValue, false);
}

TEST_F(ConcatTests, DISABLED_ConcatTwo2DConstants) {
    const std::vector<TensorDescriptor> inputs = {{{2, 2}, {1, 2, 3, 4}}, {{2, 2}, {5, 6, 7, 8}}};
    const std::vector<std::vector<int32_t>> expectedShape = {{4, 2}, {2, 4}};
    const std::vector<std::vector<float>> expectedValue = {{1, 2, 3, 4, 5, 6, 7, 8},
                                                           {1, 2, 5, 6, 3, 4, 7, 8}};
    const std::vector<int32_t> axes = {0, 1};
    for (size_t i = 0; i < axes.size(); ++i) {
        CheckConcat(inputs, axes[i], expectedShape[i], expectedValue[i], false);
    }
}

TEST_F(ConcatTests, DISABLED_ConcatTwo3DConstants) {
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
