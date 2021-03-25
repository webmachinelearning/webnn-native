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

class Conv2dTests : public WebnnTest {
  protected:
    struct Tensor {
        const std::vector<int32_t> shape;
        std::vector<float> value;
    };
    enum FusedActivation { NONE, RELU, RELU6 };
    void CheckConv2d(const Tensor& input,
                     const Tensor& filter,
                     const Tensor& expected,
                     const ml::Conv2dOptions* options = nullptr,
                     const Tensor& bias = {},
                     FusedActivation activation = FusedActivation::NONE) {
        const ml::GraphBuilder builder = ml::CreateGraphBuilder(GetContext());
        const ml::Operand inputOperand = utils::BuildInput(builder, "input", input.shape);
        const ml::Operand filterOperand = utils::BuildConstant(
            builder, filter.shape, filter.value.data(), filter.value.size() * sizeof(float));
        ml::Operand output = builder.Conv2d(inputOperand, filterOperand, options);
        if (!bias.value.empty()) {
            const ml::Operand biasOperand = utils::BuildConstant(
                builder, bias.shape, bias.value.data(), bias.value.size() * sizeof(float));
            output = builder.Add(output, biasOperand);
        }
        std::vector<float> minValue = {0};
        std::vector<float> maxValue = {6};
        if (activation == FusedActivation::RELU || activation == FusedActivation::RELU6) {
            ml::ClampOptions clampOptions;
            clampOptions.minValue =
                utils::BuildConstant(builder, {}, minValue.data(), minValue.size() * sizeof(float));
            if (activation == FusedActivation::RELU6) {
                clampOptions.maxValue = utils::BuildConstant(builder, {}, maxValue.data(),
                                                             maxValue.size() * sizeof(float));
            }
            output = builder.Clamp(output, &clampOptions);
        }
        const ml::Graph graph = utils::AwaitBuild(builder, {{"output", output}});
        ASSERT_TRUE(graph);
        const ml::Result result =
            utils::AwaitCompute(
                graph, {{"input", {input.value.data(), input.value.size() * sizeof(float)}}})
                .Get("output");
        EXPECT_TRUE(utils::CheckShape(result, expected.shape));
        EXPECT_TRUE(utils::CheckValue(result, expected.value));
    }
};

TEST_F(Conv2dTests, Conv2dWithPaddingDefault) {
    Tensor input = {{1, 1, 5, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 1, 3, 3}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 1, 5, 5},
                       {12.,  21., 27., 33.,  24.,  33.,  54.,  63., 72.,  51.,  63.,  99., 108.,
                        117., 81., 93., 144., 153., 162., 111., 72., 111., 117., 123., 84.}};
    utils::Conv2dOptions options;
    options.padding = {1, 1, 1, 1};
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithPaddingNchwOihw) {
    Tensor input = {{1, 1, 5, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 1, 3, 3}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 1, 5, 5},
                       {12.,  21., 27., 33.,  24.,  33.,  54.,  63., 72.,  51.,  63.,  99., 108.,
                        117., 81., 93., 144., 153., 162., 111., 72., 111., 117., 123., 84.}};
    utils::Conv2dOptions options;
    options.padding = {1, 1, 1, 1};
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Oihw;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithPaddingNchwHwio) {
    Tensor input = {{1, 1, 5, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{3, 3, 1, 1}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 1, 5, 5},
                       {12.,  21., 27., 33.,  24.,  33.,  54.,  63., 72.,  51.,  63.,  99., 108.,
                        117., 81., 93., 144., 153., 162., 111., 72., 111., 117., 123., 84.}};
    utils::Conv2dOptions options;
    options.padding = {1, 1, 1, 1};
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Hwio;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithPaddingNchwOhwi) {
    Tensor input = {{1, 1, 5, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 3, 3, 1}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 1, 5, 5},
                       {12.,  21., 27., 33.,  24.,  33.,  54.,  63., 72.,  51.,  63.,  99., 108.,
                        117., 81., 93., 144., 153., 162., 111., 72., 111., 117., 123., 84.}};
    utils::Conv2dOptions options;
    options.padding = {1, 1, 1, 1};
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Ohwi;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithPaddingNchwIhwo) {
    Tensor input = {{1, 1, 5, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 3, 3, 1}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 1, 5, 5},
                       {12.,  21., 27., 33.,  24.,  33.,  54.,  63., 72.,  51.,  63.,  99., 108.,
                        117., 81., 93., 144., 153., 162., 111., 72., 111., 117., 123., 84.}};
    utils::Conv2dOptions options;
    options.padding = {1, 1, 1, 1};
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Ihwo;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithPaddingNhwcOihw) {
    Tensor input = {{1, 5, 5, 1}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 1, 3, 3}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 5, 5, 1},
                       {12.,  21., 27., 33.,  24.,  33.,  54.,  63., 72.,  51.,  63.,  99., 108.,
                        117., 81., 93., 144., 153., 162., 111., 72., 111., 117., 123., 84.}};
    utils::Conv2dOptions options;
    options.padding = {1, 1, 1, 1};
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Oihw;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithPaddingNhwcHwio) {
    Tensor input = {{1, 5, 5, 1}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{3, 3, 1, 1}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 5, 5, 1},
                       {12.,  21., 27., 33.,  24.,  33.,  54.,  63., 72.,  51.,  63.,  99., 108.,
                        117., 81., 93., 144., 153., 162., 111., 72., 111., 117., 123., 84.}};
    utils::Conv2dOptions options;
    options.padding = {1, 1, 1, 1};
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Hwio;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithPaddingNhwcOhwi) {
    Tensor input = {{1, 5, 5, 1}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 3, 3, 1}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 5, 5, 1},
                       {12.,  21., 27., 33.,  24.,  33.,  54.,  63., 72.,  51.,  63.,  99., 108.,
                        117., 81., 93., 144., 153., 162., 111., 72., 111., 117., 123., 84.}};
    utils::Conv2dOptions options;
    options.padding = {1, 1, 1, 1};
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Ohwi;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithPaddingNhwIhwo) {
    Tensor input = {{1, 5, 5, 1}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 3, 3, 1}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 5, 5, 1},
                       {12.,  21., 27., 33.,  24.,  33.,  54.,  63., 72.,  51.,  63.,  99., 108.,
                        117., 81., 93., 144., 153., 162., 111., 72., 111., 117., 123., 84.}};
    utils::Conv2dOptions options;
    options.padding = {1, 1, 1, 1};
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Ihwo;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithoutPaddingDefault) {
    Tensor input = {{1, 1, 5, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 1, 3, 3}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 1, 3, 3}, {54., 63., 72., 99., 108., 117., 144., 153., 162.}};
    CheckConv2d(input, filter, expected);
}

TEST_F(Conv2dTests, Conv2dWithoutPaddingNchwHwio) {
    Tensor input = {{1, 1, 5, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{3, 3, 1, 1}, std::vector<float>(9, 1)};
    utils::Conv2dOptions options;
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Hwio;
    Tensor expected = {{1, 1, 3, 3}, {54., 63., 72., 99., 108., 117., 144., 153., 162.}};
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithoutPaddingNchwOhwi) {
    Tensor input = {{1, 1, 5, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 3, 3, 1}, std::vector<float>(9, 1)};
    utils::Conv2dOptions options;
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Ohwi;
    Tensor expected = {{1, 1, 3, 3}, {54., 63., 72., 99., 108., 117., 144., 153., 162.}};
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithoutPaddingNchwIhwo) {
    Tensor input = {{1, 1, 5, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 3, 3, 1}, std::vector<float>(9, 1)};
    utils::Conv2dOptions options;
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Ihwo;
    Tensor expected = {{1, 1, 3, 3}, {54., 63., 72., 99., 108., 117., 144., 153., 162.}};
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithoutPaddingNhwcOihw) {
    Tensor input = {{1, 5, 5, 1}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 1, 3, 3}, std::vector<float>(9, 1)};
    utils::Conv2dOptions options;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Oihw;
    Tensor expected = {{1, 3, 3, 1}, {54., 63., 72., 99., 108., 117., 144., 153., 162.}};
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithoutPaddingNhwcHwio) {
    Tensor input = {{1, 5, 5, 1}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{3, 3, 1, 1}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 3, 3, 1}, {54., 63., 72., 99., 108., 117., 144., 153., 162.}};
    utils::Conv2dOptions options;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Hwio;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithoutPaddingNhwcOhwi) {
    Tensor input = {{1, 5, 5, 1}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 3, 3, 1}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 3, 3, 1}, {54., 63., 72., 99., 108., 117., 144., 153., 162.}};
    utils::Conv2dOptions options;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Ohwi;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithoutPaddingNhwcIhwo) {
    Tensor input = {{1, 5, 5, 1}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 3, 3, 1}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 3, 3, 1}, {54., 63., 72., 99., 108., 117., 144., 153., 162.}};
    utils::Conv2dOptions options;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Ihwo;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithStrides2AndPaddingDefault) {
    Tensor input = {{1, 1, 7, 5},
                    {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                     18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34}};
    Tensor filter = {{1, 1, 3, 3}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 1, 4, 3},
                       {12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.}};
    utils::Conv2dOptions options;
    options.padding = {1, 1, 1, 1};
    options.strides = {2, 2};
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithStrides2AndPaddingNchwHwio) {
    Tensor input = {{1, 1, 7, 5},
                    {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                     18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34}};
    Tensor filter = {{3, 3, 1, 1}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 1, 4, 3},
                       {12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.}};
    utils::Conv2dOptions options;
    options.padding = {1, 1, 1, 1};
    options.strides = {2, 2};
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Hwio;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithStrides2AndPaddingNchwOhwi) {
    Tensor input = {{1, 1, 7, 5},
                    {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                     18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34}};
    Tensor filter = {{1, 3, 3, 1}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 1, 4, 3},
                       {12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.}};
    utils::Conv2dOptions options;
    options.padding = {1, 1, 1, 1};
    options.strides = {2, 2};
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Ohwi;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithStrides2AndPaddingNchwIhwo) {
    Tensor input = {{1, 1, 7, 5},
                    {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                     18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34}};
    Tensor filter = {{1, 3, 3, 1}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 1, 4, 3},
                       {12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.}};
    utils::Conv2dOptions options;
    options.padding = {1, 1, 1, 1};
    options.strides = {2, 2};
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Ihwo;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithStrides2AndPaddingNhwcOihw) {
    Tensor input = {{1, 7, 5, 1},
                    {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                     18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34}};
    Tensor filter = {{1, 1, 3, 3}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 4, 3, 1},
                       {12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.}};
    utils::Conv2dOptions options;
    options.padding = {1, 1, 1, 1};
    options.strides = {2, 2};
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Oihw;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithStrides2AndPaddingNhwcHwio) {
    Tensor input = {{1, 7, 5, 1},
                    {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                     18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34}};
    Tensor filter = {{3, 3, 1, 1}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 4, 3, 1},
                       {12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.}};
    utils::Conv2dOptions options;
    options.padding = {1, 1, 1, 1};
    options.strides = {2, 2};
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Hwio;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithStrides2AndPaddingNhwcOhwi) {
    Tensor input = {{1, 7, 5, 1},
                    {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                     18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34}};
    Tensor filter = {{1, 3, 3, 1}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 4, 3, 1},
                       {12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.}};
    utils::Conv2dOptions options;
    options.padding = {1, 1, 1, 1};
    options.strides = {2, 2};
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Ohwi;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithStrides2AndPaddingNhwcIhwo) {
    Tensor input = {{1, 7, 5, 1},
                    {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                     18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34}};
    Tensor filter = {{1, 3, 3, 1}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 4, 3, 1},
                       {12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.}};
    utils::Conv2dOptions options;
    options.padding = {1, 1, 1, 1};
    options.strides = {2, 2};
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Ihwo;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithStrides2AndAsymetricPaddingDefault) {
    Tensor input = {{1, 1, 5, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 1, 4, 2}, std::vector<float>(8, 1)};
    Tensor expected = {{1, 1, 3, 3}, {33, 45, 27, 104, 120, 66, 72, 80, 43}};
    utils::Conv2dOptions options;
    options.padding = {1, 2, 0, 1};
    options.strides = {2, 2};
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithStrides2AndAsymetricPaddingNchwHwio) {
    Tensor input = {{1, 1, 5, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{4, 2, 1, 1}, std::vector<float>(8, 1)};
    Tensor expected = {{1, 1, 3, 3}, {33, 45, 27, 104, 120, 66, 72, 80, 43}};
    utils::Conv2dOptions options;
    options.padding = {1, 2, 0, 1};
    options.strides = {2, 2};
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Hwio;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithStrides2AndAsymetricPaddingNchwOhwi) {
    Tensor input = {{1, 1, 5, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 4, 2, 1}, std::vector<float>(8, 1)};
    Tensor expected = {{1, 1, 3, 3}, {33, 45, 27, 104, 120, 66, 72, 80, 43}};
    utils::Conv2dOptions options;
    options.padding = {1, 2, 0, 1};
    options.strides = {2, 2};
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Ohwi;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithStrides2AndAsymetricPaddingNchwIhwo) {
    Tensor input = {{1, 1, 5, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 4, 2, 1}, std::vector<float>(8, 1)};
    Tensor expected = {{1, 1, 3, 3}, {33, 45, 27, 104, 120, 66, 72, 80, 43}};
    utils::Conv2dOptions options;
    options.padding = {1, 2, 0, 1};
    options.strides = {2, 2};
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Ihwo;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithStrides2AndAsymetricPaddingNhwcOihw) {
    Tensor input = {{1, 5, 5, 1}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 1, 4, 2}, std::vector<float>(8, 1)};
    Tensor expected = {{1, 3, 3, 1}, {33, 45, 27, 104, 120, 66, 72, 80, 43}};
    utils::Conv2dOptions options;
    options.padding = {1, 2, 0, 1};
    options.strides = {2, 2};
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Oihw;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithStrides2AndAsymetricPaddingNhwcHwio) {
    Tensor input = {{1, 5, 5, 1}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{4, 2, 1, 1}, std::vector<float>(8, 1)};
    Tensor expected = {{1, 3, 3, 1}, {33, 45, 27, 104, 120, 66, 72, 80, 43}};
    utils::Conv2dOptions options;
    options.padding = {1, 2, 0, 1};
    options.strides = {2, 2};
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Hwio;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithStrides2AndAsymetricPaddingNhwcOhwi) {
    Tensor input = {{1, 5, 5, 1}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    const std::vector<float> filterData(8, 1);
    Tensor filter = {{1, 4, 2, 1}, std::vector<float>(8, 1)};
    Tensor expected = {{1, 3, 3, 1}, {33, 45, 27, 104, 120, 66, 72, 80, 43}};
    utils::Conv2dOptions options;
    options.padding = {1, 2, 0, 1};
    options.strides = {2, 2};
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Ohwi;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithStrides2AndAsymetricPaddingNhwcIhwo) {
    Tensor input = {{1, 5, 5, 1}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    const std::vector<float> filterData(8, 1);
    Tensor filter = {{1, 4, 2, 1}, std::vector<float>(8, 1)};
    Tensor expected = {{1, 3, 3, 1}, {33, 45, 27, 104, 120, 66, 72, 80, 43}};
    utils::Conv2dOptions options;
    options.padding = {1, 2, 0, 1};
    options.strides = {2, 2};
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Ihwo;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, FusedDepthwiseConv2dDefault) {
    Tensor input = {{1, 4, 2, 2}, {10, 10, 10, 10, 21, 22, 23, 24, 10, 20, 30, 40, 0, 0, 0, 0}};
    Tensor filter = {{4, 1, 2, 2},
                     {0.25, 0.25, 0.25, 0.25, 0.0, 1.0, 0.0, 1.0, 10.0, 20.0, 30.0, 40.0, 50.0,
                      50.0, 50.0, 50.0}};
    Tensor bias = {{1, 4, 1, 1}, {{6000, 7000, 8000, 9000}}};
    Tensor expected = {{1, 4, 1, 1}, {6010, 7046, 11000, 9000}};
    utils::Conv2dOptions options;
    options.groups = 4;
    CheckConv2d(input, filter, expected, options.AsPtr(), bias);
}

TEST_F(Conv2dTests, FusedDepthwiseConv2dNchwHwio) {
    Tensor input = {{1, 4, 2, 2}, {10, 10, 10, 10, 21, 22, 23, 24, 10, 20, 30, 40, 0, 0, 0, 0}};
    Tensor filter = {{2, 2, 1, 4},
                     {0.25, 0.0, 10.0, 50.0, 0.25, 1.0, 20.0, 50.0, 0.25, 0.0, 30.0, 50.0, 0.25,
                      1.0, 40.0, 50.0}};
    Tensor bias = {{1, 4, 1, 1}, {{6000, 7000, 8000, 9000}}};
    Tensor expected = {{1, 4, 1, 1}, {6010, 7046, 11000, 9000}};
    utils::Conv2dOptions options;
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Hwio;
    options.groups = 4;
    CheckConv2d(input, filter, expected, options.AsPtr(), bias);
}

TEST_F(Conv2dTests, FusedDepthwiseConv2dNchwOhwi) {
    Tensor input = {{1, 4, 2, 2}, {10, 10, 10, 10, 21, 22, 23, 24, 10, 20, 30, 40, 0, 0, 0, 0}};
    Tensor filter = {{4, 2, 2, 1},
                     {0.25, 0.25, 0.25, 0.25, 0.0, 1.0, 0.0, 1.0, 10.0, 20.0, 30.0, 40.0, 50.0,
                      50.0, 50.0, 50.0}};
    Tensor bias = {{1, 4, 1, 1}, {{6000, 7000, 8000, 9000}}};
    Tensor expected = {{1, 4, 1, 1}, {6010, 7046, 11000, 9000}};
    utils::Conv2dOptions options;
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Ohwi;
    options.groups = 4;
    CheckConv2d(input, filter, expected, options.AsPtr(), bias);
}

TEST_F(Conv2dTests, FusedDepthwiseConv2dNchwIhwo) {
    Tensor input = {{1, 4, 2, 2}, {10, 10, 10, 10, 21, 22, 23, 24, 10, 20, 30, 40, 0, 0, 0, 0}};
    Tensor filter = {{1, 2, 2, 4},
                     {0.25, 0.0, 10.0, 50.0, 0.25, 1.0, 20.0, 50.0, 0.25, 0.0, 30.0, 50.0, 0.25,
                      1.0, 40.0, 50.0}};
    Tensor bias = {{1, 4, 1, 1}, {{6000, 7000, 8000, 9000}}};
    Tensor expected = {{1, 4, 1, 1}, {6010, 7046, 11000, 9000}};
    utils::Conv2dOptions options;
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Ihwo;
    options.groups = 4;
    CheckConv2d(input, filter, expected, options.AsPtr(), bias);
}

TEST_F(Conv2dTests, FusedDepthwiseConv2dWithNhwcOihw) {
    Tensor input = {{1, 2, 2, 4}, {10, 21, 10, 0, 10, 22, 20, 0, 10, 23, 30, 0, 10, 24, 40, 0}};
    Tensor filter = {{4, 1, 2, 2},
                     {0.25, 0.25, 0.25, 0.25, 0.0, 1.0, 0.0, 1.0, 10.0, 20.0, 30.0, 40.0, 50.0,
                      50.0, 50.0, 50.0}};
    Tensor bias = {{4}, {{6000, 7000, 8000, 9000}}};
    Tensor expected = {{1, 1, 1, 4}, {6010, 7046, 11000, 9000}};
    utils::Conv2dOptions options;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Oihw;
    options.groups = 4;
    CheckConv2d(input, filter, expected, options.AsPtr(), bias);
}

TEST_F(Conv2dTests, FusedDepthwiseConv2dWithNhwcHwio) {
    Tensor input = {{1, 2, 2, 4}, {10, 21, 10, 0, 10, 22, 20, 0, 10, 23, 30, 0, 10, 24, 40, 0}};
    Tensor filter = {{2, 2, 1, 4},
                     {0.25, 0.0, 10.0, 50.0, 0.25, 1.0, 20.0, 50.0, 0.25, 0.0, 30.0, 50.0, 0.25,
                      1.0, 40.0, 50.0}};
    Tensor bias = {{4}, {{6000, 7000, 8000, 9000}}};
    Tensor expected = {{1, 1, 1, 4}, {6010, 7046, 11000, 9000}};
    utils::Conv2dOptions options;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Hwio;
    options.groups = 4;
    CheckConv2d(input, filter, expected, options.AsPtr(), bias);
}

TEST_F(Conv2dTests, FusedDepthwiseConv2dWithNhwcOhwi) {
    Tensor input = {{1, 2, 2, 4}, {10, 21, 10, 0, 10, 22, 20, 0, 10, 23, 30, 0, 10, 24, 40, 0}};
    Tensor filter = {{4, 2, 2, 1},
                     {0.25, 0.25, 0.25, 0.25, 0.0, 1.0, 0.0, 1.0, 10.0, 20.0, 30.0, 40.0, 50.0,
                      50.0, 50.0, 50.0}};
    Tensor bias = {{4}, {{6000, 7000, 8000, 9000}}};
    Tensor expected = {{1, 1, 1, 4}, {6010, 7046, 11000, 9000}};
    utils::Conv2dOptions options;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Ohwi;
    options.groups = 4;
    CheckConv2d(input, filter, expected, options.AsPtr(), bias);
}

TEST_F(Conv2dTests, FusedDepthwiseConv2dWithNhwcIhwo) {
    Tensor input = {{1, 2, 2, 4}, {10, 21, 10, 0, 10, 22, 20, 0, 10, 23, 30, 0, 10, 24, 40, 0}};
    Tensor filter = {{1, 2, 2, 4},
                     {0.25, 0.0, 10.0, 50.0, 0.25, 1.0, 20.0, 50.0, 0.25, 0.0, 30.0, 50.0, 0.25,
                      1.0, 40.0, 50.0}};
    Tensor bias = {{4}, {{6000, 7000, 8000, 9000}}};
    Tensor expected = {{1, 1, 1, 4}, {6010, 7046, 11000, 9000}};
    utils::Conv2dOptions options;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Ihwo;
    options.groups = 4;
    CheckConv2d(input, filter, expected, options.AsPtr(), bias);
}

TEST_F(Conv2dTests, DepthwiseConv2dWithNchwOihw) {
    Tensor input = {{1, 4, 2, 2}, {10, 10, 10, 10, 21, 22, 23, 24, 10, 20, 30, 40, 0, 0, 0, 0}};
    Tensor filter = {{4, 1, 2, 2},
                     {0.25, 0.25, 0.25, 0.25, 0.0, 1.0, 0.0, 1.0, 10.0, 20.0, 30.0, 40.0, 50.0,
                      50.0, 50.0, 50.0}};
    Tensor expected = {{1, 4, 1, 1}, {10, 46, 3000, 0}};
    utils::Conv2dOptions options;
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Oihw;
    options.groups = 4;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, FusedConv2dWithPaddingDefault) {
    Tensor input = {{1, 1, 5, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 1, 3, 3}, std::vector<float>(9, 1)};
    Tensor bias = {{1}, {-100}};
    Tensor expected = {{1, 1, 5, 5}, {0.,  0., 0., 0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0., 8.,
                                      17., 0., 0., 44., 53., 62., 11., 0., 11., 17., 23., 0.}};
    utils::Conv2dOptions options;
    options.padding = {1, 1, 1, 1};
    CheckConv2d(input, filter, expected, options.AsPtr(), bias, FusedActivation::RELU);
}

TEST_F(Conv2dTests, FusedConv2dWithPaddingNchwHwio) {
    Tensor input = {{1, 1, 5, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{3, 3, 1, 1}, std::vector<float>(9, 1)};
    Tensor bias = {{1}, {-100}};
    Tensor expected = {{1, 1, 5, 5}, {0.,  0., 0., 0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0., 8.,
                                      17., 0., 0., 44., 53., 62., 11., 0., 11., 17., 23., 0.}};
    utils::Conv2dOptions options;
    options.padding = {1, 1, 1, 1};
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Hwio;
    CheckConv2d(input, filter, expected, options.AsPtr(), bias, FusedActivation::RELU);
}

TEST_F(Conv2dTests, FusedConv2dWithPaddingNchwOhwi) {
    Tensor input = {{1, 1, 5, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 3, 3, 1}, std::vector<float>(9, 1)};
    Tensor bias = {{1}, {-100}};
    Tensor expected = {{1, 1, 5, 5}, {0.,  0., 0., 0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0., 8.,
                                      17., 0., 0., 44., 53., 62., 11., 0., 11., 17., 23., 0.}};
    utils::Conv2dOptions options;
    options.padding = {1, 1, 1, 1};
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Ohwi;
    CheckConv2d(input, filter, expected, options.AsPtr(), bias, FusedActivation::RELU);
}

TEST_F(Conv2dTests, FusedConv2dWithPaddingNchwIhwo) {
    Tensor input = {{1, 1, 5, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 3, 3, 1}, std::vector<float>(9, 1)};
    Tensor bias = {{1}, {-100}};
    Tensor expected = {{1, 1, 5, 5}, {0.,  0., 0., 0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0., 8.,
                                      17., 0., 0., 44., 53., 62., 11., 0., 11., 17., 23., 0.}};
    utils::Conv2dOptions options;
    options.padding = {1, 1, 1, 1};
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Ihwo;
    CheckConv2d(input, filter, expected, options.AsPtr(), bias, FusedActivation::RELU);
}

TEST_F(Conv2dTests, FusedConv2dWithPaddingNhwcOihw) {
    Tensor input = {{1, 5, 5, 1}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 1, 3, 3}, std::vector<float>(9, 1)};
    Tensor bias = {{1}, {-100}};
    Tensor expected = {{1, 5, 5, 1}, {0.,  0., 0., 0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0., 8.,
                                      17., 0., 0., 44., 53., 62., 11., 0., 11., 17., 23., 0.}};
    utils::Conv2dOptions options;
    options.padding = {1, 1, 1, 1};
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Oihw;
    CheckConv2d(input, filter, expected, options.AsPtr(), bias, FusedActivation::RELU);
}

TEST_F(Conv2dTests, FusedConv2dWithPaddingNhwcHwio) {
    Tensor input = {{1, 5, 5, 1}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{3, 3, 1, 1}, std::vector<float>(9, 1)};
    Tensor bias = {{1}, {-100}};
    Tensor expected = {{1, 5, 5, 1}, {0.,  0., 0., 0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0., 8.,
                                      17., 0., 0., 44., 53., 62., 11., 0., 11., 17., 23., 0.}};
    utils::Conv2dOptions options;
    options.padding = {1, 1, 1, 1};
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Hwio;
    CheckConv2d(input, filter, expected, options.AsPtr(), bias, FusedActivation::RELU);
}

TEST_F(Conv2dTests, FusedConv2dWithPaddingNhwcOhwi) {
    Tensor input = {{1, 5, 5, 1}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 3, 3, 1}, std::vector<float>(9, 1)};
    Tensor bias = {{1}, {-100}};
    Tensor expected = {{1, 5, 5, 1}, {0.,  0., 0., 0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0., 8.,
                                      17., 0., 0., 44., 53., 62., 11., 0., 11., 17., 23., 0.}};
    utils::Conv2dOptions options;
    options.padding = {1, 1, 1, 1};
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Ohwi;
    CheckConv2d(input, filter, expected, options.AsPtr(), bias, FusedActivation::RELU);
}

TEST_F(Conv2dTests, FusedConv2dWithPaddingNhwcIhwo) {
    Tensor input = {{1, 5, 5, 1}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 3, 3, 1}, std::vector<float>(9, 1)};
    Tensor bias = {{1}, {-100}};
    Tensor expected = {{1, 5, 5, 1}, {0.,  0., 0., 0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0., 8.,
                                      17., 0., 0., 44., 53., 62., 11., 0., 11., 17., 23., 0.}};
    utils::Conv2dOptions options;
    options.padding = {1, 1, 1, 1};
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Ihwo;
    CheckConv2d(input, filter, expected, options.AsPtr(), bias, FusedActivation::RELU);
}

TEST_F(Conv2dTests, Conv2dWithAutoPadSameLowerDefault) {
    Tensor input = {{1, 1, 5, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 1, 3, 3}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 1, 3, 3}, {12., 27., 24., 63., 108., 81., 72., 117., 84.}};
    utils::Conv2dOptions options;
    options.strides = {2, 2};
    options.autoPad = ml::AutoPad::SameLower;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithAutoPadSameLowerNchwHwio) {
    Tensor input = {{1, 1, 5, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{3, 3, 1, 1}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 1, 3, 3}, {12., 27., 24., 63., 108., 81., 72., 117., 84.}};
    utils::Conv2dOptions options;
    options.strides = {2, 2};
    options.autoPad = ml::AutoPad::SameLower;
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Hwio;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithAutoPadSameLowerNchwOhwi) {
    Tensor input = {{1, 1, 5, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 3, 3, 1}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 1, 3, 3}, {12., 27., 24., 63., 108., 81., 72., 117., 84.}};
    utils::Conv2dOptions options;
    options.strides = {2, 2};
    options.autoPad = ml::AutoPad::SameLower;
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Ohwi;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithAutoPadSameLowerNchwIhwo) {
    Tensor input = {{1, 1, 5, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 3, 3, 1}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 1, 3, 3}, {12., 27., 24., 63., 108., 81., 72., 117., 84.}};
    utils::Conv2dOptions options;
    options.strides = {2, 2};
    options.autoPad = ml::AutoPad::SameLower;
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Ihwo;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithAutoPadSameLowerNhwcOihw) {
    Tensor input = {{1, 5, 5, 1}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 1, 3, 3}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 3, 3, 1}, {12., 27., 24., 63., 108., 81., 72., 117., 84.}};
    utils::Conv2dOptions options;
    options.strides = {2, 2};
    options.autoPad = ml::AutoPad::SameLower;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Oihw;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithAutoPadSameLowerNhwcHwio) {
    Tensor input = {{1, 5, 5, 1}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{3, 3, 1, 1}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 3, 3, 1}, {12., 27., 24., 63., 108., 81., 72., 117., 84.}};
    utils::Conv2dOptions options;
    options.strides = {2, 2};
    options.autoPad = ml::AutoPad::SameLower;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Hwio;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithAutoPadSameLowerNhwcOhwi) {
    Tensor input = {{1, 4, 4, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};
    Tensor filter = {{1, 3, 3, 1}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 2, 2, 1}, {10., 24., 51., 90.}};
    utils::Conv2dOptions options;
    options.strides = {2, 2};
    options.autoPad = ml::AutoPad::SameLower;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Ohwi;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithAutoPadSameLowerNhwcIhwo) {
    Tensor input = {{1, 4, 4, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};
    Tensor filter = {{1, 3, 3, 1}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 2, 2, 1}, {10., 24., 51., 90.}};
    utils::Conv2dOptions options;
    options.strides = {2, 2};
    options.autoPad = ml::AutoPad::SameLower;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Ihwo;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithAutoPadSameUpperDefault) {
    Tensor input = {{1, 1, 5, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 1, 3, 3}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 1, 3, 3}, {12., 27., 24., 63., 108., 81., 72., 117., 84.}};
    utils::Conv2dOptions options;
    options.strides = {2, 2};
    options.autoPad = ml::AutoPad::SameUpper;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithAutoPadSameUpperNchwHwio) {
    Tensor input = {{1, 1, 4, 4}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};
    Tensor filter = {{3, 3, 1, 1}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 1, 2, 2}, {45., 39., 66., 50.}};
    utils::Conv2dOptions options;
    options.strides = {2, 2};
    options.autoPad = ml::AutoPad::SameUpper;
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Hwio;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithAutoPadSameUpperNchwOhwi) {
    Tensor input = {{1, 1, 4, 4}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};
    Tensor filter = {{1, 3, 3, 1}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 1, 2, 2}, {45., 39., 66., 50.}};
    utils::Conv2dOptions options;
    options.strides = {2, 2};
    options.autoPad = ml::AutoPad::SameUpper;
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Ohwi;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithAutoPadSameUpperNchwIhwo) {
    Tensor input = {{1, 1, 4, 4}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};
    Tensor filter = {{1, 3, 3, 1}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 1, 2, 2}, {45., 39., 66., 50.}};
    utils::Conv2dOptions options;
    options.strides = {2, 2};
    options.autoPad = ml::AutoPad::SameUpper;
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Ihwo;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithAutoPadSameUpperNhwcOihw) {
    Tensor input = {{1, 4, 4, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};
    Tensor filter = {{1, 1, 3, 3}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 2, 2, 1}, {45., 39., 66., 50.}};
    utils::Conv2dOptions options;
    options.strides = {2, 2};
    options.autoPad = ml::AutoPad::SameUpper;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Oihw;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithAutoPadSameUpperNhwcHwio) {
    Tensor input = {{1, 4, 4, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};
    Tensor filter = {{3, 3, 1, 1}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 2, 2, 1}, {45., 39., 66., 50.}};
    utils::Conv2dOptions options;
    options.strides = {2, 2};
    options.autoPad = ml::AutoPad::SameUpper;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Hwio;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithAutoPadSameUpperNhwcOhwi) {
    Tensor input = {{1, 4, 4, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};
    Tensor filter = {{1, 3, 3, 1}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 2, 2, 1}, {45., 39., 66., 50.}};
    utils::Conv2dOptions options;
    options.strides = {2, 2};
    options.autoPad = ml::AutoPad::SameUpper;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Ohwi;
    CheckConv2d(input, filter, expected, options.AsPtr());
}

TEST_F(Conv2dTests, Conv2dWithAutoPadSameUpperNhwcIhwo) {
    Tensor input = {{1, 4, 4, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};
    Tensor filter = {{1, 3, 3, 1}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 2, 2, 1}, {45., 39., 66., 50.}};
    utils::Conv2dOptions options;
    options.strides = {2, 2};
    options.autoPad = ml::AutoPad::SameUpper;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Ihwo;
    CheckConv2d(input, filter, expected, options.AsPtr());
}
