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
    void SetUp() override {
        builder = ml::CreateGraphBuilder(GetContext());
    }

  protected:
    struct Tensor {
        std::vector<int32_t> shape;
        std::vector<float> value;
    };

    void CheckConv2d(const Tensor& input,
                     const Tensor& filter,
                     const Tensor& expected,
                     utils::Conv2dOptions options = {},
                     const Tensor& bias = {},
                     utils::FusedActivation activation = utils::FusedActivation::NONE,
                     bool fusion = false,
                     void* activationOptions = nullptr) {
        const ml::Operand x = utils::BuildInput(builder, "input", input.shape);
        const ml::Operand w = utils::BuildConstant(builder, filter.shape, filter.value.data(),
                                                   filter.value.size() * sizeof(float));

        ml::Operand b;
        if (!bias.value.empty()) {
            b = utils::BuildConstant(builder, bias.shape, bias.value.data(),
                                     bias.value.size() * sizeof(float));
        }

        if (fusion) {
            if (!bias.value.empty()) {
                options.bias = b;
            }
            if (activation != utils::FusedActivation::NONE) {
                options.activation =
                    utils::CreateActivationOperator(builder, activation, activationOptions);
            }
        }
        ml::Operand y = builder.Conv2d(x, w, options.AsPtr());

        if (!fusion) {
            if (!bias.value.empty()) {
                if (options.inputLayout == ml::InputOperandLayout::Nchw) {
                    std::vector<int32_t> newShape = std::vector<int32_t>({1, -1, 1, 1});
                    b = builder.Reshape(b, newShape.data(), newShape.size());
                }
                y = builder.Add(y, b);
            }
            if (activation != utils::FusedActivation::NONE) {
                y = utils::CreateActivationOperand(builder, y, activation, activationOptions);
            }
        }

        const ml::Graph graph = utils::Build(builder, {{"output", y}});
        ASSERT_TRUE(graph);
        std::vector<float> result(utils::SizeOfShape(expected.shape));
        utils::Compute(graph, {{"input", input.value}}, {{"output", result}});
        EXPECT_TRUE(utils::CheckValue(result, expected.value));
    }

    ml::GraphBuilder builder;
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dWithoutPaddingNchwOhwi) {
    Tensor input = {{1, 1, 5, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 3, 3, 1}, std::vector<float>(9, 1)};
    utils::Conv2dOptions options;
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Ohwi;
    Tensor expected = {{1, 1, 3, 3}, {54., 63., 72., 99., 108., 117., 144., 153., 162.}};
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dWithoutPaddingNchwIhwo) {
    Tensor input = {{1, 1, 5, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 3, 3, 1}, std::vector<float>(9, 1)};
    utils::Conv2dOptions options;
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Ihwo;
    Tensor expected = {{1, 1, 3, 3}, {54., 63., 72., 99., 108., 117., 144., 153., 162.}};
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dWithoutPaddingNhwcOihw) {
    Tensor input = {{1, 5, 5, 1}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 1, 3, 3}, std::vector<float>(9, 1)};
    utils::Conv2dOptions options;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Oihw;
    Tensor expected = {{1, 3, 3, 1}, {54., 63., 72., 99., 108., 117., 144., 153., 162.}};
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dWithoutPaddingNhwcHwio) {
    Tensor input = {{1, 5, 5, 1}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{3, 3, 1, 1}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 3, 3, 1}, {54., 63., 72., 99., 108., 117., 144., 153., 162.}};
    utils::Conv2dOptions options;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Hwio;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dWithoutPaddingNhwcOhwi) {
    Tensor input = {{1, 5, 5, 1}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 3, 3, 1}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 3, 3, 1}, {54., 63., 72., 99., 108., 117., 144., 153., 162.}};
    utils::Conv2dOptions options;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Ohwi;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dWithoutPaddingNhwcIhwo) {
    Tensor input = {{1, 5, 5, 1}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 3, 3, 1}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 3, 3, 1}, {54., 63., 72., 99., 108., 117., 144., 153., 162.}};
    utils::Conv2dOptions options;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Ihwo;
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dWithStrides2AndAsymetricPaddingDefault) {
    Tensor input = {{1, 1, 5, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 1, 4, 2}, std::vector<float>(8, 1)};
    Tensor expected = {{1, 1, 3, 3}, {33, 45, 27, 104, 120, 66, 72, 80, 43}};
    utils::Conv2dOptions options;
    options.padding = {1, 2, 0, 1};
    options.strides = {2, 2};
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, FusedDepthwiseConv2dDefault) {
    Tensor input = {{1, 4, 2, 2}, {10, 10, 10, 10, 21, 22, 23, 24, 10, 20, 30, 40, 0, 0, 0, 0}};
    Tensor filter = {{4, 1, 2, 2},
                     {0.25, 0.25, 0.25, 0.25, 0.0, 1.0, 0.0, 1.0, 10.0, 20.0, 30.0, 40.0, 50.0,
                      50.0, 50.0, 50.0}};
    Tensor bias = {{4}, {6000, 7000, 8000, 9000}};
    Tensor expected = {{1, 4, 1, 1}, {6010, 7046, 11000, 9000}};
    utils::Conv2dOptions options;
    options.groups = 4;
    CheckConv2d(input, filter, expected, options, bias);
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU, true);
    expected = {{1, 4, 1, 1}, {6, 6, 6, 6}};

    ml::ClampOptions clampOptions = {0, 6};
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU6, true,
                &clampOptions);
}

TEST_F(Conv2dTests, FusedDepthwiseConv2dNchwHwio) {
    Tensor input = {{1, 4, 2, 2}, {10, 10, 10, 10, 21, 22, 23, 24, 10, 20, 30, 40, 0, 0, 0, 0}};
    Tensor filter = {{2, 2, 1, 4},
                     {0.25, 0.0, 10.0, 50.0, 0.25, 1.0, 20.0, 50.0, 0.25, 0.0, 30.0, 50.0, 0.25,
                      1.0, 40.0, 50.0}};
    Tensor bias = {{4}, {6000, 7000, 8000, 9000}};
    Tensor expected = {{1, 4, 1, 1}, {6010, 7046, 11000, 9000}};
    utils::Conv2dOptions options;
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Hwio;
    options.groups = 4;
    CheckConv2d(input, filter, expected, options, bias);
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU, true);
    expected = {{4}, {6, 6, 6, 6}};

    ml::ClampOptions clampOptions = {0, 6};
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU6, true,
                &clampOptions);
}

TEST_F(Conv2dTests, FusedDepthwiseConv2dNchwOhwi) {
    Tensor input = {{1, 4, 2, 2}, {10, 10, 10, 10, 21, 22, 23, 24, 10, 20, 30, 40, 0, 0, 0, 0}};
    Tensor filter = {{4, 2, 2, 1},
                     {0.25, 0.25, 0.25, 0.25, 0.0, 1.0, 0.0, 1.0, 10.0, 20.0, 30.0, 40.0, 50.0,
                      50.0, 50.0, 50.0}};
    Tensor bias = {{4}, {{6000, 7000, 8000, 9000}}};
    Tensor expected = {{1, 4, 1, 1}, {6010, 7046, 11000, 9000}};
    utils::Conv2dOptions options;
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Ohwi;
    options.groups = 4;
    CheckConv2d(input, filter, expected, options, bias);
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU, true);
    expected = {{1, 4, 1, 1}, {6, 6, 6, 6}};

    ml::ClampOptions clampOptions = {0, 6};
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU6, true,
                &clampOptions);
}

TEST_F(Conv2dTests, FusedDepthwiseConv2dNchwIhwo) {
    Tensor input = {{1, 4, 2, 2}, {10, 10, 10, 10, 21, 22, 23, 24, 10, 20, 30, 40, 0, 0, 0, 0}};
    Tensor filter = {{1, 2, 2, 4},
                     {0.25, 0.0, 10.0, 50.0, 0.25, 1.0, 20.0, 50.0, 0.25, 0.0, 30.0, 50.0, 0.25,
                      1.0, 40.0, 50.0}};
    Tensor bias = {{4}, {{6000, 7000, 8000, 9000}}};
    Tensor expected = {{1, 4, 1, 1}, {6010, 7046, 11000, 9000}};
    utils::Conv2dOptions options;
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Ihwo;
    options.groups = 4;
    CheckConv2d(input, filter, expected, options, bias);
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU, true);
    expected = {{1, 4, 1, 1}, {6, 6, 6, 6}};

    ml::ClampOptions clampOptions = {0, 6};
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU6, true,
                &clampOptions);
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
    CheckConv2d(input, filter, expected, options, bias);
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU, true);
    expected = {{1, 1, 1, 4}, {6, 6, 6, 6}};

    ml::ClampOptions clampOptions = {0, 6};
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU6, true,
                &clampOptions);
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
    CheckConv2d(input, filter, expected, options, bias);
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU, true);
    expected = {{1, 1, 1, 4}, {6, 6, 6, 6}};

    ml::ClampOptions clampOptions = {0, 6};
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU6, true,
                &clampOptions);
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
    CheckConv2d(input, filter, expected, options, bias);
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU, true);
    expected = {{1, 1, 1, 4}, {6, 6, 6, 6}};

    ml::ClampOptions clampOptions = {0, 6};
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU6, true,
                &clampOptions);
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
    CheckConv2d(input, filter, expected, options, bias);
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU, true);
    expected = {{1, 1, 1, 4}, {6, 6, 6, 6}};

    ml::ClampOptions clampOptions = {0, 6};
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU6, true,
                &clampOptions);
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
    CheckConv2d(input, filter, expected, options);
    CheckConv2d(input, filter, expected, options, {}, utils::FusedActivation::RELU, true);
    expected = {{1, 4, 1, 1}, {6, 6, 6, 0}};

    ml::ClampOptions clampOptions = {0, 6};
    CheckConv2d(input, filter, expected, options, {}, utils::FusedActivation::RELU6, true,
                &clampOptions);
}

TEST_F(Conv2dTests, FusedDepthwiseConv2dWithExplicitAutoPad) {
    Tensor input = {{1, 2, 3, 3},
                    {10, 10, 10, 10, 10, 10, 10, 10, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29}};
    Tensor filter = {{2, 1, 2, 2}, {0.25, 0.25, 0.25, 0.25, 0.0, 1.0, 0.0, 1.0}};
    Tensor expected = {{1, 2, 3, 3},
                       {10, 10, 5, 10, 10, 5, 5, 5, 2.5, 47, 49, 0, 53, 55, 0, 28, 29, 0}};
    utils::Conv2dOptions options;
    options.groups = 2;
    options.padding = {0, 1, 0, 1};
    options.autoPad = ml::AutoPad::Explicit;
    CheckConv2d(input, filter, expected, options);
    CheckConv2d(input, filter, expected, options, {}, utils::FusedActivation::RELU, true);
    expected = {{1, 2, 3, 3}, {6, 6, 5, 6, 6, 5, 5, 5, 2.5, 6, 6, 0, 6, 6, 0, 6, 6, 0}};

    ml::ClampOptions clampOptions = {0, 6};
    CheckConv2d(input, filter, expected, options, {}, utils::FusedActivation::RELU6, true,
                &clampOptions);
}

TEST_F(Conv2dTests, FusedDepthwiseConv2dWithSameUpperAutoPad) {
    Tensor input = {{1, 2, 3, 3},
                    {10, 10, 10, 10, 10, 10, 10, 10, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29}};
    Tensor filter = {{2, 1, 2, 2}, {0.25, 0.25, 0.25, 0.25, 0.0, 1.0, 0.0, 1.0}};
    Tensor expected = {{1, 2, 3, 3},
                       {10, 10, 5, 10, 10, 5, 5, 5, 2.5, 47, 49, 0, 53, 55, 0, 28, 29, 0}};
    utils::Conv2dOptions options;
    options.groups = 2;
    options.autoPad = ml::AutoPad::SameUpper;
    CheckConv2d(input, filter, expected, options);
    CheckConv2d(input, filter, expected, options, {}, utils::FusedActivation::RELU, true);
    expected = {{1, 2, 3, 3}, {6, 6, 5, 6, 6, 5, 5, 5, 2.5, 6, 6, 0, 6, 6, 0, 6, 6, 0}};

    ml::ClampOptions clampOptions = {0, 6};
    CheckConv2d(input, filter, expected, options, {}, utils::FusedActivation::RELU6, true,
                &clampOptions);
}

TEST_F(Conv2dTests, FusedDepthwiseConv2dWithSameLowerAutoPad) {
    Tensor input = {{1, 2, 3, 3},
                    {10, 10, 10, 10, 10, 10, 10, 10, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29}};
    Tensor filter = {{2, 1, 2, 2}, {0.25, 0.25, 0.25, 0.25, 0.0, 1.0, 0.0, 1.0}};
    Tensor expected = {{1, 2, 3, 3},
                       {2.5, 5, 5, 5, 10, 10, 5, 10, 10, 21, 22, 23, 45, 47, 49, 51, 53, 55}};
    utils::Conv2dOptions options;
    options.groups = 2;
    options.autoPad = ml::AutoPad::SameLower;
    CheckConv2d(input, filter, expected, options);
    CheckConv2d(input, filter, expected, options, {}, utils::FusedActivation::RELU, true);
    expected = {{1, 2, 3, 3}, {2.5, 5, 5, 5, 6, 6, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6}};

    ml::ClampOptions clampOptions = {0, 6};
    CheckConv2d(input, filter, expected, options, {}, utils::FusedActivation::RELU6, true,
                &clampOptions);
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
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU);
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU, true);
    expected = {{1, 1, 5, 5}, {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 6.,
                               6., 0., 0., 6., 6., 6., 6., 0., 6., 6., 6., 0.}};

    ml::ClampOptions clampOptions = {0, 6};
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU6, true,
                &clampOptions);
    expected = {{1, 1, 5, 5},
                {-8.800000190734863,
                 -7.900000095367432,
                 -7.300000190734863,
                 -6.700000286102295,
                 -7.599999904632568,
                 -6.700000286102295,
                 -4.599999904632568,
                 -3.700000047683716,
                 -2.799999952316284,
                 -4.900000095367432,
                 -3.700000047683716,
                 -0.10000000149011612,
                 8,
                 17,
                 -1.899999976158142,
                 -0.699999988079071,
                 44,
                 53,
                 62,
                 11,
                 -2.799999952316284,
                 11,
                 17,
                 23,
                 -1.600000023841858}};
    ml::LeakyReluOptions leakyReluOptions = {0.10000000149011612};
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::LEAKYRELU, true,
                &leakyReluOptions);
    expected = {{1, 1, 5, 5},
                {6.054601485195952e-39,
                 4.906094994852858e-35,
                 1.9792599190321352e-32,
                 7.984904044796711e-30,
                 9.854154449263851e-34,
                 7.984904044796711e-30,
                 1.0530617466355953e-20,
                 8.533047630075754e-17,
                 6.914400150527522e-13,
                 5.242885696424093e-22,
                 8.533047630075754e-17,
                 0.2689414322376251,
                 0.9996646642684937,
                 0.9999999403953552,
                 5.602796449011294e-9,
                 0.0009110511746257544,
                 1,
                 1,
                 1,
                 0.9999833106994629,
                 6.914400150527522e-13,
                 0.9999833106994629,
                 0.9999999403953552,
                 1,
                 1.1253516163378663e-7}};
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::SIGMOID, true);
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
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU);
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU, true);
    expected = {{1, 1, 5, 5}, {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 6.,
                               6., 0., 0., 6., 6., 6., 6., 0., 6., 6., 6., 0.}};

    ml::ClampOptions clampOptions = {0, 6};
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU6, true,
                &clampOptions);
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
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU);
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU, true);
    expected = {{1, 1, 5, 5}, {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 6.,
                               6., 0., 0., 6., 6., 6., 6., 0., 6., 6., 6., 0.}};

    ml::ClampOptions clampOptions = {0, 6};
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU6, true,
                &clampOptions);
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
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU);
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU, true);
    expected = {{1, 1, 5, 5}, {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 6.,
                               6., 0., 0., 6., 6., 6., 6., 0., 6., 6., 6., 0.}};

    ml::ClampOptions clampOptions = {0, 6};
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU6, true,
                &clampOptions);
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
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU);
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU, true);
    expected = {{1, 5, 5, 1}, {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 6.,
                               6., 0., 0., 6., 6., 6., 6., 0., 6., 6., 6., 0.}};

    ml::ClampOptions clampOptions = {0, 6};
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU6, true,
                &clampOptions);
    expected = {{1, 5, 5, 1},
                {-8.800000190734863,
                 -7.900000095367432,
                 -7.300000190734863,
                 -6.700000286102295,
                 -7.599999904632568,
                 -6.700000286102295,
                 -4.599999904632568,
                 -3.700000047683716,
                 -2.799999952316284,
                 -4.900000095367432,
                 -3.700000047683716,
                 -0.10000000149011612,
                 8,
                 17,
                 -1.899999976158142,
                 -0.699999988079071,
                 44,
                 53,
                 62,
                 11,
                 -2.799999952316284,
                 11,
                 17,
                 23,
                 -1.600000023841858}};
    ml::LeakyReluOptions leakyReluOptions = {0.10000000149011612};
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::LEAKYRELU, true,
                &leakyReluOptions);
    expected = {{1, 5, 5, 1},
                {6.054601485195952e-39,
                 4.906094994852858e-35,
                 1.9792599190321352e-32,
                 7.984904044796711e-30,
                 9.854154449263851e-34,
                 7.984904044796711e-30,
                 1.0530617466355953e-20,
                 8.533047630075754e-17,
                 6.914400150527522e-13,
                 5.242885696424093e-22,
                 8.533047630075754e-17,
                 0.2689414322376251,
                 0.9996646642684937,
                 0.9999999403953552,
                 5.602796449011294e-9,
                 0.0009110511746257544,
                 1,
                 1,
                 1,
                 0.9999833106994629,
                 6.914400150527522e-13,
                 0.9999833106994629,
                 0.9999999403953552,
                 1,
                 1.1253516163378663e-7}};
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::SIGMOID, true);
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
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU);
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU, true);
    expected = {{1, 5, 5, 1}, {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 6.,
                               6., 0., 0., 6., 6., 6., 6., 0., 6., 6., 6., 0.}};

    ml::ClampOptions clampOptions = {0, 6};
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU6, true,
                &clampOptions);
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
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU);
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU, true);
    expected = {{1, 5, 5, 1}, {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 6.,
                               6., 0., 0., 6., 6., 6., 6., 0., 6., 6., 6., 0.}};

    ml::ClampOptions clampOptions = {0, 6};
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU6, true,
                &clampOptions);
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
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU);
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU, true);
    expected = {{1, 5, 5, 1}, {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 6.,
                               6., 0., 0., 6., 6., 6., 6., 0., 6., 6., 6., 0.}};

    ml::ClampOptions clampOptions = {0, 6};
    CheckConv2d(input, filter, expected, options, bias, utils::FusedActivation::RELU6, true,
                &clampOptions);
}

TEST_F(Conv2dTests, Conv2dWithAutoPadSameLowerDefault) {
    Tensor input = {{1, 1, 5, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 1, 3, 3}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 1, 3, 3}, {12., 27., 24., 63., 108., 81., 72., 117., 84.}};
    utils::Conv2dOptions options;
    options.strides = {2, 2};
    options.autoPad = ml::AutoPad::SameLower;
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dWithAutoPadSameUpperDefault) {
    Tensor input = {{1, 1, 5, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    Tensor filter = {{1, 1, 3, 3}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 1, 3, 3}, {12., 27., 24., 63., 108., 81., 72., 117., 84.}};
    utils::Conv2dOptions options;
    options.strides = {2, 2};
    options.autoPad = ml::AutoPad::SameUpper;
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
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
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dTransposeDefault) {
    Tensor input = {{1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{2, 1, 3, 3}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 2, 5, 5},
        {0.,  1.,  3.,  3.,  2.,  3.,  8.,  15., 12., 7.,  9.,  21., 36., 27., 15., 9.,  20.,
         33., 24., 13., 6.,  13., 21., 15., 8.,  0.,  1.,  3.,  3.,  2.,  3.,  8.,  15., 12.,
         7.,  9.,  21., 36., 27., 15., 9.,  20., 33., 24., 13., 6.,  13., 21., 15., 8.},
    };
    utils::Conv2dOptions options;
    options.transpose = true;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dTransposeNchwHwio) {
    Tensor input = {{1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{3, 3, 1, 2}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 2, 5, 5},
        {0.,  1.,  3.,  3.,  2.,  3.,  8.,  15., 12., 7.,  9.,  21., 36., 27., 15., 9.,  20.,
         33., 24., 13., 6.,  13., 21., 15., 8.,  0.,  1.,  3.,  3.,  2.,  3.,  8.,  15., 12.,
         7.,  9.,  21., 36., 27., 15., 9.,  20., 33., 24., 13., 6.,  13., 21., 15., 8.},
    };
    utils::Conv2dOptions options;
    options.transpose = true;
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Hwio;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dTransposeNchwOhwi) {
    Tensor input = {{1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{2, 3, 3, 1}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 2, 5, 5},
        {0.,  1.,  3.,  3.,  2.,  3.,  8.,  15., 12., 7.,  9.,  21., 36., 27., 15., 9.,  20.,
         33., 24., 13., 6.,  13., 21., 15., 8.,  0.,  1.,  3.,  3.,  2.,  3.,  8.,  15., 12.,
         7.,  9.,  21., 36., 27., 15., 9.,  20., 33., 24., 13., 6.,  13., 21., 15., 8.},
    };
    utils::Conv2dOptions options;
    options.transpose = true;
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Ohwi;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dTransposeNchwIhwo) {
    Tensor input = {{1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{1, 3, 3, 2}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 2, 5, 5},
        {0.,  1.,  3.,  3.,  2.,  3.,  8.,  15., 12., 7.,  9.,  21., 36., 27., 15., 9.,  20.,
         33., 24., 13., 6.,  13., 21., 15., 8.,  0.,  1.,  3.,  3.,  2.,  3.,  8.,  15., 12.,
         7.,  9.,  21., 36., 27., 15., 9.,  20., 33., 24., 13., 6.,  13., 21., 15., 8.},
    };
    utils::Conv2dOptions options;
    options.transpose = true;
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Ihwo;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dTransposeNhwcOihw) {
    Tensor input = {{1, 3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{2, 1, 3, 3}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 5, 5, 2},
        {0.,  0.,  1.,  1.,  3.,  3.,  3.,  3.,  2.,  2.,  3.,  3.,  8.,  8.,  15., 15., 12.,
         12., 7.,  7.,  9.,  9.,  21., 21., 36., 36., 27., 27., 15., 15., 9.,  9.,  20., 20.,
         33., 33., 24., 24., 13., 13., 6.,  6.,  13., 13., 21., 21., 15., 15., 8.,  8.},
    };
    utils::Conv2dOptions options;
    options.transpose = true;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Oihw;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dTransposeNhwcHwio) {
    Tensor input = {{1, 3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{3, 3, 1, 2}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 5, 5, 2},
        {0.,  0.,  1.,  1.,  3.,  3.,  3.,  3.,  2.,  2.,  3.,  3.,  8.,  8.,  15., 15., 12.,
         12., 7.,  7.,  9.,  9.,  21., 21., 36., 36., 27., 27., 15., 15., 9.,  9.,  20., 20.,
         33., 33., 24., 24., 13., 13., 6.,  6.,  13., 13., 21., 21., 15., 15., 8.,  8.},
    };
    utils::Conv2dOptions options;
    options.transpose = true;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Hwio;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dTransposeNhwcOhwi) {
    Tensor input = {{1, 3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{2, 3, 3, 1}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 5, 5, 2},
        {0.,  0.,  1.,  1.,  3.,  3.,  3.,  3.,  2.,  2.,  3.,  3.,  8.,  8.,  15., 15., 12.,
         12., 7.,  7.,  9.,  9.,  21., 21., 36., 36., 27., 27., 15., 15., 9.,  9.,  20., 20.,
         33., 33., 24., 24., 13., 13., 6.,  6.,  13., 13., 21., 21., 15., 15., 8.,  8.},
    };
    utils::Conv2dOptions options;
    options.transpose = true;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Ohwi;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dTransposeNhwcIhwo) {
    Tensor input = {{1, 3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{1, 3, 3, 2}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 5, 5, 2},
        {0.,  0.,  1.,  1.,  3.,  3.,  3.,  3.,  2.,  2.,  3.,  3.,  8.,  8.,  15., 15., 12.,
         12., 7.,  7.,  9.,  9.,  21., 21., 36., 36., 27., 27., 15., 15., 9.,  9.,  20., 20.,
         33., 33., 24., 24., 13., 13., 6.,  6.,  13., 13., 21., 21., 15., 15., 8.,  8.},
    };
    utils::Conv2dOptions options;
    options.transpose = true;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Ihwo;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dTransposeWithOutputShapeDefault) {
    Tensor input = {{1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{2, 1, 3, 3}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 2, 10, 8},
        {0.,  0., 1., 1., 3., 2., 2.,  0., 0.,  0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1.,
         3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5.,  0., 3.,  3., 7., 4., 9., 5., 5.,  0.,
         3.,  3., 7., 4., 9., 5., 5.,  0., 6.,  6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7.,
         15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8.,  0., 0.,  0., 0., 0., 0., 0., 0.,  0.,
         0.,  0., 1., 1., 3., 2., 2.,  0., 0.,  0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1.,
         3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5.,  0., 3.,  3., 7., 4., 9., 5., 5.,  0.,
         3.,  3., 7., 4., 9., 5., 5.,  0., 6.,  6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7.,
         15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8.,  0., 0.,  0., 0., 0., 0., 0., 0.,  0.},
    };
    utils::Conv2dOptions options;
    options.strides = {3, 2};
    options.outputSizes = {10, 8};
    options.transpose = true;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dTransposeWithOutputShapeNchwHwio) {
    Tensor input = {{1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{3, 3, 1, 2}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 2, 10, 8},
        {0.,  0., 1., 1., 3., 2., 2.,  0., 0.,  0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1.,
         3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5.,  0., 3.,  3., 7., 4., 9., 5., 5.,  0.,
         3.,  3., 7., 4., 9., 5., 5.,  0., 6.,  6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7.,
         15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8.,  0., 0.,  0., 0., 0., 0., 0., 0.,  0.,
         0.,  0., 1., 1., 3., 2., 2.,  0., 0.,  0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1.,
         3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5.,  0., 3.,  3., 7., 4., 9., 5., 5.,  0.,
         3.,  3., 7., 4., 9., 5., 5.,  0., 6.,  6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7.,
         15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8.,  0., 0.,  0., 0., 0., 0., 0., 0.,  0.},
    };
    utils::Conv2dOptions options;
    options.strides = {3, 2};
    options.outputSizes = {10, 8};
    options.transpose = true;
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Hwio;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dTransposeWithOutputShapeNchwOhwi) {
    Tensor input = {{1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{2, 3, 3, 1}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 2, 10, 8},
        {0.,  0., 1., 1., 3., 2., 2.,  0., 0.,  0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1.,
         3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5.,  0., 3.,  3., 7., 4., 9., 5., 5.,  0.,
         3.,  3., 7., 4., 9., 5., 5.,  0., 6.,  6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7.,
         15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8.,  0., 0.,  0., 0., 0., 0., 0., 0.,  0.,
         0.,  0., 1., 1., 3., 2., 2.,  0., 0.,  0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1.,
         3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5.,  0., 3.,  3., 7., 4., 9., 5., 5.,  0.,
         3.,  3., 7., 4., 9., 5., 5.,  0., 6.,  6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7.,
         15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8.,  0., 0.,  0., 0., 0., 0., 0., 0.,  0.},
    };
    utils::Conv2dOptions options;
    options.strides = {3, 2};
    options.outputSizes = {10, 8};
    options.transpose = true;
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Ohwi;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dTransposeWithOutputShapeNchwIhwo) {
    Tensor input = {{1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{1, 3, 3, 2}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 2, 10, 8},
        {0.,  0., 1., 1., 3., 2., 2.,  0., 0.,  0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1.,
         3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5.,  0., 3.,  3., 7., 4., 9., 5., 5.,  0.,
         3.,  3., 7., 4., 9., 5., 5.,  0., 6.,  6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7.,
         15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8.,  0., 0.,  0., 0., 0., 0., 0., 0.,  0.,
         0.,  0., 1., 1., 3., 2., 2.,  0., 0.,  0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1.,
         3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5.,  0., 3.,  3., 7., 4., 9., 5., 5.,  0.,
         3.,  3., 7., 4., 9., 5., 5.,  0., 6.,  6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7.,
         15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8.,  0., 0.,  0., 0., 0., 0., 0., 0.,  0.},
    };
    utils::Conv2dOptions options;
    options.strides = {3, 2};
    options.outputSizes = {10, 8};
    options.transpose = true;
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Ihwo;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dTransposeWithOutputShapeNhwcOihw) {
    Tensor input = {{1, 3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{2, 1, 3, 3}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 2, 10, 8},
        {0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 1., 1., 1.,
         1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2.,
         0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7.,
         7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5.,
         5., 5., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0., 6., 6., 6.,
         6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15,
         8., 8., 8., 8., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
    };
    utils::Conv2dOptions options;
    options.strides = {3, 2};
    options.outputSizes = {10, 8};
    options.transpose = true;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Oihw;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dTransposeWithOutputShapeNhwcHwio) {
    Tensor input = {{1, 3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{3, 3, 1, 2}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 2, 10, 8},
        {0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 1., 1., 1.,
         1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2.,
         0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7.,
         7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5.,
         5., 5., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0., 6., 6., 6.,
         6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15,
         8., 8., 8., 8., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
    };
    utils::Conv2dOptions options;
    options.strides = {3, 2};
    options.outputSizes = {10, 8};
    options.transpose = true;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Hwio;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dTransposeWithOutputShapeNhwcOhwi) {
    Tensor input = {{1, 3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{2, 3, 3, 1}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 2, 10, 8},
        {0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 1., 1., 1.,
         1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2.,
         0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7.,
         7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5.,
         5., 5., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0., 6., 6., 6.,
         6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15,
         8., 8., 8., 8., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
    };
    utils::Conv2dOptions options;
    options.strides = {3, 2};
    options.outputSizes = {10, 8};
    options.transpose = true;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Ohwi;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dTransposeWithOutputShapeNhwcIhwo) {
    Tensor input = {{1, 3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{1, 3, 3, 2}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 2, 10, 8},
        {0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 1., 1., 1.,
         1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2.,
         0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7.,
         7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5.,
         5., 5., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0., 6., 6., 6.,
         6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15,
         8., 8., 8., 8., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
    };
    utils::Conv2dOptions options;
    options.strides = {3, 2};
    options.outputSizes = {10, 8};
    options.transpose = true;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Ihwo;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dTransposeWithOutputPaddingDefault) {
    Tensor input = {{1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{2, 1, 3, 3}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 2, 10, 8},
        {0.,  0., 1., 1., 3., 2., 2.,  0., 0.,  0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1.,
         3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5.,  0., 3.,  3., 7., 4., 9., 5., 5.,  0.,
         3.,  3., 7., 4., 9., 5., 5.,  0., 6.,  6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7.,
         15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8.,  0., 0.,  0., 0., 0., 0., 0., 0.,  0.,
         0.,  0., 1., 1., 3., 2., 2.,  0., 0.,  0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1.,
         3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5.,  0., 3.,  3., 7., 4., 9., 5., 5.,  0.,
         3.,  3., 7., 4., 9., 5., 5.,  0., 6.,  6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7.,
         15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8.,  0., 0.,  0., 0., 0., 0., 0., 0.,  0.},
    };
    utils::Conv2dOptions options;
    options.strides = {3, 2};
    options.outputPadding = {1, 1};
    options.transpose = true;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dTransposeWithOutputPaddingNchwHwio) {
    Tensor input = {{1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{3, 3, 1, 2}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 2, 10, 8},
        {0.,  0., 1., 1., 3., 2., 2.,  0., 0.,  0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1.,
         3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5.,  0., 3.,  3., 7., 4., 9., 5., 5.,  0.,
         3.,  3., 7., 4., 9., 5., 5.,  0., 6.,  6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7.,
         15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8.,  0., 0.,  0., 0., 0., 0., 0., 0.,  0.,
         0.,  0., 1., 1., 3., 2., 2.,  0., 0.,  0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1.,
         3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5.,  0., 3.,  3., 7., 4., 9., 5., 5.,  0.,
         3.,  3., 7., 4., 9., 5., 5.,  0., 6.,  6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7.,
         15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8.,  0., 0.,  0., 0., 0., 0., 0., 0.,  0.},
    };
    utils::Conv2dOptions options;
    options.strides = {3, 2};
    options.outputPadding = {1, 1};
    options.transpose = true;
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Hwio;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dTransposeWithOutputPaddingNchwOhwi) {
    Tensor input = {{1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{2, 3, 3, 1}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 2, 10, 8},
        {0.,  0., 1., 1., 3., 2., 2.,  0., 0.,  0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1.,
         3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5.,  0., 3.,  3., 7., 4., 9., 5., 5.,  0.,
         3.,  3., 7., 4., 9., 5., 5.,  0., 6.,  6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7.,
         15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8.,  0., 0.,  0., 0., 0., 0., 0., 0.,  0.,
         0.,  0., 1., 1., 3., 2., 2.,  0., 0.,  0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1.,
         3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5.,  0., 3.,  3., 7., 4., 9., 5., 5.,  0.,
         3.,  3., 7., 4., 9., 5., 5.,  0., 6.,  6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7.,
         15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8.,  0., 0.,  0., 0., 0., 0., 0., 0.,  0.},
    };
    utils::Conv2dOptions options;
    options.strides = {3, 2};
    options.outputPadding = {1, 1};
    options.transpose = true;
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Ohwi;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dTransposeWithOutputPaddingNchwIhwo) {
    Tensor input = {{1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{1, 3, 3, 2}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 2, 10, 8},
        {0.,  0., 1., 1., 3., 2., 2.,  0., 0.,  0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1.,
         3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5.,  0., 3.,  3., 7., 4., 9., 5., 5.,  0.,
         3.,  3., 7., 4., 9., 5., 5.,  0., 6.,  6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7.,
         15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8.,  0., 0.,  0., 0., 0., 0., 0., 0.,  0.,
         0.,  0., 1., 1., 3., 2., 2.,  0., 0.,  0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1.,
         3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5.,  0., 3.,  3., 7., 4., 9., 5., 5.,  0.,
         3.,  3., 7., 4., 9., 5., 5.,  0., 6.,  6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7.,
         15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8.,  0., 0.,  0., 0., 0., 0., 0., 0.,  0.},
    };
    utils::Conv2dOptions options;
    options.strides = {3, 2};
    options.outputPadding = {1, 1};
    options.transpose = true;
    options.inputLayout = ml::InputOperandLayout::Nchw;
    options.filterLayout = ml::FilterOperandLayout::Ihwo;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dTransposeWithOutputPaddingNhwcOihw) {
    Tensor input = {{1, 3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{2, 1, 3, 3}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 2, 10, 8},
        {0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 1., 1., 1.,
         1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2.,
         0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7.,
         7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5.,
         5., 5., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0., 6., 6., 6.,
         6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15,
         8., 8., 8., 8., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
    };
    utils::Conv2dOptions options;
    options.strides = {3, 2};
    options.outputPadding = {1, 1};
    options.transpose = true;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Oihw;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dTransposeWithOutputPaddingNhwcHwio) {
    Tensor input = {{1, 3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{3, 3, 1, 2}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 2, 10, 8},
        {0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 1., 1., 1.,
         1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2.,
         0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7.,
         7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5.,
         5., 5., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0., 6., 6., 6.,
         6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15,
         8., 8., 8., 8., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
    };
    utils::Conv2dOptions options;
    options.strides = {3, 2};
    options.outputPadding = {1, 1};
    options.transpose = true;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Hwio;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dTransposeWithOutputPaddingNhwcOhwi) {
    Tensor input = {{1, 3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{2, 3, 3, 1}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 2, 10, 8},
        {0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 1., 1., 1.,
         1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2.,
         0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7.,
         7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5.,
         5., 5., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0., 6., 6., 6.,
         6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15,
         8., 8., 8., 8., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
    };
    utils::Conv2dOptions options;
    options.strides = {3, 2};
    options.outputPadding = {1, 1};
    options.transpose = true;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Ohwi;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dTransposeWithOutputPaddingNhwcIhwo) {
    Tensor input = {{1, 3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{1, 3, 3, 2}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 2, 10, 8},
        {0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 1., 1., 1.,
         1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2.,
         0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7.,
         7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5.,
         5., 5., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0., 6., 6., 6.,
         6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15,
         8., 8., 8., 8., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
    };
    utils::Conv2dOptions options;
    options.strides = {3, 2};
    options.outputPadding = {1, 1};
    options.transpose = true;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Ihwo;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dTransposeWithAutoPadSameUpperDefault) {
    Tensor input = {{1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{2, 1, 3, 3}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 2, 6, 6},
        {0., 0., 1., 1., 3., 2., 0., 0., 1.,  1.,  3.,  2.,  3., 3., 8.,  5., 12., 7.,
         3., 3., 7., 4., 9., 5., 9., 9., 20., 11., 24., 13., 6., 6., 13., 7., 15., 8.,
         0., 0., 1., 1., 3., 2., 0., 0., 1.,  1.,  3.,  2.,  3., 3., 8.,  5., 12., 7.,
         3., 3., 7., 4., 9., 5., 9., 9., 20., 11., 24., 13., 6., 6., 13., 7., 15., 8.},
    };
    utils::Conv2dOptions options;
    options.strides = {2, 2};
    options.autoPad = ml::AutoPad::SameUpper;
    options.transpose = true;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dTransposeWithAutoPadExplicitNhwcIhwo) {
    Tensor input = {{1, 3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{1, 3, 3, 2}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 6, 6, 2},
        {0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 0., 0., 0., 0., 1., 1.,
         1., 1., 3., 3., 2., 2., 3., 3., 3., 3., 8., 8., 5., 5., 12, 12, 7., 7.,
         3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 9., 9., 9., 9., 20, 20,
         11, 11, 24, 24, 13, 13, 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8.},
    };
    utils::Conv2dOptions options;
    options.strides = {2, 2};
    options.padding = {0, 1, 0, 1};
    options.autoPad = ml::AutoPad::Explicit;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Ihwo;
    options.transpose = true;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dTransposeWithAutoPadSameLowerNhwcIhwo) {
    Tensor input = {{1, 3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{1, 3, 3, 2}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 6, 6, 2},
        {0.,  0.,  1.,  1.,  1.,  1.,  3.,  3.,  2.,  2.,  2.,  2.,  3.,  3.,  8.,  8.,  5., 5.,
         12., 12., 7.,  7.,  7.,  7.,  3.,  3.,  7.,  7.,  4.,  4.,  9.,  9.,  5.,  5.,  5., 5.,
         9.,  9.,  20., 20., 11., 11., 24., 24., 13., 13., 13., 13., 6.,  6.,  13., 13., 7., 7.,
         15., 15., 8.,  8.,  8.,  8.,  6.,  6.,  13., 13., 7.,  7.,  15., 15., 8.,  8.,  8., 8.},
    };
    utils::Conv2dOptions options;
    options.strides = {2, 2};
    options.padding = {0, 1, 0, 1};
    options.autoPad = ml::AutoPad::SameLower;
    options.inputLayout = ml::InputOperandLayout::Nhwc;
    options.filterLayout = ml::FilterOperandLayout::Ihwo;
    options.transpose = true;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dTransposeWithOutputSizeIgnoredOutputPadding) {
    Tensor input = {{1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{2, 1, 3, 3}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 2, 10, 8},
        {0.,  0., 1., 1., 3., 2., 2.,  0., 0.,  0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1.,
         3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5.,  0., 3.,  3., 7., 4., 9., 5., 5.,  0.,
         3.,  3., 7., 4., 9., 5., 5.,  0., 6.,  6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7.,
         15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8.,  0., 0.,  0., 0., 0., 0., 0., 0.,  0.,
         0.,  0., 1., 1., 3., 2., 2.,  0., 0.,  0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1.,
         3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5.,  0., 3.,  3., 7., 4., 9., 5., 5.,  0.,
         3.,  3., 7., 4., 9., 5., 5.,  0., 6.,  6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7.,
         15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8.,  0., 0.,  0., 0., 0., 0., 0., 0.,  0.},
    };
    utils::Conv2dOptions options;
    options.strides = {3, 2};
    options.outputPadding = {1, 1};
    options.outputSizes = {10, 8};
    options.transpose = true;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dWithTransposeFalse) {
    Tensor input = {
        {1, 1, 5, 5},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
    };
    Tensor filter = {{1, 1, 3, 3}, std::vector<float>(9, 1)};
    Tensor expected = {
        {1, 1, 3, 3},
        {54., 63., 72., 99., 108., 117., 144., 153., 162.},
    };
    utils::Conv2dOptions options;
    options.transpose = false;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dWithTransposeFalseIgnoredOutputPadding) {
    Tensor input = {
        {1, 1, 5, 5},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
    };
    Tensor filter = {{1, 1, 3, 3}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 1, 3, 3}, {54., 63., 72., 99., 108., 117., 144., 153., 162.}};
    utils::Conv2dOptions options;
    options.outputPadding = {1, 1};
    options.transpose = false;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dWithTransposeFalseIgnoredOutputShape) {
    Tensor input = {
        {1, 1, 5, 5},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
    };
    Tensor filter = {{1, 1, 3, 3}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 1, 3, 3}, {54., 63., 72., 99., 108., 117., 144., 153., 162.}};
    utils::Conv2dOptions options;
    options.outputSizes = {1, 9};
    options.transpose = false;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dWithTransposeFalseIgnoredOutputPaddingAndOutputShape) {
    Tensor input = {
        {1, 1, 5, 5},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
    };
    Tensor filter = {{1, 1, 3, 3}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 1, 3, 3}, {54., 63., 72., 99., 108., 117., 144., 153., 162.}};
    utils::Conv2dOptions options;
    options.outputPadding = {1, 1};
    options.outputSizes = {1, 9};
    options.transpose = false;
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dWithTransposeDefaultIgnoredOutputPadding) {
    Tensor input = {
        {1, 1, 5, 5},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
    };
    Tensor filter = {{1, 1, 3, 3}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 1, 3, 3}, {54., 63., 72., 99., 108., 117., 144., 153., 162.}};
    utils::Conv2dOptions options;
    options.outputPadding = {1, 1};
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dWithTransposeDefaultIgnoredOutputShape) {
    Tensor input = {
        {1, 1, 5, 5},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
    };
    Tensor filter = {{1, 1, 3, 3}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 1, 3, 3}, {54., 63., 72., 99., 108., 117., 144., 153., 162.}};
    utils::Conv2dOptions options;
    options.outputSizes = {1, 9};
    CheckConv2d(input, filter, expected, options);
}

TEST_F(Conv2dTests, Conv2dWithTransposeDefaultIgnoredOutputPaddingAndOutputShape) {
    Tensor input = {
        {1, 1, 5, 5},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
    };
    Tensor filter = {{1, 1, 3, 3}, std::vector<float>(9, 1)};
    Tensor expected = {{1, 1, 3, 3}, {54., 63., 72., 99., 108., 117., 144., 153., 162.}};
    utils::Conv2dOptions options;
    options.outputPadding = {1, 1};
    options.outputSizes = {1, 9};
    CheckConv2d(input, filter, expected, options);
}
