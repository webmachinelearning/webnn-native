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

class ConvTranspose2dTests : public WebnnTest {
    void SetUp() override {
        builder = wnn::CreateGraphBuilder(GetContext());
    }

  protected:
    struct Tensor {
        std::vector<int32_t> shape;
        std::vector<float> value;
    };

    void CheckConvTranspose2d(const Tensor& input,
                              const Tensor& filter,
                              const Tensor& expected,
                              utils::ConvTranspose2dOptions options = {},
                              const Tensor& bias = {},
                              utils::FusedActivation activation = utils::FusedActivation::NONE,
                              bool fusion = false,
                              void* activationOptions = nullptr) {
        const wnn::Operand x = utils::BuildInput(builder, "input", input.shape);
        const wnn::Operand w = utils::BuildConstant(builder, filter.shape, filter.value.data(),
                                                    filter.value.size() * sizeof(float));

        wnn::Operand b;
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
        wnn::Operand y = builder.ConvTranspose2d(x, w, options.AsPtr());

        if (!fusion) {
            if (!bias.value.empty()) {
                if (options.inputLayout == wnn::InputOperandLayout::Nchw) {
                    std::vector<int32_t> newShape = std::vector<int32_t>({1, -1, 1, 1});
                    b = builder.Reshape(b, newShape.data(), newShape.size());
                }
                y = builder.Add(y, b);
            }
            if (activation != utils::FusedActivation::NONE) {
                y = utils::CreateActivationOperand(builder, y, activation, activationOptions);
            }
        }

        const wnn::Graph graph = utils::Build(builder, {{"output", y}});
        ASSERT_TRUE(graph);
        std::vector<float> result(utils::SizeOfShape(expected.shape));
        utils::Compute(graph, {{"input", input.value}}, {{"output", result}});
        EXPECT_TRUE(utils::CheckValue(result, expected.value));
    }

    wnn::GraphBuilder builder;
};

TEST_F(ConvTranspose2dTests, Conv2dTransposeDefault) {
    Tensor input = {{1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{1, 2, 3, 3}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 2, 5, 5},
        {0.,  1.,  3.,  3.,  2.,  3.,  8.,  15., 12., 7.,  9.,  21., 36., 27., 15., 9.,  20.,
         33., 24., 13., 6.,  13., 21., 15., 8.,  0.,  1.,  3.,  3.,  2.,  3.,  8.,  15., 12.,
         7.,  9.,  21., 36., 27., 15., 9.,  20., 33., 24., 13., 6.,  13., 21., 15., 8.},
    };
    utils::ConvTranspose2dOptions options;
    CheckConvTranspose2d(input, filter, expected, options);
}

TEST_F(ConvTranspose2dTests, Conv2dTransposeNchwHwoi) {
    Tensor input = {{1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{3, 3, 2, 1}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 2, 5, 5},
        {0.,  1.,  3.,  3.,  2.,  3.,  8.,  15., 12., 7.,  9.,  21., 36., 27., 15., 9.,  20.,
         33., 24., 13., 6.,  13., 21., 15., 8.,  0.,  1.,  3.,  3.,  2.,  3.,  8.,  15., 12.,
         7.,  9.,  21., 36., 27., 15., 9.,  20., 33., 24., 13., 6.,  13., 21., 15., 8.},
    };
    utils::ConvTranspose2dOptions options;
    options.inputLayout = wnn::InputOperandLayout::Nchw;
    options.filterLayout = wnn::ConvTranspose2dFilterOperandLayout::Hwoi;
    CheckConvTranspose2d(input, filter, expected, options);
}

TEST_F(ConvTranspose2dTests, Conv2dTransposeNchwOhwi) {
    Tensor input = {{1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{2, 3, 3, 1}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 2, 5, 5},
        {0.,  1.,  3.,  3.,  2.,  3.,  8.,  15., 12., 7.,  9.,  21., 36., 27., 15., 9.,  20.,
         33., 24., 13., 6.,  13., 21., 15., 8.,  0.,  1.,  3.,  3.,  2.,  3.,  8.,  15., 12.,
         7.,  9.,  21., 36., 27., 15., 9.,  20., 33., 24., 13., 6.,  13., 21., 15., 8.},
    };
    utils::ConvTranspose2dOptions options;
    options.inputLayout = wnn::InputOperandLayout::Nchw;
    options.filterLayout = wnn::ConvTranspose2dFilterOperandLayout::Ohwi;
    CheckConvTranspose2d(input, filter, expected, options);
}

TEST_F(ConvTranspose2dTests, Conv2dTransposeNhwcIohw) {
    Tensor input = {{1, 3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{1, 2, 3, 3}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 5, 5, 2},
        {0.,  0.,  1.,  1.,  3.,  3.,  3.,  3.,  2.,  2.,  3.,  3.,  8.,  8.,  15., 15., 12.,
         12., 7.,  7.,  9.,  9.,  21., 21., 36., 36., 27., 27., 15., 15., 9.,  9.,  20., 20.,
         33., 33., 24., 24., 13., 13., 6.,  6.,  13., 13., 21., 21., 15., 15., 8.,  8.},
    };
    utils::ConvTranspose2dOptions options;
    options.inputLayout = wnn::InputOperandLayout::Nhwc;
    options.filterLayout = wnn::ConvTranspose2dFilterOperandLayout::Iohw;
    CheckConvTranspose2d(input, filter, expected, options);
}

TEST_F(ConvTranspose2dTests, Conv2dTransposeNhwcHwoi) {
    Tensor input = {{1, 3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{3, 3, 2, 1}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 5, 5, 2},
        {0.,  0.,  1.,  1.,  3.,  3.,  3.,  3.,  2.,  2.,  3.,  3.,  8.,  8.,  15., 15., 12.,
         12., 7.,  7.,  9.,  9.,  21., 21., 36., 36., 27., 27., 15., 15., 9.,  9.,  20., 20.,
         33., 33., 24., 24., 13., 13., 6.,  6.,  13., 13., 21., 21., 15., 15., 8.,  8.},
    };
    utils::ConvTranspose2dOptions options;
    options.inputLayout = wnn::InputOperandLayout::Nhwc;
    options.filterLayout = wnn::ConvTranspose2dFilterOperandLayout::Hwoi;
    CheckConvTranspose2d(input, filter, expected, options);
}

TEST_F(ConvTranspose2dTests, Conv2dTransposeNhwcOhwi) {
    Tensor input = {{1, 3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{2, 3, 3, 1}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 5, 5, 2},
        {0.,  0.,  1.,  1.,  3.,  3.,  3.,  3.,  2.,  2.,  3.,  3.,  8.,  8.,  15., 15., 12.,
         12., 7.,  7.,  9.,  9.,  21., 21., 36., 36., 27., 27., 15., 15., 9.,  9.,  20., 20.,
         33., 33., 24., 24., 13., 13., 6.,  6.,  13., 13., 21., 21., 15., 15., 8.,  8.},
    };
    utils::ConvTranspose2dOptions options;
    options.inputLayout = wnn::InputOperandLayout::Nhwc;
    options.filterLayout = wnn::ConvTranspose2dFilterOperandLayout::Ohwi;
    CheckConvTranspose2d(input, filter, expected, options);
}

TEST_F(ConvTranspose2dTests, Conv2dTransposeWithOutputShapeDefault) {
    Tensor input = {{1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{1, 2, 3, 3}, std::vector<float>(18, 1)};
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
    utils::ConvTranspose2dOptions options;
    options.strides = {3, 2};
    options.outputSizes = {10, 8};
    CheckConvTranspose2d(input, filter, expected, options);
}

TEST_F(ConvTranspose2dTests, Conv2dTransposeWithOutputShapeNchwHwoi) {
    Tensor input = {{1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{3, 3, 2, 1}, std::vector<float>(18, 1)};
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
    utils::ConvTranspose2dOptions options;
    options.strides = {3, 2};
    options.outputSizes = {10, 8};
    options.inputLayout = wnn::InputOperandLayout::Nchw;
    options.filterLayout = wnn::ConvTranspose2dFilterOperandLayout::Hwoi;
    CheckConvTranspose2d(input, filter, expected, options);
}

TEST_F(ConvTranspose2dTests, Conv2dTransposeWithOutputShapeNchwOhwi) {
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
    utils::ConvTranspose2dOptions options;
    options.strides = {3, 2};
    options.outputSizes = {10, 8};
    options.inputLayout = wnn::InputOperandLayout::Nchw;
    options.filterLayout = wnn::ConvTranspose2dFilterOperandLayout::Ohwi;
    CheckConvTranspose2d(input, filter, expected, options);
}

TEST_F(ConvTranspose2dTests, Conv2dTransposeWithOutputShapeNhwcIohw) {
    Tensor input = {{1, 3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{1, 2, 3, 3}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 10, 8, 2},
        {0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 1., 1., 1.,
         1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2.,
         0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7.,
         7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5.,
         5., 5., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0., 6., 6., 6.,
         6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15,
         8., 8., 8., 8., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
    };
    utils::ConvTranspose2dOptions options;
    options.strides = {3, 2};
    options.outputSizes = {10, 8};
    options.inputLayout = wnn::InputOperandLayout::Nhwc;
    options.filterLayout = wnn::ConvTranspose2dFilterOperandLayout::Iohw;
    CheckConvTranspose2d(input, filter, expected, options);
}

TEST_F(ConvTranspose2dTests, Conv2dTransposeWithOutputShapeNhwcHwoi) {
    Tensor input = {{1, 3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{3, 3, 2, 1}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 10, 8, 2},
        {0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 1., 1., 1.,
         1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2.,
         0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7.,
         7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5.,
         5., 5., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0., 6., 6., 6.,
         6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15,
         8., 8., 8., 8., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
    };
    utils::ConvTranspose2dOptions options;
    options.strides = {3, 2};
    options.outputSizes = {10, 8};
    options.inputLayout = wnn::InputOperandLayout::Nhwc;
    options.filterLayout = wnn::ConvTranspose2dFilterOperandLayout::Hwoi;
    CheckConvTranspose2d(input, filter, expected, options);
}

TEST_F(ConvTranspose2dTests, Conv2dTransposeWithOutputShapeNhwcOhwi) {
    Tensor input = {{1, 3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{2, 3, 3, 1}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 10, 8, 2},
        {0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 1., 1., 1.,
         1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2.,
         0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7.,
         7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5.,
         5., 5., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0., 6., 6., 6.,
         6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15,
         8., 8., 8., 8., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
    };
    utils::ConvTranspose2dOptions options;
    options.strides = {3, 2};
    options.outputSizes = {10, 8};
    options.inputLayout = wnn::InputOperandLayout::Nhwc;
    options.filterLayout = wnn::ConvTranspose2dFilterOperandLayout::Ohwi;
    CheckConvTranspose2d(input, filter, expected, options);
}

TEST_F(ConvTranspose2dTests, Conv2dTransposeWithOutputPaddingDefault) {
    Tensor input = {{1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{1, 2, 3, 3}, std::vector<float>(18, 1)};
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
    utils::ConvTranspose2dOptions options;
    options.strides = {3, 2};
    options.outputPadding = {1, 1};
    CheckConvTranspose2d(input, filter, expected, options);
}

TEST_F(ConvTranspose2dTests, Conv2dTransposeWithOutputPaddingNchwHwoi) {
    Tensor input = {{1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{3, 3, 2, 1}, std::vector<float>(18, 1)};
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
    utils::ConvTranspose2dOptions options;
    options.strides = {3, 2};
    options.outputPadding = {1, 1};
    options.inputLayout = wnn::InputOperandLayout::Nchw;
    options.filterLayout = wnn::ConvTranspose2dFilterOperandLayout::Hwoi;
    CheckConvTranspose2d(input, filter, expected, options);
}

TEST_F(ConvTranspose2dTests, Conv2dTransposeWithOutputPaddingNchwOhwi) {
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
    utils::ConvTranspose2dOptions options;
    options.strides = {3, 2};
    options.outputPadding = {1, 1};

    options.inputLayout = wnn::InputOperandLayout::Nchw;
    options.filterLayout = wnn::ConvTranspose2dFilterOperandLayout::Ohwi;
    CheckConvTranspose2d(input, filter, expected, options);
}

TEST_F(ConvTranspose2dTests, Conv2dTransposeWithOutputPaddingNhwcIohw) {
    Tensor input = {{1, 3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{1, 2, 3, 3}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 10, 8, 2},
        {0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 1., 1., 1.,
         1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2.,
         0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7.,
         7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5.,
         5., 5., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0., 6., 6., 6.,
         6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15,
         8., 8., 8., 8., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
    };
    utils::ConvTranspose2dOptions options;
    options.strides = {3, 2};
    options.outputPadding = {1, 1};
    options.inputLayout = wnn::InputOperandLayout::Nhwc;
    options.filterLayout = wnn::ConvTranspose2dFilterOperandLayout::Iohw;
    CheckConvTranspose2d(input, filter, expected, options);
}

TEST_F(ConvTranspose2dTests, Conv2dTransposeWithOutputPaddingNhwcHwoi) {
    Tensor input = {{1, 3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{3, 3, 2, 1}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 10, 8, 2},
        {0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 1., 1., 1.,
         1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2.,
         0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7.,
         7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5.,
         5., 5., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0., 6., 6., 6.,
         6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15,
         8., 8., 8., 8., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
    };
    utils::ConvTranspose2dOptions options;
    options.strides = {3, 2};
    options.outputPadding = {1, 1};
    options.inputLayout = wnn::InputOperandLayout::Nhwc;
    options.filterLayout = wnn::ConvTranspose2dFilterOperandLayout::Hwoi;
    CheckConvTranspose2d(input, filter, expected, options);
}

TEST_F(ConvTranspose2dTests, Conv2dTransposeWithOutputPaddingNhwcOhwi) {
    Tensor input = {{1, 3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{2, 3, 3, 1}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 10, 8, 2},
        {0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 1., 1., 1.,
         1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2.,
         0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7.,
         7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5.,
         5., 5., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0., 6., 6., 6.,
         6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15,
         8., 8., 8., 8., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
    };
    utils::ConvTranspose2dOptions options;
    options.strides = {3, 2};
    options.outputPadding = {1, 1};
    options.inputLayout = wnn::InputOperandLayout::Nhwc;
    options.filterLayout = wnn::ConvTranspose2dFilterOperandLayout::Ohwi;
    CheckConvTranspose2d(input, filter, expected, options);
}

TEST_F(ConvTranspose2dTests, Conv2dTransposeWithAutoPadSameUpperDefault) {
    Tensor input = {{1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{1, 2, 3, 3}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 2, 6, 6},
        {0., 0., 1., 1., 3., 2., 0., 0., 1.,  1.,  3.,  2.,  3., 3., 8.,  5., 12., 7.,
         3., 3., 7., 4., 9., 5., 9., 9., 20., 11., 24., 13., 6., 6., 13., 7., 15., 8.,
         0., 0., 1., 1., 3., 2., 0., 0., 1.,  1.,  3.,  2.,  3., 3., 8.,  5., 12., 7.,
         3., 3., 7., 4., 9., 5., 9., 9., 20., 11., 24., 13., 6., 6., 13., 7., 15., 8.},
    };
    utils::ConvTranspose2dOptions options;
    options.strides = {2, 2};
    options.autoPad = wnn::AutoPad::SameUpper;
    CheckConvTranspose2d(input, filter, expected, options);
}

TEST_F(ConvTranspose2dTests, Conv2dTransposeWithAutoPadExplicitDefault) {
    Tensor input = {{1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{1, 2, 3, 3}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 6, 6, 2},
        {0, 0, 1,  1,  3,  2,  0, 0, 1,  1, 3,  2, 3, 3, 8,  5,  12, 7,  3, 3, 7,  4, 9,  5,
         9, 9, 20, 11, 24, 13, 6, 6, 13, 7, 15, 8, 0, 0, 1,  1,  3,  2,  0, 0, 1,  1, 3,  2,
         3, 3, 8,  5,  12, 7,  3, 3, 7,  4, 9,  5, 9, 9, 20, 11, 24, 13, 6, 6, 13, 7, 15, 8},
    };
    utils::ConvTranspose2dOptions options;
    options.strides = {2, 2};
    options.padding = {0, 1, 0, 1};
    options.autoPad = wnn::AutoPad::Explicit;
    CheckConvTranspose2d(input, filter, expected, options);
}

TEST_F(ConvTranspose2dTests, Conv2dTransposeWithAutoPadSameLowerDefault) {
    Tensor input = {{1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    Tensor filter = {{1, 2, 3, 3}, std::vector<float>(18, 1)};
    Tensor expected = {
        {1, 6, 6, 2},
        {0, 1,  1, 3,  2, 2, 3, 8,  5,  12, 7,  7,  3, 7,  4, 9,  5, 5, 9, 20, 11, 24, 13, 13,
         6, 13, 7, 15, 8, 8, 6, 13, 7,  15, 8,  8,  0, 1,  1, 3,  2, 2, 3, 8,  5,  12, 7,  7,
         3, 7,  4, 9,  5, 5, 9, 20, 11, 24, 13, 13, 6, 13, 7, 15, 8, 8, 6, 13, 7,  15, 8,  8},
    };
    utils::ConvTranspose2dOptions options;
    options.strides = {2, 2};
    options.autoPad = wnn::AutoPad::SameLower;
    CheckConvTranspose2d(input, filter, expected, options);
}
