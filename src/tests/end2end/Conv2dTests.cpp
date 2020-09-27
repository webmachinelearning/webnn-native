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

class Conv2dTests : public WebnnTest {};

TEST_F(Conv2dTests, Conv2dWithPadding) {
    const webnn::ModelBuilder builder = GetContext().CreateModelBuilder();
    const webnn::Operand input = utils::BuildInput(builder, "input", {1, 1, 5, 5});
    const std::vector<float> filterData(9, 1);
    const webnn::Operand filter = utils::BuildConstant(builder, {1, 1, 3, 3}, filterData.data(),
                                                       filterData.size() * sizeof(float));
    utils::Conv2dOptions options;
    options.padding = {1, 1, 1, 1};
    const webnn::Operand output = builder.Conv2d(input, filter, options.AsPtr());
    const webnn::Model model = utils::CreateModel(builder, {{"output", output}});
    const webnn::Compilation compiledModel = utils::AwaitCompile(model);
    const std::vector<float> inputData = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                          13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    const webnn::Result result =
        utils::AwaitCompute(compiledModel,
                            {{"input", {inputData.data(), inputData.size() * sizeof(float)}}})
            .Get("output");
    EXPECT_TRUE(utils::CheckShape(result, {1, 1, 5, 5}));
    const std::vector<float> expectedValue({12.,  21.,  27., 33.,  24.,  33.,  54., 63.,  72.,
                                            51.,  63.,  99., 108., 117., 81.,  93., 144., 153.,
                                            162., 111., 72., 111., 117., 123., 84.});
    EXPECT_TRUE(utils::CheckValue(result, expectedValue));
}

TEST_F(Conv2dTests, Conv2dWithoutPadding) {
    const webnn::ModelBuilder builder = GetContext().CreateModelBuilder();
    const webnn::Operand input = utils::BuildInput(builder, "input", {1, 1, 5, 5});
    const std::vector<float> filterData(9, 1);
    const webnn::Operand filter = utils::BuildConstant(builder, {1, 1, 3, 3}, filterData.data(),
                                                       filterData.size() * sizeof(float));
    const webnn::Operand output = builder.Conv2d(input, filter);
    const webnn::Model model = utils::CreateModel(builder, {{"output", output}});
    const webnn::Compilation compiledModel = utils::AwaitCompile(model);
    const std::vector<float> inputData = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                          13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    const webnn::Result result =
        utils::AwaitCompute(compiledModel,
                            {{"input", {inputData.data(), inputData.size() * sizeof(float)}}})
            .Get("output");
    EXPECT_TRUE(utils::CheckShape(result, {1, 1, 3, 3}));
    const std::vector<float> expectedValue({54., 63., 72., 99., 108., 117., 144., 153., 162.});
    EXPECT_TRUE(utils::CheckValue(result, expectedValue));
}

TEST_F(Conv2dTests, Conv2dWithStrides2AndPadding) {
    const webnn::ModelBuilder builder = GetContext().CreateModelBuilder();
    const webnn::Operand input = utils::BuildInput(builder, "input", {1, 1, 7, 5});
    const std::vector<float> filterData(9, 1);
    const webnn::Operand filter = utils::BuildConstant(builder, {1, 1, 3, 3}, filterData.data(),
                                                       filterData.size() * sizeof(float));
    utils::Conv2dOptions options;
    options.padding = {1, 1, 1, 1};
    options.strides = {2, 2};
    const webnn::Operand output = builder.Conv2d(input, filter, options.AsPtr());
    const webnn::Model model = utils::CreateModel(builder, {{"output", output}});
    const webnn::Compilation compiledModel = utils::AwaitCompile(model);
    const std::vector<float> inputData = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                          12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                          24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34};
    const webnn::Result result =
        utils::AwaitCompute(compiledModel,
                            {{"input", {inputData.data(), inputData.size() * sizeof(float)}}})
            .Get("output");
    EXPECT_TRUE(utils::CheckShape(result, {1, 1, 4, 3}));
    const std::vector<float> expectedValue(
        {12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.});
    EXPECT_TRUE(utils::CheckValue(result, expectedValue));
}

TEST_F(Conv2dTests, Conv2dWithStrides2AndAsymetricPadding) {
    const webnn::ModelBuilder builder = GetContext().CreateModelBuilder();
    const webnn::Operand input = utils::BuildInput(builder, "input", {1, 1, 5, 5});
    const std::vector<float> filterData(8, 1);
    const webnn::Operand filter = utils::BuildConstant(builder, {1, 1, 4, 2}, filterData.data(),
                                                       filterData.size() * sizeof(float));
    utils::Conv2dOptions options;
    options.padding = {1, 2, 0, 1};
    options.strides = {2, 2};
    const webnn::Operand output = builder.Conv2d(input, filter, options.AsPtr());
    const webnn::Model model = utils::CreateModel(builder, {{"output", output}});
    const webnn::Compilation compiledModel = utils::AwaitCompile(model);
    const std::vector<float> inputData = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                          13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    const webnn::Result result =
        utils::AwaitCompute(compiledModel,
                            {{"input", {inputData.data(), inputData.size() * sizeof(float)}}})
            .Get("output");
    EXPECT_TRUE(utils::CheckShape(result, {1, 1, 3, 3}));
    const std::vector<float> expectedValue({33, 45, 27, 104, 120, 66, 72, 80, 43});
    EXPECT_TRUE(utils::CheckValue(result, expectedValue));
}