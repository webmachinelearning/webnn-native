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

#include "tests/unittests/validation/ValidationTest.h"

#include <memory>

using namespace testing;

class Conv2dValidationTest : public ValidationTest {
  protected:
    void SetUp() override {
        ValidationTest::SetUp();
        std::vector<int32_t> shape = {1, 1, 5, 5};
        webnn::OperandDescriptor inputDesc = {webnn::OperandType::Float32, shape.data(),
                                              (uint32_t)shape.size()};
        mInput = mBuilder.Input("input", &inputDesc);

        shape = {1, 1, 3, 3};
        std::vector<float> data(9, 1);
        inputDesc = {webnn::OperandType::Float32, shape.data(), (uint32_t)shape.size()};
        mFilter = mBuilder.Constant(&inputDesc, data.data(), data.size() * sizeof(float));
    }
    webnn::Operand mInput;
    webnn::Operand mFilter;
};

TEST_F(Conv2dValidationTest, CreateByDefaultOptions) {
    // Success
    {
        // using default value for options
        webnn::Conv2dOptions conv2dOptions = {};
        webnn::Operand conv = mBuilder.Conv2d(mInput, mFilter, &conv2dOptions);
    }
    { webnn::Operand conv = mBuilder.Conv2d(mInput, mFilter); }
}

TEST_F(Conv2dValidationTest, DifferentTypeError) {
    // input type is fp32 while filter type is int32
    std::vector<int32_t> shape = {1, 1, 3, 3};
    std::vector<int32_t> data(9, 1);
    webnn::OperandDescriptor inputDesc = {webnn::OperandType::Int32, shape.data(),
                                          (uint32_t)shape.size()};
    webnn::Operand filter =
        mBuilder.Constant(&inputDesc, data.data(), data.size() * sizeof(int32_t));
    webnn::Conv2dOptions conv2dOptions = {};
    ASSERT_CONTEXT_ERROR(mBuilder.Conv2d(mInput, filter, &conv2dOptions));
}

TEST_F(Conv2dValidationTest, InvalidInputDimsError) {
    // input rank is not 4
    std::vector<int32_t> shape = {1, 1, 5};
    webnn::OperandDescriptor inputDesc = {webnn::OperandType::Float32, shape.data(),
                                          (uint32_t)shape.size()};
    webnn::Operand input = mBuilder.Input("input", &inputDesc);
    webnn::Conv2dOptions conv2dOptions = {};
    ASSERT_CONTEXT_ERROR(mBuilder.Conv2d(input, mFilter, &conv2dOptions));
}

TEST_F(Conv2dValidationTest, InvalidFilterDimsError) {
    // filter rank is 3
    std::vector<int32_t> shape = {1, 1, 3};
    std::vector<float> data(3, 1);
    webnn::OperandDescriptor inputDesc = {webnn::OperandType::Float32, shape.data(),
                                          (uint32_t)shape.size()};
    webnn::Operand filter = mBuilder.Constant(&inputDesc, data.data(), data.size() * sizeof(float));
    webnn::Conv2dOptions conv2dOptions = {};
    ASSERT_CONTEXT_ERROR(mBuilder.Conv2d(mInput, filter, &conv2dOptions));
}

TEST_F(Conv2dValidationTest, InvalidOptions) {
    webnn::Conv2dOptions options = {};
    {
        // invalid paddingCount
        std::vector<int32_t> padding = {1, 1, 1};
        options.padding = padding.data();
        options.paddingCount = 3;
        options.strides = nullptr;
        options.dilations = nullptr;
        ASSERT_CONTEXT_ERROR(mBuilder.Conv2d(mInput, mFilter, &options));
    }
    {
        // invalid stridesCount
        options.padding = nullptr;
        std::vector<int32_t> strides = {1, 1, 1};
        options.strides = strides.data();
        options.stridesCount = 3;
        options.dilations = nullptr;
        ASSERT_CONTEXT_ERROR(mBuilder.Conv2d(mInput, mFilter, &options));
    }
    {
        // invalid dilationCount
        options.padding = nullptr;
        std::vector<int32_t> dilations = {1, 1, 1};
        options.dilations = dilations.data();
        options.dilationsCount = 3;
        options.strides = nullptr;
        ASSERT_CONTEXT_ERROR(mBuilder.Conv2d(mInput, mFilter, &options));
    }
}
