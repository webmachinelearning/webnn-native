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

#include "tests/unittests/validation/ValidationTest.h"

#include <memory>

using namespace testing;

class Conv2dValidationTest : public ValidationTest {
  protected:
    void SetUp() override {
        ValidationTest::SetUp();
        std::vector<int32_t> shape = {1, 1, 5, 5};
        ml::OperandDescriptor inputDesc = {ml::OperandType::Float32, shape.data(),
                                           (uint32_t)shape.size()};
        mInput = mBuilder.Input("input", &inputDesc);

        shape = {1, 1, 3, 3};
        std::vector<float> data(9, 1);
        inputDesc = {ml::OperandType::Float32, shape.data(), (uint32_t)shape.size()};
        ml::ArrayBufferView arrayBuffer = {data.data(), data.size() * sizeof(float)};
        mFilter = mBuilder.Constant(&inputDesc, &arrayBuffer);
    }
    ml::Operand mInput;
    ml::Operand mFilter;
};

TEST_F(Conv2dValidationTest, CreateByDefaultOptions) {
    // Success
    {
        // using default value for options
        ml::Conv2dOptions conv2dOptions = {};
        ml::Operand conv = mBuilder.Conv2d(mInput, mFilter, &conv2dOptions);
    }
    { ml::Operand conv = mBuilder.Conv2d(mInput, mFilter); }
}

TEST_F(Conv2dValidationTest, DifferentTypeError) {
    // input type is fp32 while filter type is int32
    std::vector<int32_t> shape = {1, 1, 3, 3};
    std::vector<int32_t> data(9, 1);
    ml::OperandDescriptor inputDesc = {ml::OperandType::Int32, shape.data(),
                                       (uint32_t)shape.size()};
    ml::ArrayBufferView arrayBuffer = {data.data(), data.size() * sizeof(int32_t)};
    ml::Operand filter = mBuilder.Constant(&inputDesc, &arrayBuffer);
    ml::Conv2dOptions conv2dOptions = {};
    ASSERT_CONTEXT_ERROR(mBuilder.Conv2d(mInput, filter, &conv2dOptions));
}

TEST_F(Conv2dValidationTest, InvalidInputDimsError) {
    // input rank is not 4
    std::vector<int32_t> shape = {1, 1, 5};
    ml::OperandDescriptor inputDesc = {ml::OperandType::Float32, shape.data(),
                                       (uint32_t)shape.size()};
    ml::Operand input = mBuilder.Input("input", &inputDesc);
    ml::Conv2dOptions conv2dOptions = {};
    ASSERT_CONTEXT_ERROR(mBuilder.Conv2d(input, mFilter, &conv2dOptions));
}

TEST_F(Conv2dValidationTest, InvalidFilterDimsError) {
    // filter rank is 3
    std::vector<int32_t> shape = {1, 1, 3};
    std::vector<float> data(3, 1);
    ml::OperandDescriptor inputDesc = {ml::OperandType::Float32, shape.data(),
                                       (uint32_t)shape.size()};
    ml::ArrayBufferView arrayBuffer = {data.data(), data.size() * sizeof(float)};
    ml::Operand filter = mBuilder.Constant(&inputDesc, &arrayBuffer);
    ml::Conv2dOptions conv2dOptions = {};
    ASSERT_CONTEXT_ERROR(mBuilder.Conv2d(mInput, filter, &conv2dOptions));
}

TEST_F(Conv2dValidationTest, InvalidOptions) {
    ml::Conv2dOptions options = {};
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
