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

class PoolValidationTest : public ValidationTest {};

TEST_F(PoolValidationTest, CreateByDefaultOptions) {
    // Success
    std::vector<int32_t> shape = {1, 100, 1000, 1000};
    ml::OperandDescriptor inputDesc = {ml::OperandType::Float32, shape.data(),
                                       (uint32_t)shape.size()};
    ml::Operand input = mBuilder.Input("input", &inputDesc);
    {
        // using default value for options
        ml::Pool2dOptions pool2dOptions = {};
        ml::Operand pool = mBuilder.AveragePool2d(input, &pool2dOptions);
    }

    { ml::Operand pool = mBuilder.MaxPool2d(input); }
}

TEST_F(PoolValidationTest, InputDimsError) {
    // input is not a 4D tensor
    std::vector<int32_t> shape = {1, 100, 1000, 1000, 1};
    ml::OperandDescriptor inputDesc = {ml::OperandType::Float32, shape.data(),
                                       (uint32_t)shape.size()};
    ml::Operand input = mBuilder.Input("input", &inputDesc);

    ml::Pool2dOptions pool2dOptions = {};
    ml::Operand pool;
    ASSERT_CONTEXT_ERROR(pool = mBuilder.MaxPool2d(input, &pool2dOptions));
    // input variable pool is not valid
    ASSERT_CONTEXT_ERROR(mBuilder.MaxPool2d(pool));
}

TEST_F(PoolValidationTest, FilterCountError) {
    std::vector<int32_t> shape = {1, 100, 1000, 1000};
    ml::OperandDescriptor inputDesc = {ml::OperandType::Float32, shape.data(),
                                       (uint32_t)shape.size()};
    ml::Operand input = mBuilder.Input("input", &inputDesc);
    // windowDimensionsCount is incorrect
    {
        ml::Pool2dOptions options;
        std::vector<int32_t> windowDimensions = {2, 2, 1};
        options.windowDimensions = windowDimensions.data();
        options.windowDimensionsCount = 3;
        options.strides = nullptr;
        options.padding = nullptr;
        options.dilations = nullptr;
        ASSERT_CONTEXT_ERROR(mBuilder.MaxPool2d(input, &options));
    }
    // paddingCount is incorrect
    {
        ml::Pool2dOptions options;
        options.windowDimensions = nullptr;
        options.strides = nullptr;
        std::vector<int32_t> padding = {1, 1};
        options.padding = padding.data();
        options.paddingCount = 2;
        options.dilations = nullptr;
        ASSERT_CONTEXT_ERROR(mBuilder.MaxPool2d(input, &options));
    }
    // stridesCount is incorrect
    {
        ml::Pool2dOptions options;
        options.windowDimensions = nullptr;
        std::vector<int32_t> strides = {1};
        options.strides = strides.data();
        options.stridesCount = 1;
        options.padding = nullptr;
        options.dilations = nullptr;
        ASSERT_CONTEXT_ERROR(mBuilder.MaxPool2d(input, &options));
    }
    // dilationsCount is incorrect
    {
        ml::Pool2dOptions options;
        options.windowDimensions = nullptr;
        options.strides = nullptr;
        options.padding = nullptr;
        std::vector<int32_t> dilations = {1};
        options.dilations = dilations.data();
        options.dilationsCount = 1;
        ASSERT_CONTEXT_ERROR(mBuilder.MaxPool2d(input, &options));
    }
}
