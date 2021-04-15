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

class UnaryValidationTest : public ValidationTest {};

TEST_F(UnaryValidationTest, SoftmaxValidation) {
    // success
    {
        std::vector<int32_t> shape = {2, 2};
        ml::OperandDescriptor inputDesc = {ml::OperandType::Float32, shape.data(),
                                           (uint32_t)shape.size()};
        ml::Operand a = mBuilder.Input("input", &inputDesc);
        ml::Operand softmax = mBuilder.Softmax(a);
    }
    // Input dimensions is incorrect
    {
        std::vector<int32_t> shape = {2, 2, 2};
        ml::OperandDescriptor inputDesc = {ml::OperandType::Float32, shape.data(),
                                           (uint32_t)shape.size()};
        ml::Operand a = mBuilder.Input("input", &inputDesc);
        ASSERT_CONTEXT_ERROR(mBuilder.Softmax(a));
    }
}

TEST_F(UnaryValidationTest, ReluValidation) {
    // success
    {
        std::vector<int32_t> shape = {2, 2};
        ml::OperandDescriptor inputDesc = {ml::OperandType::Float32, shape.data(),
                                           (uint32_t)shape.size()};
        ml::Operand a = mBuilder.Input("input", &inputDesc);
        ml::Operand relu = mBuilder.Relu(a);
    }
}
