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

class TransposeValidationTest : public ValidationTest {
  protected:
    void SetUp() override {
        ValidationTest::SetUp();
        std::vector<int32_t> shape = {2, 3, 4};
        webnn::OperandDescriptor inputDesc = {webnn::OperandType::Float32, shape.data(),
                                              (uint32_t)shape.size()};
        mInput = mBuilder.Input("input", &inputDesc);
    }
    webnn::Operand mInput;
};

TEST_F(TransposeValidationTest, CreateByDefaultOptions) {
    // success
    { webnn::Operand transpose = mBuilder.Transpose(mInput); }
    {
        webnn::TransposeOptions options = {};
        webnn::Operand transpose = mBuilder.Transpose(mInput, &options);
    }
}

TEST_F(TransposeValidationTest, InvalidOptions) {
    // success
    {
        std::vector<int32_t> permutation = {2, 0, 1};
        webnn::TransposeOptions options;
        options.permutation = permutation.data();
        options.permutationCount = permutation.size();
        webnn::Operand transpose = mBuilder.Transpose(mInput, &options);
    }
    // permutation size is invalid
    {
        std::vector<int32_t> permutation = {2, 0, 1, 3};
        webnn::TransposeOptions options;
        options.permutation = permutation.data();
        options.permutationCount = permutation.size();
        ASSERT_CONTEXT_ERROR(mBuilder.Transpose(mInput, &options));
    }
    // permutation value is invalid
    {
        std::vector<int32_t> permutation = {3, 2, 2};
        webnn::TransposeOptions options;
        options.permutation = permutation.data();
        options.permutationCount = permutation.size();
        ASSERT_CONTEXT_ERROR(mBuilder.Transpose(mInput, &options));
    }
    // permutation value is invalid
    {
        std::vector<int32_t> permutation = {3, 2, 4};
        webnn::TransposeOptions options;
        options.permutation = permutation.data();
        options.permutationCount = permutation.size();
        ASSERT_CONTEXT_ERROR(mBuilder.Transpose(mInput, &options));
    }
}
