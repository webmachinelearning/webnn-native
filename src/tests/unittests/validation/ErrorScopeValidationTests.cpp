// Copyright 2018 The Dawn Authors
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

#include <gmock/gmock.h>

using namespace testing;

class MockContextPopErrorScopeCallback {
  public:
    MOCK_METHOD(void, Call, (MLErrorType type, const char* message, void* userdata));
};

static std::unique_ptr<MockContextPopErrorScopeCallback> mockContextPopErrorScopeCallback;
static void ToMockContextPopErrorScopeCallback(MLErrorType type,
                                               const char* message,
                                               void* userdata) {
    mockContextPopErrorScopeCallback->Call(type, message, userdata);
}

class ErrorScopeValidationTest : public ValidationTest {
  private:
    void SetUp() override {
        ValidationTest::SetUp();
        mockContextPopErrorScopeCallback = std::make_unique<MockContextPopErrorScopeCallback>();
    }

    void TearDown() override {
        ValidationTest::TearDown();

        // Delete mocks so that expectations are checked
        mockContextPopErrorScopeCallback = nullptr;
    }
};

// Test the simple success case.
TEST_F(ErrorScopeValidationTest, Success) {
    mContext.PushErrorScope(ml::ErrorFilter::Validation);

    EXPECT_CALL(*mockContextPopErrorScopeCallback, Call(MLErrorType_NoError, _, this)).Times(1);
    mContext.PopErrorScope(ToMockContextPopErrorScopeCallback, this);
}

// Test the simple case where the error scope catches an error.
TEST_F(ErrorScopeValidationTest, CatchesError) {
    mContext.PushErrorScope(ml::ErrorFilter::Validation);

    std::vector<int32_t> shape = {2, 2, 2};
    ml::OperandDescriptor inputDesc = {ml::OperandType::Float32, shape.data(),
                                       (uint32_t)shape.size()};
    ml::Operand a = mBuilder.Input("input", &inputDesc);
    mBuilder.Softmax(a);

    EXPECT_CALL(*mockContextPopErrorScopeCallback, Call(MLErrorType_Validation, _, this)).Times(1);
    mContext.PopErrorScope(ToMockContextPopErrorScopeCallback, this);
}

// Test that if no error scope handles an error, it goes to the context UncapturedError callback
TEST_F(ErrorScopeValidationTest, UnhandledErrorsMatchUncapturedErrorCallback) {
    mContext.PushErrorScope(ml::ErrorFilter::OutOfMemory);

    std::vector<int32_t> shape = {2, 2, 2};
    ml::OperandDescriptor inputDesc = {ml::OperandType::Float32, shape.data(),
                                       (uint32_t)shape.size()};
    ml::Operand a = mBuilder.Input("input", &inputDesc);
    ASSERT_CONTEXT_ERROR(mBuilder.Softmax(a));

    EXPECT_CALL(*mockContextPopErrorScopeCallback, Call(MLErrorType_NoError, _, this)).Times(1);
    mContext.PopErrorScope(ToMockContextPopErrorScopeCallback, this);
}

// Check that push/popping error scopes must be balanced.
TEST_F(ErrorScopeValidationTest, PushPopBalanced) {
    // No error scopes to pop.
    { EXPECT_FALSE(mContext.PopErrorScope(ToMockContextPopErrorScopeCallback, this)); }

    // Too many pops
    {
        mContext.PushErrorScope(ml::ErrorFilter::Validation);

        EXPECT_CALL(*mockContextPopErrorScopeCallback, Call(MLErrorType_NoError, _, this + 1))
            .Times(1);
        mContext.PopErrorScope(ToMockContextPopErrorScopeCallback, this + 1);

        EXPECT_FALSE(mContext.PopErrorScope(ToMockContextPopErrorScopeCallback, this + 2));
    }
}
