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

#include "examples/SampleUtils.h"
#include "tests/unittests/validation/ValidationTest.h"

#include <gmock/gmock.h>

using namespace testing;

class MockGraphBuildCallback {
  public:
    MOCK_METHOD(void,
                Call,
                (MLBuildGraphStatus status, MLGraph impl, const char* message, void* userdata));
};

static std::unique_ptr<MockGraphBuildCallback> mockGraphBuildCallback;
static void ToMockGraphBuildCallback(MLBuildGraphStatus status,
                                     MLGraph impl,
                                     const char* message,
                                     void* userdata) {
    mockGraphBuildCallback->Call(status, impl, message, userdata);
}

class GraphValidationTest : public ValidationTest {
  protected:
    void SetUp() override {
        ValidationTest::SetUp();
        mockGraphBuildCallback = std::make_unique<MockGraphBuildCallback>();
        std::vector<int32_t> shape = {2, 2};
        ml::OperandDescriptor inputDesc = {ml::OperandType::Float32, shape.data(),
                                           (uint32_t)shape.size()};
        ml::Operand a = mBuilder.Input("input", &inputDesc);
        std::vector<float> data(4, 1);
        ml::Operand b = mBuilder.Constant(&inputDesc, data.data(), data.size() * sizeof(float));
        mOutput = mBuilder.Add(a, b);
    }

    void TearDown() override {
        ValidationTest::TearDown();

        // Delete mocks so that expectations are checked
        mockGraphBuildCallback = nullptr;
    }

    ml::Operand mOutput;
};

// Test the simple success case.
TEST_F(GraphValidationTest, BuildCallBackSuccess) {
    ml::NamedOperands namedOperands = ml::CreateNamedOperands();
    namedOperands.Set("output", mOutput);
    EXPECT_CALL(*mockGraphBuildCallback, Call(MLBuildGraphStatus_Error, _, _, this)).Times(1);
    mBuilder.Build(namedOperands, ToMockGraphBuildCallback, this);
}

// Create model with null nameOperands
TEST_F(GraphValidationTest, BuildCallBackError) {
    ml::NamedOperands namedOperands = ml::CreateNamedOperands();
    EXPECT_CALL(*mockGraphBuildCallback, Call(MLBuildGraphStatus_Error, _, _, this)).Times(1);
    mBuilder.Build(namedOperands, ToMockGraphBuildCallback, this);
}
