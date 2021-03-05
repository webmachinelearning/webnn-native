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

class MockModelCompileCallback {
  public:
    MOCK_METHOD(
        void,
        Call,
        (WebnnCompileStatus status, WebnnCompilation impl, const char* message, void* userdata));
};

static std::unique_ptr<MockModelCompileCallback> mockModelCompileCallback;
static void ToMockModelCompileCallback(WebnnCompileStatus status,
                                       WebnnCompilation impl,
                                       const char* message,
                                       void* userdata) {
    mockModelCompileCallback->Call(status, impl, message, userdata);
}

class ModelValidationTest : public ValidationTest {
  protected:
    void SetUp() override {
        ValidationTest::SetUp();
        mockModelCompileCallback = std::make_unique<MockModelCompileCallback>();
        std::vector<int32_t> shape = {2, 2};
        webnn::OperandDescriptor inputDesc = {webnn::OperandType::Float32, shape.data(),
                                              (uint32_t)shape.size()};
        webnn::Operand a = mBuilder.Input("input", &inputDesc);
        std::vector<float> data(4, 1);
        webnn::Operand b = mBuilder.Constant(&inputDesc, data.data(), data.size() * sizeof(float));
        mOutput = mBuilder.Add(a, b);
    }

    void TearDown() override {
        ValidationTest::TearDown();

        // Delete mocks so that expectations are checked
        mockModelCompileCallback = nullptr;
    }

    webnn::Operand mOutput;
};

// Test the simple success case.
TEST_F(ModelValidationTest, CompileCallBackSuccess) {
    webnn::NamedOperands namedOperands = webnn::CreateNamedOperands();
    namedOperands.Set("output", mOutput);
    webnn::Model model = mBuilder.CreateModel(namedOperands);
    EXPECT_CALL(*mockModelCompileCallback, Call(WebnnCompileStatus_Success, _, nullptr, this))
        .Times(1);
    model.Compile(ToMockModelCompileCallback, this);
}

// Create model with null nameOperands
TEST_F(ModelValidationTest, CompileCallBackError) {
    webnn::NamedOperands namedOperands = webnn::CreateNamedOperands();
    webnn::Model model = mBuilder.CreateModel(namedOperands);
    EXPECT_CALL(*mockModelCompileCallback, Call(WebnnCompileStatus_Error, _, _, this)).Times(1);
    model.Compile(ToMockModelCompileCallback, this);
}
