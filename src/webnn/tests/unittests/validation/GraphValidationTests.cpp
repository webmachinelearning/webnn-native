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
#include "webnn/tests/unittests/validation/ValidationTest.h"

using namespace testing;

class GraphValidationTest : public ValidationTest {
  protected:
    void SetUp() override {
        ValidationTest::SetUp();
        std::vector<int32_t> shape = {2, 2};
        wnn::OperandDescriptor inputDesc = {wnn::OperandType::Float32, shape.data(),
                                            (uint32_t)shape.size()};
        wnn::Operand a = mBuilder.Input("input", &inputDesc);
        std::vector<float> data(4, 1);
        wnn::ArrayBufferView arrayBuffer = {data.data(), data.size() * sizeof(float)};
        wnn::Operand b = mBuilder.Constant(&inputDesc, &arrayBuffer);
        mOutput = mBuilder.Add(a, b);
    }

    void TearDown() override {
        ValidationTest::TearDown();
    }

    wnn::Operand mOutput;
};

// Test the simple success case.
TEST_F(GraphValidationTest, BuildGraphSuccess) {
    // TODO::Use instance->CreateNamedOperands instead of wnn::CreateNamedOperands
    //  that is removed.
    //  wnn::NamedOperands namedOperands = wnn::CreateNamedOperands();
    //  namedOperands.Set("output", mOutput);
    //  mBuilder.Build(namedOperands);
}

// Create model with null nameOperands
TEST_F(GraphValidationTest, BuildGraphError) {
    // TODO::Use instance->CreateNamedOperands instead of wnn::CreateNamedOperands
    //  that is removed.
    // wnn::NamedOperands namedOperands = wnn::CreateNamedOperands();
    // DAWN_ASSERT(mBuilder.Build(namedOperands) == nullptr);
}
