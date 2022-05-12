// Copyright 2021 The WebNN-native Authors

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

#include "webnn/tests/WebnnTest.h"

class PadTests : public WebnnTest {
  public:
    void TestPad(const std::vector<int32_t>& inputShape,
                 const std::vector<float>& inputData,
                 const std::vector<int32_t>& paddingShape,
                 const std::vector<uint32_t>& paddingData,
                 const std::vector<int32_t>& expectedShape,
                 const std::vector<float>& expectedValue,
                 wnn::PaddingMode mode = wnn::PaddingMode::Constant) {
        const wnn::GraphBuilder builder = wnn::CreateGraphBuilder(GetContext());
        const wnn::Operand x = utils::BuildInput(builder, "x", inputShape);
        const wnn::Operand padding =
            utils::BuildConstant(builder, paddingShape, paddingData.data(),
                                 paddingData.size() * sizeof(int32_t), wnn::OperandType::Uint32);
        wnn::PadOptions options;
        options.mode = mode;
        wnn::Operand y = builder.Pad(x, padding, &options);
        const wnn::Graph graph = utils::Build(builder, {{"y", y}});
        ASSERT_TRUE(graph);
        std::vector<float> result(utils::SizeOfShape(expectedShape));
        utils::Compute(graph, {{"x", inputData}}, {{"y", result}});
        EXPECT_TRUE(utils::CheckValue(result, expectedValue));
    }
};

TEST_F(PadTests, PadDefault) {
    TestPad({2, 3}, {1, 2, 3, 4, 5, 6}, {2, 2}, {1, 1, 2, 2}, {4, 7},
            {0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2., 3., 0., 0.,
             0., 0., 4., 5., 6., 0., 0., 0., 0., 0., 0., 0., 0., 0.});
}

TEST_F(PadTests, PadEdgeMode) {
    TestPad({2, 3}, {1, 2, 3, 4, 5, 6}, {2, 2}, {1, 1, 2, 2}, {4, 7},
            {1., 1., 1., 2., 3., 3., 3., 1., 1., 1., 2., 3., 3., 3.,
             4., 4., 4., 5., 6., 6., 6., 4., 4., 4., 5., 6., 6., 6.},
            wnn::PaddingMode::Edge);
}

TEST_F(PadTests, PadReflectionMode) {
    TestPad({2, 3}, {1, 2, 3, 4, 5, 6}, {2, 2}, {1, 1, 2, 2}, {4, 7},
            {6., 5., 4., 5., 6., 5., 4., 3., 2., 1., 2., 3., 2., 1.,
             6., 5., 4., 5., 6., 5., 4., 3., 2., 1., 2., 3., 2., 1.},
            wnn::PaddingMode::Reflection);
}

TEST_F(PadTests, PadSymmetricMode) {
    TestPad({2, 3}, {1, 2, 3, 4, 5, 6}, {2, 2}, {1, 1, 2, 2}, {4, 7},
            {2., 1., 1., 2., 3., 3., 2., 2., 1., 1., 2., 3., 3., 2.,
             5., 4., 4., 5., 6., 6., 5., 5., 4., 4., 5., 6., 6., 5.},
            wnn::PaddingMode::Symmetric);
}
