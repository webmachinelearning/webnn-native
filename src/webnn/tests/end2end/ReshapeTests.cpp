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

#include "webnn/tests/WebnnTest.h"

class ReshapeTests : public WebnnTest {
  public:
    void TestReshape(const std::vector<int32_t>& oldShape,
                     const std::vector<int32_t>& newShape,
                     const std::vector<int32_t>& expectedShape) {
        const wnn::GraphBuilder builder = wnn::CreateGraphBuilder(GetContext());
        const wnn::Operand a = utils::BuildInput(builder, "a", oldShape);
        const wnn::Operand b = builder.Reshape(a, newShape.data(), newShape.size());
        const wnn::Graph graph = utils::Build(builder, {{"b", b}});
        ASSERT_TRUE(graph);
        const std::vector inputData =
            newShape.empty() ? std::vector<float>{1}
                             : std::vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                                  13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        std::vector<float> result(utils::SizeOfShape(expectedShape));
        utils::Compute(graph, {{"a", inputData}}, {{"b", result}});
        EXPECT_TRUE(utils::CheckValue(result, inputData));
    }
};

TEST_F(ReshapeTests, ReshapeReorderedAllDims) {
    TestReshape({2, 3, 4}, {4, 2, 3}, {4, 2, 3});
}

TEST_F(ReshapeTests, ReshapeReorderedLastDims) {
    TestReshape({2, 3, 4}, {2, 4, 3}, {2, 4, 3});
}

TEST_F(ReshapeTests, ReshapeReducedDims) {
    TestReshape({2, 3, 4}, {2, 12}, {2, 12});
}

TEST_F(ReshapeTests, ReshapeExtendedDims) {
    TestReshape({2, 3, 4}, {2, 3, 2, 2}, {2, 3, 2, 2});
}

TEST_F(ReshapeTests, ReshapeOneDim) {
    TestReshape({2, 3, 4}, {24}, {24});
}

TEST_F(ReshapeTests, ReshapeNegativeDim) {
    TestReshape({2, 3, 4}, {2, -1, 2}, {2, 6, 2});
}

TEST_F(ReshapeTests, ReshapeNegativeDim1) {
    TestReshape({2, 3, 4}, {-1, 2, 3, 4}, {1, 2, 3, 4});
}

TEST_F(ReshapeTests, ReshapeZeroDim) {
    TestReshape({}, {}, {});
}