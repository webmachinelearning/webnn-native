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

#include <cstdlib>

class SqueezeTests : public WebnnTest {
  protected:
    void CheckSqueeze(const std::vector<int32_t>& inputShape,
                      const std::vector<int32_t>& axes,
                      const std::vector<int32_t>& expectedShape) {
        const wnn::GraphBuilder builder = utils::CreateGraphBuilder(GetContext());
        const wnn::Operand x = utils::BuildInput(builder, "x", inputShape);
        wnn::SqueezeOptions options;
        if (!axes.empty()) {
            options.axes = axes.data();
            options.axesCount = axes.size();
        }
        const wnn::Operand y = builder.Squeeze(x, axes.empty() ? nullptr : &options);
        const wnn::Graph graph = utils::Build(builder, {{"y", y}});
        ASSERT_TRUE(graph);
        std::vector<float> inputBuffer(utils::SizeOfShape(inputShape));
        for (auto& input : inputBuffer) {
            input = std::rand();
        }
        std::vector<float> result(utils::SizeOfShape(expectedShape));
        utils::Compute(graph, {{"x", inputBuffer}}, {{"y", result}});
        EXPECT_TRUE(utils::CheckValue(result, inputBuffer));
    }
};

TEST_F(SqueezeTests, SqueezeOneDimensionByDefault) {
    CheckSqueeze({1, 3, 4, 5}, {}, {3, 4, 5});
}

TEST_F(SqueezeTests, SqueezeOneDimensionWithAxes) {
    CheckSqueeze({1, 3, 1, 5}, {0}, {3, 1, 5});
}

TEST_F(SqueezeTests, SqueezeTwoDimensionByDefault) {
    CheckSqueeze({1, 3, 1, 5}, {}, {3, 5});
}

TEST_F(SqueezeTests, SqueezeTwoDimensionWithAxes) {
    CheckSqueeze({1, 3, 1, 5}, {0, 2}, {3, 5});
}
