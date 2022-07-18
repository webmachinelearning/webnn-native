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

class HardSwishTests : public WebnnTest {
  protected:
    void CheckHardSwish(const std::vector<int32_t>& inputShape,
                        const std::vector<float>& inputBuffer,
                        const std::vector<float>& expectedBuffer) {
        const wnn::GraphBuilder builder = wnn::CreateGraphBuilder(GetContext());
        const wnn::Operand x = utils::BuildInput(builder, "x", inputShape);
        const wnn::Operand y = builder.HardSwish(x);
        const wnn::Graph graph = utils::Build(builder, {{"y", y}});
        ASSERT_TRUE(graph);
        std::vector<float> result(utils::SizeOfShape(inputShape));
        utils::Compute(GetContext(), graph, {{"x", inputBuffer}}, {{"y", result}});
        EXPECT_TRUE(utils::CheckValue(result, expectedBuffer));
    }
};

TEST_F(HardSwishTests, HardSwishByDefault) {
    CheckHardSwish({2, 3}, {-4.2, -3.001, -3., 0.6, 2.994, 3.001},
                   {0., 0., 0., 0.36, 2.991006, 3.001});
}
