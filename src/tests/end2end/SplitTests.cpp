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

#include "src/tests/WebnnTest.h"

#include <cstdlib>

class SplitTests : public WebnnTest {
  private:
    struct expected {
        std::vector<int32_t> shape;
        std::vector<float> buffer;
    };

  protected:
    void testSplit(const std::vector<int32_t>& inputShape,
                   const std::vector<float>& inputBuffer,
                   const std::vector<uint32_t>& splits,
                   const std::vector<expected>& expectedArray,
                   int32_t axis = 0) {
        const wnn::GraphBuilder builder = wnn::CreateGraphBuilder(GetContext());
        const wnn::Operand input = utils::BuildInput(builder, "input", inputShape);
        wnn::SplitOptions options = {axis};
        const wnn::OperandArray splittedOperands =
            builder.Split(input, splits.data(), splits.size(), &options);
        std::vector<utils::NamedOperand> namedOperands;
        for (size_t i = 0; i < splittedOperands.Size(); ++i) {
            namedOperands.push_back({"split" + std::to_string(i), splittedOperands.Get(i)});
        }
        const wnn::Graph graph = utils::Build(builder, namedOperands);
        ASSERT_TRUE(graph);
        std::vector<utils::NamedOutput<float>> namedOutputs;
        std::vector<std::vector<float>> results;
        results.reserve(splittedOperands.Size());
        for (size_t i = 0; i < splittedOperands.Size(); ++i) {
            results.push_back(std::vector<float>(utils::SizeOfShape(expectedArray[i].shape)));
            namedOutputs.push_back({"split" + std::to_string(i), results.back()});
        }
        utils::Compute(graph, {{"input", inputBuffer}}, namedOutputs);
        for (size_t i = 0; i < splittedOperands.Size(); ++i) {
            EXPECT_TRUE(utils::CheckValue(namedOutputs[i].resource, expectedArray[i].buffer));
        }
    }
};

TEST_F(SplitTests, SplitEvenByDefault) {
    testSplit({6}, {1, 2, 3, 4, 5, 6}, {3},
              {
                  {{2}, {1, 2}},
                  {{2}, {3, 4}},
                  {{2}, {5, 6}},
              });
}

TEST_F(SplitTests, SplitByDefault) {
    testSplit({6}, {1, 2, 3, 4, 5, 6}, {2, 4}, {{{2}, {1, 2}}, {{4}, {3, 4, 5, 6}}});
}

TEST_F(SplitTests, SplitEvenOneDimension) {
    testSplit({2, 6}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2},
              {
                  {{2, 3}, {1, 2, 3, 7, 8, 9}},
                  {{2, 3}, {4, 5, 6, 10, 11, 12}},
              },
              1);
}

TEST_F(SplitTests, SplitOneDimension) {
    testSplit({2, 6}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 4},
              {
                  {{2, 2}, {1, 2, 7, 8}},
                  {{2, 4}, {3, 4, 5, 6, 9, 10, 11, 12}},
              },
              1);
}
