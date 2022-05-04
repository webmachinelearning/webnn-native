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

class SliceTests : public WebnnTest {
    void SetUp() override {
        builder = utils::CreateGraphBuilder(GetContext());
    }

  protected:
    struct Tensor {
        std::vector<int32_t> shape;
        std::vector<float> value;
    };

    void CheckSlice(const Tensor& input,
                    const std::vector<int>& starts,
                    const std::vector<int>& sizes,
                    const Tensor& expected,
                    utils::SliceOptions options = {}) {
        const wnn::Operand x = utils::BuildInput(builder, "input", input.shape);
        const wnn::Operand output =
            builder.Slice(x, (int32_t*)starts.data(), starts.size(), (int32_t*)sizes.data(),
                          sizes.size(), options.AsPtr());
        const wnn::Graph graph = utils::Build(builder, {{"output", output}});
        ASSERT_TRUE(graph);
        std::vector<float> result(utils::SizeOfShape(expected.shape));
        utils::Compute(graph, {{"input", input.value}}, {{"output", result}});
        EXPECT_TRUE(utils::CheckValue(result, expected.value));
    }

    wnn::GraphBuilder builder;
};

TEST_F(SliceTests, SliceTests) {
    Tensor input = {
        {3, 4, 5},
        {1.3165863,  4.1239005,  4.6697399,  -6.6145003, -3.7128052, -1.0660021, 7.5784922,
         3.5759725,  1.9211160,  -8.1603736, 1.1800343,  -1.8293047, -2.1316205, -3.6369815,
         6.4205879,  7.1544610,  6.8498695,  1.0001093,  -5.6261641, -7.3343945, 1.6827687,
         1.2653192,  5.8872145,  3.1535852,  3.5038650,  3.5865438,  -3.6469769, -8.7751287,
         2.7995768,  -1.6042528, 8.6336482,  -1.7991974, -6.8652731, 1.3729302,  -7.7775210,
         1.0199220,  4.2299256,  1.1432177,  -5.0116669, 1.5525131,  -8.7060851, 4.5739245,
         1.3543987,  -1.5927458, 9.1792661,  -4.5001405, 1.9954188,  -5.1338053, -4.1026011,
         -1.2718531, 4.2538303,  -1.5449624, -3.4380481, 7.8374326,  1.7837452,  9.6105379,
         -4.8783422, -9.4987392, -8.8750905, -9.8019439}};
    std::vector<int> starts = {0, 0, 1};
    std::vector<int> sizes = {2, 3, 4};
    Tensor expected = {{2, 3, 4},
                       {4.1239005, 4.6697399,  -6.6145003, -3.7128052, 7.5784922,  3.5759725,
                        1.9211160, -8.1603736, -1.8293047, -2.1316205, -3.6369815, 6.4205879,
                        1.2653192, 5.8872145,  3.1535852,  3.5038650,  -3.6469769, -8.7751287,
                        2.7995768, -1.6042528, -1.7991974, -6.8652731, 1.3729302,  -7.7775210}};
    CheckSlice(input, starts, sizes, expected);

    starts = {-3, -4, -4};
    sizes = {2, 3, 4};
    CheckSlice(input, starts, sizes, expected);
}

TEST_F(SliceTests, SliceTestsWithAxes) {
    Tensor input = {
        {3, 4, 5},
        {1.3165863,  4.1239005,  4.6697399,  -6.6145003, -3.7128052, -1.0660021, 7.5784922,
         3.5759725,  1.9211160,  -8.1603736, 1.1800343,  -1.8293047, -2.1316205, -3.6369815,
         6.4205879,  7.1544610,  6.8498695,  1.0001093,  -5.6261641, -7.3343945, 1.6827687,
         1.2653192,  5.8872145,  3.1535852,  3.5038650,  3.5865438,  -3.6469769, -8.7751287,
         2.7995768,  -1.6042528, 8.6336482,  -1.7991974, -6.8652731, 1.3729302,  -7.7775210,
         1.0199220,  4.2299256,  1.1432177,  -5.0116669, 1.5525131,  -8.7060851, 4.5739245,
         1.3543987,  -1.5927458, 9.1792661,  -4.5001405, 1.9954188,  -5.1338053, -4.1026011,
         -1.2718531, 4.2538303,  -1.5449624, -3.4380481, 7.8374326,  1.7837452,  9.6105379,
         -4.8783422, -9.4987392, -8.8750905, -9.8019439}};
    std::vector<int> starts = {0, 1};
    std::vector<int> sizes = {2, 4};
    utils::SliceOptions options;
    options.axes = {0, 2};
    Tensor expected = {
        {2, 4, 4},
        {4.1239005,  4.6697399,  -6.6145003, -3.7128052, 7.5784922,  3.5759725, 1.9211160,
         -8.1603736, -1.8293047, -2.1316205, -3.6369815, 6.4205879,  6.8498695, 1.0001093,
         -5.6261641, -7.3343945, 1.2653192,  5.8872145,  3.1535852,  3.5038650, -3.6469769,
         -8.7751287, 2.7995768,  -1.6042528, -1.7991974, -6.8652731, 1.3729302, -7.7775210,
         4.2299256,  1.1432177,  -5.0116669, 1.5525131}};
    CheckSlice(input, starts, sizes, expected, options);

    options.axes = {-3, -1};
    CheckSlice(input, starts, sizes, expected, options);
}
