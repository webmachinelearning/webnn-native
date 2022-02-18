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

class TanhTests : public WebnnTest {
  public:
    void TestTanh(const std::vector<float>& inputData,
                  const std::vector<float>& expectedData,
                  const std::vector<int32_t>& shape) {
        const wnn::GraphBuilder builder = wnn::CreateGraphBuilder(GetContext());
        const wnn::Operand a = utils::BuildInput(builder, "a", shape);
        const wnn::Operand b = builder.Tanh(a);
        const wnn::Graph graph = utils::Build(builder, {{"b", b}});
        ASSERT_TRUE(graph);
        std::vector<float> result(utils::SizeOfShape(shape));
        utils::Compute(graph, {{"a", inputData}}, {{"b", result}});
        EXPECT_TRUE(utils::CheckValue(result, expectedData));
    }
};

TEST_F(TanhTests, TanhWith1DTensor) {
    const std::vector<int32_t> shape = {3};
    const std::vector<float> inputData = {-1, 0, 1};
    const std::vector<float> expectedData = {-0.76159418, 0., 0.76159418};
    TestTanh(inputData, expectedData, shape);
}

TEST_F(TanhTests, TanhWith3DTensor) {
    const std::vector<int32_t> shape = {3, 4, 5};
    const std::vector<float> inputData = {
        0.15102264,  -1.1556778,  -0.0657572,  -0.04362043, 1.13937,     0.5458485,   -1.1451102,
        0.3929889,   0.56226826,  -0.68606883, 0.46685237,  -0.53841704, 0.7025275,   -1.5314125,
        0.28699,     0.84823394,  -0.18585628, -0.319641,   0.41442505,  0.88782656,  1.0844846,
        -0.56016934, 0.531165,    0.73836696,  1.0364187,   -0.07221687, -0.9580888,  1.8173703,
        -1.5682113,  -1.272829,   2.331454,    0.2967249,   0.21472701,  -0.9332915,  2.3962052,
        0.498327,    0.53040606,  1.6241137,   0.8147571,   -0.6471784,  0.8489049,   -0.33946696,
        -0.67703784, -0.07758674, 0.7667829,   0.58996105,  0.7728692,   -0.47817922, 2.1541011,
        -1.1611695,  2.1465113,   0.64678246,  1.239878,    -0.10861816, 0.07814338,  -1.026162,
        -0.8464255,  0.53589034,  0.93667775,  1.2927296};
    const std::vector<float> expectedData = {
        0.14988485,  -0.8196263,  -0.06566259, -0.04359278, 0.81420183,  0.49740228,  -0.8161277,
        0.37393406,  0.50965846,  -0.59545064, 0.43565258,  -0.4917888,  0.60596967,  -0.910666,
        0.27936205,  0.69014573,  -0.18374546, -0.30918226, 0.39222348,  0.71031857,  0.79485625,
        -0.5081031,  0.4862711,   0.6281575,   0.7764699,   -0.07209159, -0.74342316, 0.94857556,
        -0.9167408,  -0.8545626,  0.98129857,  0.28831258,  0.21148658,  -0.7321248,  0.9835515,
        0.4608004,   0.48569143,  0.9252187,   0.67220616,  -0.5697674,  0.6904969,   -0.32700142,
        -0.5895903,  -0.07743143, 0.6450549,   0.5298676,   0.64859474,  -0.44478422, 0.97344196,
        -0.82142067, 0.97304124,  0.56949997,  0.84542084,  -0.10819301, 0.07798471,  -0.7723645,
        -0.6891974,  0.48987082,  0.7336921,   0.85983974};
    TestTanh(inputData, expectedData, shape);
}
