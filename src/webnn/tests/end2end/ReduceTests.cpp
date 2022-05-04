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

class ReduceTests : public WebnnTest {
  protected:
    enum ReduceType {
        kReduceL1 = 0,
        kReduceL2,
        kReduceMax,
        kReduceMean,
        kReduceMin,
        kReduceProduct,
        kReduceSum,
    };

    void CheckReduce(ReduceType type,
                     const std::vector<int32_t>& inputShape,
                     const std::vector<float>& inputData,
                     const std::vector<int32_t>& expectedShape,
                     const std::vector<float>& expectedValue,
                     const std::vector<int32_t>& axes = {},
                     bool keepDimensions = false) {
        const wnn::GraphBuilder builder = utils::CreateGraphBuilder(GetContext());
        const wnn::Operand a = utils::BuildInput(builder, "a", inputShape);
        wnn::ReduceOptions options;
        if (!axes.empty()) {
            options.axes = axes.data();
            options.axesCount = axes.size();
        }
        if (keepDimensions) {
            options.keepDimensions = keepDimensions;
        }
        wnn::Operand b;
        switch (type) {
            case kReduceL1:
                b = builder.ReduceL1(a, &options);
                break;
            case kReduceL2:
                b = builder.ReduceL2(a, &options);
                break;
            case kReduceMax:
                b = builder.ReduceMax(a, &options);
                break;
            case kReduceMean:
                b = builder.ReduceMean(a, &options);
                break;
            case kReduceMin:
                b = builder.ReduceMin(a, &options);
                break;
            case kReduceProduct:
                b = builder.ReduceProduct(a, &options);
                break;
            case kReduceSum:
                b = builder.ReduceSum(a, &options);
                break;
            default:
                DAWN_ASSERT(0);
        }
        const wnn::Graph graph = utils::Build(builder, {{"b", b}});
        ASSERT_TRUE(graph);
        std::vector<float> result(utils::SizeOfShape(expectedShape));
        utils::Compute(graph, {{"a", inputData}}, {{"b", result}});
        EXPECT_TRUE(utils::CheckValue(result, expectedValue));
    }
};

TEST_F(ReduceTests, ReduceL1Default) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0.9762701, 4.303787,   2.0552676,  0.89766365,
                                          -1.526904, 2.9178822,  -1.2482557, 7.83546,
                                          9.273255,  -2.3311696, 5.834501,   0.5778984};
    const std::vector<int32_t> expectedShape = {};
    const std::vector<float> expectedValue = {39.778313};
    CheckReduce(ReduceType::kReduceL1, inputShape, inputData, expectedShape, expectedValue, {});
}

TEST_F(ReduceTests, ReduceL1DefaultAxesKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0.9762701, 4.303787,   2.0552676,  0.89766365,
                                          -1.526904, 2.9178822,  -1.2482557, 7.83546,
                                          9.273255,  -2.3311696, 5.834501,   0.5778984};
    const std::vector<int32_t> expectedShape = {1, 1, 1};
    const std::vector<float> expectedValue = {39.778313};
    CheckReduce(ReduceType::kReduceL1, inputShape, inputData, expectedShape, expectedValue, {},
                true);
}

TEST_F(ReduceTests, ReduceL1Axes0NotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0.9762701, 4.303787,   2.0552676,  0.89766365,
                                          -1.526904, 2.9178822,  -1.2482557, 7.83546,
                                          9.273255,  -2.3311696, 5.834501,   0.5778984};
    const std::vector<int32_t> expectedShape = {2, 2};
    const std::vector<float> expectedValue = {11.776429, 9.552839, 9.138024, 9.311022};
    CheckReduce(ReduceType::kReduceL1, inputShape, inputData, expectedShape, expectedValue, {0});
}

TEST_F(ReduceTests, ReduceL1Axes1NotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0.9762701, 4.303787,   2.0552676,  0.89766365,
                                          -1.526904, 2.9178822,  -1.2482557, 7.83546,
                                          9.273255,  -2.3311696, 5.834501,   0.5778984};
    const std::vector<int32_t> expectedShape = {3, 2};
    const std::vector<float> expectedValue = {3.0315375, 5.201451,  2.7751598,
                                              10.753343, 15.107756, 2.909068};
    CheckReduce(ReduceType::kReduceL1, inputShape, inputData, expectedShape, expectedValue, {1});
}

TEST_F(ReduceTests, ReduceL1Axes2NotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0.9762701, 4.303787,   2.0552676,  0.89766365,
                                          -1.526904, 2.9178822,  -1.2482557, 7.83546,
                                          9.273255,  -2.3311696, 5.834501,   0.5778984};
    const std::vector<int32_t> expectedShape = {3, 2};
    const std::vector<float> expectedValue = {5.2800574, 2.9529312, 4.444786,
                                              9.083715,  11.604425, 6.4123993};
    CheckReduce(ReduceType::kReduceL1, inputShape, inputData, expectedShape, expectedValue, {2});
}

TEST_F(ReduceTests, ReduceL1NegativeAxesNotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0.9762701, 4.303787,   2.0552676,  0.89766365,
                                          -1.526904, 2.9178822,  -1.2482557, 7.83546,
                                          9.273255,  -2.3311696, 5.834501,   0.5778984};
    const std::vector<int32_t> expectedShape = {3, 2};
    const std::vector<float> expectedValue = {5.2800574, 2.9529312, 4.444786,
                                              9.083715,  11.604425, 6.4123993};
    CheckReduce(ReduceType::kReduceL1, inputShape, inputData, expectedShape, expectedValue, {-1});
}

TEST_F(ReduceTests, ReduceL1Axes0KeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0.9762701, 4.303787,   2.0552676,  0.89766365,
                                          -1.526904, 2.9178822,  -1.2482557, 7.83546,
                                          9.273255,  -2.3311696, 5.834501,   0.5778984};
    const std::vector<int32_t> expectedShape = {1, 2, 2};
    const std::vector<float> expectedValue = {11.776429, 9.552839, 9.138024, 9.311022};
    CheckReduce(ReduceType::kReduceL1, inputShape, inputData, expectedShape, expectedValue, {0},
                true);
}

TEST_F(ReduceTests, ReduceL1Axes1KeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0.9762701, 4.303787,   2.0552676,  0.89766365,
                                          -1.526904, 2.9178822,  -1.2482557, 7.83546,
                                          9.273255,  -2.3311696, 5.834501,   0.5778984};
    const std::vector<int32_t> expectedShape = {3, 1, 2};
    const std::vector<float> expectedValue = {3.0315375, 5.201451,  2.7751598,
                                              10.753343, 15.107756, 2.909068};
    CheckReduce(ReduceType::kReduceL1, inputShape, inputData, expectedShape, expectedValue, {1},
                true);
}

TEST_F(ReduceTests, ReduceL1Axes2KeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0.9762701, 4.303787,   2.0552676,  0.89766365,
                                          -1.526904, 2.9178822,  -1.2482557, 7.83546,
                                          9.273255,  -2.3311696, 5.834501,   0.5778984};
    const std::vector<int32_t> expectedShape = {3, 2, 1};
    const std::vector<float> expectedValue = {5.2800574, 2.9529312, 4.444786,
                                              9.083715,  11.604425, 6.4123993};
    CheckReduce(ReduceType::kReduceL1, inputShape, inputData, expectedShape, expectedValue, {2},
                true);
}

TEST_F(ReduceTests, ReduceL1NegativeAxesKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0.9762701, 4.303787,   2.0552676,  0.89766365,
                                          -1.526904, 2.9178822,  -1.2482557, 7.83546,
                                          9.273255,  -2.3311696, 5.834501,   0.5778984};
    const std::vector<int32_t> expectedShape = {3, 2, 1};
    const std::vector<float> expectedValue = {5.2800574, 2.9529312, 4.444786,
                                              9.083715,  11.604425, 6.4123993};
    CheckReduce(ReduceType::kReduceL1, inputShape, inputData, expectedShape, expectedValue, {-1},
                true);
}

TEST_F(ReduceTests, ReduceL2Default) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0.9762701, 4.303787,   2.0552676,  0.89766365,
                                          -1.526904, 2.9178822,  -1.2482557, 7.83546,
                                          9.273255,  -2.3311696, 5.834501,   0.5778984};
    const std::vector<int32_t> expectedShape = {};
    const std::vector<float> expectedValue = {14.970192};
    CheckReduce(ReduceType::kReduceL2, inputShape, inputData, expectedShape, expectedValue, {});
}

TEST_F(ReduceTests, ReduceL2DefaultAxesKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0.9762701, 4.303787,   2.0552676,  0.89766365,
                                          -1.526904, 2.9178822,  -1.2482557, 7.83546,
                                          9.273255,  -2.3311696, 5.834501,   0.5778984};
    const std::vector<int32_t> expectedShape = {1, 1, 1};
    const std::vector<float> expectedValue = {14.970192};
    CheckReduce(ReduceType::kReduceL2, inputShape, inputData, expectedShape, expectedValue, {},
                true);
}

TEST_F(ReduceTests, ReduceL2Axes0NotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0.9762701, 4.303787,   2.0552676,  0.89766365,
                                          -1.526904, 2.9178822,  -1.2482557, 7.83546,
                                          9.273255,  -2.3311696, 5.834501,   0.5778984};
    const std::vector<int32_t> expectedShape = {2, 2};
    const std::vector<float> expectedValue = {9.448693, 5.698331, 6.3106, 7.907857};
    CheckReduce(ReduceType::kReduceL2, inputShape, inputData, expectedShape, expectedValue, {0});
}

TEST_F(ReduceTests, ReduceL2Axes1NotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0.9762701, 4.303787,   2.0552676,  0.89766365,
                                          -1.526904, 2.9178822,  -1.2482557, 7.83546,
                                          9.273255,  -2.3311696, 5.834501,   0.5778984};
    const std::vector<int32_t> expectedShape = {3, 2};
    const std::vector<float> expectedValue = {2.2753522, 4.3964057, 1.9722013,
                                              8.361129,  10.956034, 2.4017324};
    CheckReduce(ReduceType::kReduceL2, inputShape, inputData, expectedShape, expectedValue, {1});
}

TEST_F(ReduceTests, ReduceL2Axes2NotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0.9762701, 4.303787,   2.0552676,  0.89766365,
                                          -1.526904, 2.9178822,  -1.2482557, 7.83546,
                                          9.273255,  -2.3311696, 5.834501,   0.5778984};
    const std::vector<int32_t> expectedShape = {3, 2};
    const std::vector<float> expectedValue = {4.413127, 2.2427495, 3.2932465,
                                              7.934266, 9.561779,  5.86305};
    CheckReduce(ReduceType::kReduceL2, inputShape, inputData, expectedShape, expectedValue, {2});
}

TEST_F(ReduceTests, ReduceL2NegativeAxesNotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0.9762701, 4.303787,   2.0552676,  0.89766365,
                                          -1.526904, 2.9178822,  -1.2482557, 7.83546,
                                          9.273255,  -2.3311696, 5.834501,   0.5778984};
    const std::vector<int32_t> expectedShape = {3, 2};
    const std::vector<float> expectedValue = {4.413127, 2.2427495, 3.2932465,
                                              7.934266, 9.561779,  5.863051};
    CheckReduce(ReduceType::kReduceL2, inputShape, inputData, expectedShape, expectedValue, {-1});
}

TEST_F(ReduceTests, ReduceL2Axes0KeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0.9762701, 4.303787,   2.0552676,  0.89766365,
                                          -1.526904, 2.9178822,  -1.2482557, 7.83546,
                                          9.273255,  -2.3311696, 5.834501,   0.5778984};
    const std::vector<int32_t> expectedShape = {1, 2, 2};
    const std::vector<float> expectedValue = {9.448693, 5.698331, 6.3106, 7.907857};
    CheckReduce(ReduceType::kReduceL2, inputShape, inputData, expectedShape, expectedValue, {0},
                true);
}

TEST_F(ReduceTests, ReduceL2Axes1KeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0.9762701, 4.303787,   2.0552676,  0.89766365,
                                          -1.526904, 2.9178822,  -1.2482557, 7.83546,
                                          9.273255,  -2.3311696, 5.834501,   0.5778984};
    const std::vector<int32_t> expectedShape = {3, 1, 2};
    const std::vector<float> expectedValue = {2.2753522, 4.3964057, 1.9722013,
                                              8.361129,  10.956034, 2.4017324};
    CheckReduce(ReduceType::kReduceL2, inputShape, inputData, expectedShape, expectedValue, {1},
                true);
}

TEST_F(ReduceTests, ReduceL2Axes2KeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0.9762701, 4.303787,   2.0552676,  0.89766365,
                                          -1.526904, 2.9178822,  -1.2482557, 7.83546,
                                          9.273255,  -2.3311696, 5.834501,   0.5778984};
    const std::vector<int32_t> expectedShape = {3, 2, 1};
    const std::vector<float> expectedValue = {4.413127, 2.2427495, 3.2932465,
                                              7.934266, 9.561779,  5.863051};
    CheckReduce(ReduceType::kReduceL2, inputShape, inputData, expectedShape, expectedValue, {2},
                true);
}

TEST_F(ReduceTests, ReduceL2NegativeAxesKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0.9762701, 4.303787,   2.0552676,  0.89766365,
                                          -1.526904, 2.9178822,  -1.2482557, 7.83546,
                                          9.273255,  -2.3311696, 5.834501,   0.5778984};
    const std::vector<int32_t> expectedShape = {3, 2, 1};
    const std::vector<float> expectedValue = {4.413127, 2.2427495, 3.2932465,
                                              7.934266, 9.561779,  5.863051};
    CheckReduce(ReduceType::kReduceL2, inputShape, inputData, expectedShape, expectedValue, {-1},
                true);
}

TEST_F(ReduceTests, ReduceMaxDefault) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {1., 100., 200., 2., 300., 3.,
                                          4., 400., 500., 5., 600., 6.};
    const std::vector<int32_t> expectedShape = {};
    const std::vector<float> expectedValue = {600};
    CheckReduce(ReduceType::kReduceMax, inputShape, inputData, expectedShape, expectedValue, {});
}

TEST_F(ReduceTests, ReduceMaxDefaultAxesKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {1., 100., 200., 2., 300., 3.,
                                          4., 400., 500., 5., 600., 6.};
    const std::vector<int32_t> expectedShape = {1, 1, 1};
    const std::vector<float> expectedValue = {600};
    CheckReduce(ReduceType::kReduceMax, inputShape, inputData, expectedShape, expectedValue, {},
                true);
}

TEST_F(ReduceTests, ReduceMaxAxes0NotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {1., 100., 200., 2., 300., 3.,
                                          4., 400., 500., 5., 600., 6.};
    const std::vector<int32_t> expectedShape = {2, 2};
    const std::vector<float> expectedValue = {500., 100., 600., 400.};
    CheckReduce(ReduceType::kReduceMax, inputShape, inputData, expectedShape, expectedValue, {0});
}

TEST_F(ReduceTests, ReduceMaxAxes1NotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {1., 100., 200., 2., 300., 3.,
                                          4., 400., 500., 5., 600., 6.};
    const std::vector<int32_t> expectedShape = {3, 2};
    const std::vector<float> expectedValue = {200., 100., 300., 400., 600., 6.};
    CheckReduce(ReduceType::kReduceMax, inputShape, inputData, expectedShape, expectedValue, {1});
}

TEST_F(ReduceTests, ReduceMaxAxes2NotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {1., 100., 200., 2., 300., 3.,
                                          4., 400., 500., 5., 600., 6.};
    const std::vector<int32_t> expectedShape = {3, 2};
    const std::vector<float> expectedValue = {100., 200., 300., 400., 500., 600.};
    CheckReduce(ReduceType::kReduceMax, inputShape, inputData, expectedShape, expectedValue, {2});
}

TEST_F(ReduceTests, ReduceMaxNegativeAxesNotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {1., 100., 200., 2., 300., 3.,
                                          4., 400., 500., 5., 600., 6.};
    const std::vector<int32_t> expectedShape = {3, 2};
    const std::vector<float> expectedValue = {100., 200., 300., 400., 500., 600.};
    CheckReduce(ReduceType::kReduceMax, inputShape, inputData, expectedShape, expectedValue, {-1});
}

TEST_F(ReduceTests, ReduceMaxAxes0KeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {1., 100., 200., 2., 300., 3.,
                                          4., 400., 500., 5., 600., 6.};
    const std::vector<int32_t> expectedShape = {1, 2, 2};
    const std::vector<float> expectedValue = {500., 100., 600., 400.};
    CheckReduce(ReduceType::kReduceMax, inputShape, inputData, expectedShape, expectedValue, {0},
                true);
}

TEST_F(ReduceTests, ReduceMaxAxes1KeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {1., 100., 200., 2., 300., 3.,
                                          4., 400., 500., 5., 600., 6.};
    const std::vector<int32_t> expectedShape = {3, 1, 2};
    const std::vector<float> expectedValue = {200., 100., 300., 400., 600., 6.};
    CheckReduce(ReduceType::kReduceMax, inputShape, inputData, expectedShape, expectedValue, {1},
                true);
}

TEST_F(ReduceTests, ReduceMaxAxes2KeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {1., 100., 200., 2., 300., 3.,
                                          4., 400., 500., 5., 600., 6.};
    const std::vector<int32_t> expectedShape = {3, 2, 1};
    const std::vector<float> expectedValue = {100., 200., 300., 400., 500., 600.};
    CheckReduce(ReduceType::kReduceMax, inputShape, inputData, expectedShape, expectedValue, {2},
                true);
}

TEST_F(ReduceTests, ReduceMaxNegativeAxesKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {1., 100., 200., 2., 300., 3.,
                                          4., 400., 500., 5., 600., 6.};
    const std::vector<int32_t> expectedShape = {3, 2, 1};
    const std::vector<float> expectedValue = {100., 200., 300., 400., 500., 600.};
    CheckReduce(ReduceType::kReduceMax, inputShape, inputData, expectedShape, expectedValue, {-1},
                true);
}

TEST_F(ReduceTests, ReduceMeanDefault) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.};
    const std::vector<int32_t> expectedShape = {};
    const std::vector<float> expectedValue = {18.25};
    CheckReduce(ReduceType::kReduceMean, inputShape, inputData, expectedShape, expectedValue, {});
}

TEST_F(ReduceTests, ReduceMeanDefaultAxesKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.};
    const std::vector<int32_t> expectedShape = {1, 1, 1};
    const std::vector<float> expectedValue = {18.25};
    CheckReduce(ReduceType::kReduceMean, inputShape, inputData, expectedShape, expectedValue, {},
                true);
}

TEST_F(ReduceTests, ReduceMeanAxes0NotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.};
    const std::vector<int32_t> expectedShape = {2, 2};
    const std::vector<float> expectedValue = {30., 1., 40., 2.};
    CheckReduce(ReduceType::kReduceMean, inputShape, inputData, expectedShape, expectedValue, {0});
}

TEST_F(ReduceTests, ReduceMeanAxes1NotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.};
    const std::vector<int32_t> expectedShape = {3, 2};
    const std::vector<float> expectedValue = {12.5, 1.5, 35., 1.5, 57.5, 1.5};
    CheckReduce(ReduceType::kReduceMean, inputShape, inputData, expectedShape, expectedValue, {1});
}

TEST_F(ReduceTests, ReduceMeanAxes2NotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.};
    const std::vector<int32_t> expectedShape = {3, 2};
    const std::vector<float> expectedValue = {3., 11., 15.5, 21., 28., 31.};
    CheckReduce(ReduceType::kReduceMean, inputShape, inputData, expectedShape, expectedValue, {2});
}

TEST_F(ReduceTests, ReduceMeanNegativeAxesNotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.};
    const std::vector<int32_t> expectedShape = {3, 2};
    const std::vector<float> expectedValue = {3., 11., 15.5, 21., 28., 31.};
    CheckReduce(ReduceType::kReduceMean, inputShape, inputData, expectedShape, expectedValue, {-1});
}

TEST_F(ReduceTests, ReduceMeanAxes0KeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.};
    const std::vector<int32_t> expectedShape = {1, 2, 2};
    const std::vector<float> expectedValue = {30., 1., 40., 2.};
    CheckReduce(ReduceType::kReduceMean, inputShape, inputData, expectedShape, expectedValue, {0},
                true);
}

TEST_F(ReduceTests, ReduceMeanAxes1KeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.};
    const std::vector<int32_t> expectedShape = {3, 1, 2};
    const std::vector<float> expectedValue = {12.5, 1.5, 35., 1.5, 57.5, 1.5};
    CheckReduce(ReduceType::kReduceMean, inputShape, inputData, expectedShape, expectedValue, {1},
                true);
}

TEST_F(ReduceTests, ReduceMeanAxes2KeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.};
    const std::vector<int32_t> expectedShape = {3, 2, 1};
    const std::vector<float> expectedValue = {3., 11., 15.5, 21., 28., 31.};
    CheckReduce(ReduceType::kReduceMean, inputShape, inputData, expectedShape, expectedValue, {2},
                true);
}

TEST_F(ReduceTests, ReduceMeanNegativeAxesKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.};
    const std::vector<int32_t> expectedShape = {3, 2, 1};
    const std::vector<float> expectedValue = {3., 11., 15.5, 21., 28., 31.};
    CheckReduce(ReduceType::kReduceMean, inputShape, inputData, expectedShape, expectedValue, {-1},
                true);
}

TEST_F(ReduceTests, ReduceMinDefault) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {1., 100., 200., 2., 300., 3.,
                                          4., 400., 500., 5., 600., 6.};
    const std::vector<int32_t> expectedShape = {};
    const std::vector<float> expectedValue = {1};
    CheckReduce(ReduceType::kReduceMin, inputShape, inputData, expectedShape, expectedValue, {});
}

TEST_F(ReduceTests, ReduceMinDefaultAxesKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {1., 100., 200., 2., 300., 3.,
                                          4., 400., 500., 5., 600., 6.};
    const std::vector<int32_t> expectedShape = {1, 1, 1};
    const std::vector<float> expectedValue = {1};
    CheckReduce(ReduceType::kReduceMin, inputShape, inputData, expectedShape, expectedValue, {},
                true);
}

TEST_F(ReduceTests, ReduceMinAxes0NotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {1., 100., 200., 2., 300., 3.,
                                          4., 400., 500., 5., 600., 6.};
    const std::vector<int32_t> expectedShape = {2, 2};
    const std::vector<float> expectedValue = {1., 3., 4., 2.};
    CheckReduce(ReduceType::kReduceMin, inputShape, inputData, expectedShape, expectedValue, {0});
}

TEST_F(ReduceTests, ReduceMinAxes1NotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {1., 100., 200., 2., 300., 3.,
                                          4., 400., 500., 5., 600., 6.};
    const std::vector<int32_t> expectedShape = {3, 2};
    const std::vector<float> expectedValue = {1., 2., 4., 3., 500., 5.};
    CheckReduce(ReduceType::kReduceMin, inputShape, inputData, expectedShape, expectedValue, {1});
}

TEST_F(ReduceTests, ReduceMinAxes2NotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {1., 100., 200., 2., 300., 3.,
                                          4., 400., 500., 5., 600., 6.};
    const std::vector<int32_t> expectedShape = {3, 2};
    const std::vector<float> expectedValue = {1., 2., 3., 4., 5., 6.};
    CheckReduce(ReduceType::kReduceMin, inputShape, inputData, expectedShape, expectedValue, {2});
}

TEST_F(ReduceTests, ReduceMinNegativeAxesNotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {1., 100., 200., 2., 300., 3.,
                                          4., 400., 500., 5., 600., 6.};
    const std::vector<int32_t> expectedShape = {3, 2};
    const std::vector<float> expectedValue = {1., 2., 3., 4., 5., 6.};
    CheckReduce(ReduceType::kReduceMin, inputShape, inputData, expectedShape, expectedValue, {-1});
}

TEST_F(ReduceTests, ReduceMinAxes0KeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {1., 100., 200., 2., 300., 3.,
                                          4., 400., 500., 5., 600., 6.};
    const std::vector<int32_t> expectedShape = {1, 2, 2};
    const std::vector<float> expectedValue = {1., 3., 4., 2.};
    CheckReduce(ReduceType::kReduceMin, inputShape, inputData, expectedShape, expectedValue, {0},
                true);
}

TEST_F(ReduceTests, ReduceMinAxes1KeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {1., 100., 200., 2., 300., 3.,
                                          4., 400., 500., 5., 600., 6.};
    const std::vector<int32_t> expectedShape = {3, 1, 2};
    const std::vector<float> expectedValue = {1., 2., 4., 3., 500., 5.};
    CheckReduce(ReduceType::kReduceMin, inputShape, inputData, expectedShape, expectedValue, {1},
                true);
}

TEST_F(ReduceTests, ReduceMinAxes2KeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {1., 100., 200., 2., 300., 3.,
                                          4., 400., 500., 5., 600., 6.};
    const std::vector<int32_t> expectedShape = {3, 2, 1};
    const std::vector<float> expectedValue = {1., 2., 3., 4., 5., 6.};
    CheckReduce(ReduceType::kReduceMin, inputShape, inputData, expectedShape, expectedValue, {2},
                true);
}

TEST_F(ReduceTests, ReduceMinNegativeAxesKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {1., 100., 200., 2., 300., 3.,
                                          4., 400., 500., 5., 600., 6.};
    const std::vector<int32_t> expectedShape = {3, 2, 1};
    const std::vector<float> expectedValue = {1., 2., 3., 4., 5., 6.};
    CheckReduce(ReduceType::kReduceMin, inputShape, inputData, expectedShape, expectedValue, {-1},
                true);
}

TEST_F(ReduceTests, ReduceProductDefault) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.};
    const std::vector<int32_t> expectedShape = {};
    const std::vector<float> expectedValue = {0};
    CheckReduce(ReduceType::kReduceProduct, inputShape, inputData, expectedShape, expectedValue,
                {});
}

TEST_F(ReduceTests, ReduceProductDefaultAxesKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.};
    const std::vector<int32_t> expectedShape = {1, 1, 1};
    const std::vector<float> expectedValue = {0};
    CheckReduce(ReduceType::kReduceProduct, inputShape, inputData, expectedShape, expectedValue, {},
                true);
}

TEST_F(ReduceTests, ReduceProductAxes0NotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.};
    const std::vector<int32_t> expectedShape = {2, 2};
    const std::vector<float> expectedValue = {0., 45., 120., 231.};
    CheckReduce(ReduceType::kReduceProduct, inputShape, inputData, expectedShape, expectedValue,
                {0});
}

TEST_F(ReduceTests, ReduceProductAxes1NotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.};
    const std::vector<int32_t> expectedShape = {3, 2};
    const std::vector<float> expectedValue = {0., 3., 24., 35., 80., 99.};
    CheckReduce(ReduceType::kReduceProduct, inputShape, inputData, expectedShape, expectedValue,
                {1});
}

TEST_F(ReduceTests, ReduceProductAxes2NotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.};
    const std::vector<int32_t> expectedShape = {3, 2};
    const std::vector<float> expectedValue = {0., 6., 20., 42., 72., 110.};
    CheckReduce(ReduceType::kReduceProduct, inputShape, inputData, expectedShape, expectedValue,
                {2});
}

TEST_F(ReduceTests, ReduceProductNegativeAxesNotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.};
    const std::vector<int32_t> expectedShape = {3, 2};
    const std::vector<float> expectedValue = {0., 6., 20., 42., 72., 110.};
    CheckReduce(ReduceType::kReduceProduct, inputShape, inputData, expectedShape, expectedValue,
                {-1});
}

TEST_F(ReduceTests, ReduceProductAxes0KeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.};
    const std::vector<int32_t> expectedShape = {1, 2, 2};
    const std::vector<float> expectedValue = {0., 45., 120., 231.};
    CheckReduce(ReduceType::kReduceProduct, inputShape, inputData, expectedShape, expectedValue,
                {0}, true);
}

TEST_F(ReduceTests, ReduceProductAxes1KeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.};
    const std::vector<int32_t> expectedShape = {3, 1, 2};
    const std::vector<float> expectedValue = {0., 3., 24., 35., 80., 99.};
    CheckReduce(ReduceType::kReduceProduct, inputShape, inputData, expectedShape, expectedValue,
                {1}, true);
}

TEST_F(ReduceTests, ReduceProductAxes2KeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.};
    const std::vector<int32_t> expectedShape = {3, 2, 1};
    const std::vector<float> expectedValue = {0., 6., 20., 42., 72., 110.};
    CheckReduce(ReduceType::kReduceProduct, inputShape, inputData, expectedShape, expectedValue,
                {2}, true);
}

TEST_F(ReduceTests, ReduceProductNegativeAxesKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.};
    const std::vector<int32_t> expectedShape = {3, 2, 1};
    const std::vector<float> expectedValue = {0., 6., 20., 42., 72., 110.};
    CheckReduce(ReduceType::kReduceProduct, inputShape, inputData, expectedShape, expectedValue,
                {-1}, true);
}

TEST_F(ReduceTests, ReduceSumDefault) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.};
    const std::vector<int32_t> expectedShape = {};
    const std::vector<float> expectedValue = {66};
    CheckReduce(ReduceType::kReduceSum, inputShape, inputData, expectedShape, expectedValue, {});
}

TEST_F(ReduceTests, ReduceSumDefaultAxesKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.};
    const std::vector<int32_t> expectedShape = {1, 1, 1};
    const std::vector<float> expectedValue = {66};
    CheckReduce(ReduceType::kReduceSum, inputShape, inputData, expectedShape, expectedValue, {},
                true);
}

TEST_F(ReduceTests, ReduceSumAxes0NotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.};
    const std::vector<int32_t> expectedShape = {2, 2};
    const std::vector<float> expectedValue = {12., 15., 18., 21.};
    CheckReduce(ReduceType::kReduceSum, inputShape, inputData, expectedShape, expectedValue, {0});
}

TEST_F(ReduceTests, ReduceSumAxes1NotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.};
    const std::vector<int32_t> expectedShape = {3, 2};
    const std::vector<float> expectedValue = {2., 4., 10., 12., 18., 20.};
    CheckReduce(ReduceType::kReduceSum, inputShape, inputData, expectedShape, expectedValue, {1});
}

TEST_F(ReduceTests, ReduceSumAxes2NotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.};
    const std::vector<int32_t> expectedShape = {3, 2};
    const std::vector<float> expectedValue = {1., 5., 9., 13., 17., 21.};
    CheckReduce(ReduceType::kReduceSum, inputShape, inputData, expectedShape, expectedValue, {2});
}

TEST_F(ReduceTests, ReduceSumNegativeAxesNotKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.};
    const std::vector<int32_t> expectedShape = {3, 2};
    const std::vector<float> expectedValue = {1., 5., 9., 13., 17., 21.};
    CheckReduce(ReduceType::kReduceSum, inputShape, inputData, expectedShape, expectedValue, {-1});
}

TEST_F(ReduceTests, ReduceSumAxes0KeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.};
    const std::vector<int32_t> expectedShape = {1, 2, 2};
    const std::vector<float> expectedValue = {12., 15., 18., 21.};
    CheckReduce(ReduceType::kReduceSum, inputShape, inputData, expectedShape, expectedValue, {0},
                true);
}

TEST_F(ReduceTests, ReduceSumAxes1KeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.};
    const std::vector<int32_t> expectedShape = {3, 1, 2};
    const std::vector<float> expectedValue = {2., 4., 10., 12., 18., 20.};
    CheckReduce(ReduceType::kReduceSum, inputShape, inputData, expectedShape, expectedValue, {1},
                true);
}

TEST_F(ReduceTests, ReduceSumAxes2KeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.};
    const std::vector<int32_t> expectedShape = {3, 2, 1};
    const std::vector<float> expectedValue = {1., 5., 9., 13., 17., 21};
    CheckReduce(ReduceType::kReduceSum, inputShape, inputData, expectedShape, expectedValue, {2},
                true);
}

TEST_F(ReduceTests, ReduceSumNegativeAxesKeepDims) {
    const std::vector<int32_t> inputShape = {3, 2, 2};
    const std::vector<float> inputData = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.};
    const std::vector<int32_t> expectedShape = {3, 2, 1};
    const std::vector<float> expectedValue = {1., 5., 9., 13., 17., 21.};
    CheckReduce(ReduceType::kReduceSum, inputShape, inputData, expectedShape, expectedValue, {-1},
                true);
}
