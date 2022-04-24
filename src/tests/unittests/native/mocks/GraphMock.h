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

#ifndef TESTS_UNITTESTS_NATIVE_MOCKS_GRAPH_MOCK_H_
#define TESTS_UNITTESTS_NATIVE_MOCKS_GRAPH_MOCK_H_

#include "webnn_native/Graph.h"

#include <gmock/gmock.h>

namespace webnn::native {

    class GraphMock : public GraphBase {
      public:
        GraphMock() : GraphBase(nullptr) {
        }
        ~GraphMock() override = default;

        MOCK_METHOD(MaybeError, AddConstant, (const op::Constant* constant), (override));
        MOCK_METHOD(MaybeError, AddInput, (const op::Input* input), (override));
        MOCK_METHOD(MaybeError,
                    AddOutput,
                    (std::string_view name, const OperandBase* output),
                    (override));
        MOCK_METHOD(MaybeError, AddBatchNorm, (const op::BatchNorm* batchNorm), (override));
        MOCK_METHOD(MaybeError, AddBinary, (const op::Binary* binary), (override));
        MOCK_METHOD(MaybeError, AddConv2d, (const op::Conv2d* conv2d), (override));
        MOCK_METHOD(MaybeError, AddGru, (const op::Gru* gru), (override));
        MOCK_METHOD(MaybeError, AddPad, (const op::Pad* pad), (override));
        MOCK_METHOD(MaybeError, AddPool2d, (const op::Pool2d* pool2d), (override));
        MOCK_METHOD(MaybeError, AddReduce, (const op::Reduce* reduce), (override));
        MOCK_METHOD(MaybeError, AddResample2d, (const op::Resample2d* resample2d), (override));
        MOCK_METHOD(MaybeError, AddReshape, (const op::Reshape* reshape), (override));
        MOCK_METHOD(MaybeError, AddSqueeze, (const op::Squeeze* squeeze), (override));
        MOCK_METHOD(MaybeError, AddSlice, (const op::Slice* slice), (override));
        MOCK_METHOD(MaybeError, AddSplit, (const op::Split* split), (override));
        MOCK_METHOD(MaybeError, AddTranspose, (const op::Transpose* transpose), (override));
        MOCK_METHOD(MaybeError, AddUnary, (const op::Unary* unary), (override));
        MOCK_METHOD(MaybeError, AddConcat, (const op::Concat* concat), (override));
        MOCK_METHOD(MaybeError, AddGemm, (const op::Gemm* gemm), (override));
        MOCK_METHOD(MaybeError, AddClamp, (const op::Clamp* clamp), (override));
        MOCK_METHOD(MaybeError,
                    AddInstanceNorm,
                    (const op::InstanceNorm* instanceNorm),
                    (override));
        MOCK_METHOD(MaybeError, Finish, (), (override));
        MOCK_METHOD(MaybeError, CompileImpl, (), (override));
        MOCK_METHOD(MaybeError,
                    ComputeImpl,
                    (NamedInputsBase * inputs, NamedOutputsBase* outputs),
                    (override));
    };

}  // namespace webnn::native
#endif  // TESTS_UNITTESTS_NATIVE_MOCKS_GRAPH_MOCK_H_
