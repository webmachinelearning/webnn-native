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

#ifndef WEBNN_NATIVE_MLAS_GRAPH_MLAS_H_
#define WEBNN_NATIVE_MLAS_GRAPH_MLAS_H_

#include <unordered_map>
#include <unordered_set>

#include "webnn_native/Graph.h"
#include "webnn_native/Operand.h"
#include "webnn_native/mlas/ContextMLAS.h"
#include "webnn_native/ops/Binary.h"
#include "webnn_native/ops/Clamp.h"
#include "webnn_native/ops/Constant.h"
#include "webnn_native/ops/Conv2d.h"
#include "webnn_native/ops/Input.h"
#include "webnn_native/ops/LeakyRelu.h"
#include "webnn_native/ops/Pool2d.h"
#include "webnn_native/ops/Reshape.h"
#include "webnn_native/ops/Transpose.h"
#include "webnn_native/ops/Unary.h"

namespace webnn_native::mlas {

    class Memory;
    class Kernel;
    class Conv2d;

    class Graph : public GraphBase {
      public:
        explicit Graph(Context* context);
        ~Graph() override;

        virtual MaybeError AddConstant(const op::Constant* constant) override;
        virtual MaybeError AddInput(const op::Input* input) override;
        virtual MaybeError AddOutput(const std::string& name, const OperandBase* output) override;
        virtual MaybeError AddBinary(const op::Binary* binary) override;
        virtual MaybeError AddClamp(const op::Clamp* clamp) override;
        virtual MaybeError AddConv2d(const op::Conv2d* conv2d) override;
        virtual MaybeError AddPool2d(const op::Pool2d* pool2d) override;
        virtual MaybeError AddUnary(const op::Unary* unary) override;
        virtual MaybeError Finish() override;

      private:
        MaybeError CompileImpl() override;
        WNNComputeGraphStatus ComputeImpl(NamedInputsBase* inputs,
                                          NamedOutputsBase* outputs) override;

        std::unordered_map<std::string, Ref<Memory>> mInputs;
        std::unordered_map<std::string, Ref<Memory>> mOutputs;
        std::unordered_map<const OperandBase*, Ref<Memory>> mMemoryMap;
        std::unordered_map<const OperatorBase*, Ref<Conv2d>> mConv2dKernels;
        std::vector<Ref<Kernel>> mKernels;
    };

}  // namespace webnn_native::mlas

#endif  // WEBNN_NATIVE_MLAS_GRAPH_MLAS_H_
