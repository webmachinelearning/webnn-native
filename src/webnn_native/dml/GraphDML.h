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

#ifndef WEBNN_NATIVE_DML_MODEL_DML_H_
#define WEBNN_NATIVE_DML_MODEL_DML_H_

#include <map>
#include <mutex>
#include <set>
#include <unordered_set>

#include "webnn_native/Graph.h"
#include "webnn_native/Operand.h"
#include "webnn_native/Operator.h"
#include "webnn_native/dml/ContextDML.h"
#include "webnn_native/dml/deps/src/precomp.h"
#include "webnn_native/ops/BatchNorm.h"
#include "webnn_native/ops/Binary.h"
#include "webnn_native/ops/Clamp.h"
#include "webnn_native/ops/Concat.h"
#include "webnn_native/ops/Constant.h"
#include "webnn_native/ops/Conv2d.h"
#include "webnn_native/ops/Gemm.h"
#include "webnn_native/ops/Input.h"
#include "webnn_native/ops/InstanceNorm.h"
#include "webnn_native/ops/LeakyRelu.h"
#include "webnn_native/ops/Pad.h"
#include "webnn_native/ops/Pool2d.h"
#include "webnn_native/ops/Reduce.h"
#include "webnn_native/ops/Resample.h"
#include "webnn_native/ops/Reshape.h"
#include "webnn_native/ops/Split.h"
#include "webnn_native/ops/Squeeze.h"
#include "webnn_native/ops/Transpose.h"
#include "webnn_native/ops/Unary.h"

namespace webnn_native { namespace dml {

    std::string DmlTensorDimensionsToString(const ::dml::TensorDimensions&);
    std::string DmlTensorDataTypeToString(DML_TENSOR_DATA_TYPE type);

    class Graph : public GraphBase {
      public:
        explicit Graph(Context* context);
        ~Graph() override = default;

        virtual MaybeError AddConstant(const op::Constant* constant) override;
        virtual MaybeError AddInput(const op::Input* input) override;
        virtual MaybeError AddOutput(const std::string& name, const OperandBase* output) override;
        virtual MaybeError AddBatchNorm(const op::BatchNorm* batchNorm) override;
        virtual MaybeError AddBinary(const op::Binary* binary) override;
        virtual MaybeError AddConv2d(const op::Conv2d* conv2d) override;
        virtual MaybeError AddPad(const op::Pad* pad) override;
        virtual MaybeError AddPool2d(const op::Pool2d* pool2d) override;
        virtual MaybeError AddReduce(const op::Reduce* reduce) override;
        virtual MaybeError AddResample(const op::Resample* resample) override;
        virtual MaybeError AddReshape(const op::Reshape* reshape) override;
        virtual MaybeError AddSplit(const op::Split* split) override;
        virtual MaybeError AddSqueeze(const op::Squeeze* squeeze,
                                      std::vector<int32_t>& outputDims) override;
        virtual MaybeError AddTranspose(const op::Transpose* transpose) override;
        virtual MaybeError AddUnary(const op::Unary* unary) override;
        virtual MaybeError AddGemm(const op::Gemm* Gemm) override;
        virtual MaybeError AddConcat(const op::Concat* concat) override;
        virtual MaybeError AddClamp(const op::Clamp* clamp) override;
        virtual MaybeError AddInstanceNorm(const op::InstanceNorm* instanceNorm) override;
        virtual MaybeError Finish() override;

      private:
        MaybeError CompileImpl() override;
        MLComputeGraphStatus ComputeImpl(NamedInputsBase* inputs,
                                         NamedOutputsBase* outputs) override;

        ::dml::Expression BindingConstant(DML_TENSOR_DATA_TYPE dmlTensorType,
                                          ::dml::TensorDimensions dmlTensorDims,
                                          void const* value,
                                          size_t size);
        ::dml::Expression HardSwish(::dml::Expression& input);
        ::dml::Expression EmulateFusedActivation(OperatorBase* activation,
                                                 ::dml::Expression& input);

        std::shared_ptr<::pydml::Device> mDevice;
        // The mutex is used to lock mDevice.
        std::mutex mMutex;
        std::unique_ptr<::dml::Graph> mGraph;
        std::map<const OperandBase*, ::dml::Expression> mExpression;
        std::vector<std::unique_ptr<::pydml::Binding>> mBindings;
        std::vector<std::unique_ptr<char>> mConstantBuffers;
        std::unordered_set<const OperandBase*> mConstantSet;
        std::map<std::string, ::pydml::Binding*> mInputs;
        std::map<std::string, ::dml::Expression> mOutputs;
        std::unique_ptr<pydml::CompiledModel> mCompiledModel;
    };

}}  // namespace webnn_native::dml

#endif  // WEBNN_NATIVE_DML_MODEL_DML_H_
