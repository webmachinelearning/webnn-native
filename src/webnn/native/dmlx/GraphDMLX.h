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

#ifndef WEBNN_NATIVE_DMLX_GRAPH_DMLX_H_
#define WEBNN_NATIVE_DMLX_GRAPH_DMLX_H_

#include <map>
#include <mutex>
#include <set>
#include <unordered_set>

#include "webnn/native/Graph.h"
#include "webnn/native/Operand.h"
#include "webnn/native/Operator.h"
#include "webnn/native/dmlx/ContextDMLX.h"
#include "webnn/native/ops/BatchNorm.h"
#include "webnn/native/ops/Binary.h"
#include "webnn/native/ops/Clamp.h"
#include "webnn/native/ops/Concat.h"
#include "webnn/native/ops/Constant.h"
#include "webnn/native/ops/Conv2d.h"
#include "webnn/native/ops/Gemm.h"
#include "webnn/native/ops/Gru.h"
#include "webnn/native/ops/Input.h"
#include "webnn/native/ops/InstanceNorm.h"
#include "webnn/native/ops/LeakyRelu.h"
#include "webnn/native/ops/Pad.h"
#include "webnn/native/ops/Pool2d.h"
#include "webnn/native/ops/Reduce.h"
#include "webnn/native/ops/Resample2d.h"
#include "webnn/native/ops/Reshape.h"
#include "webnn/native/ops/Slice.h"
#include "webnn/native/ops/Split.h"
#include "webnn/native/ops/Squeeze.h"
#include "webnn/native/ops/Transpose.h"
#include "webnn/native/ops/Unary.h"

#if defined(WEBNN_ENABLE_GPU_BUFFER)
// Support DirectMLX
#    include <initguid.h>
#    include <wrl/client.h>
#    include <wrl/implements.h>

#    include <Windows.h>
#    include <d3d12.h>

#    define DML_TARGET_VERSION_USE_LATEST 1
#    include <DirectML.h>
#    include "DirectMLX.h"

#    include "dawn/native/dml/deps/src/dmldevice.h"
#else
#    include "webnn/native/dmlx/deps/src/precomp.h"
#endif

namespace webnn::native::dmlx {

    std::string DmlTensorDimensionsToString(const ::dml::TensorDimensions&);
    std::string DmlTensorDataTypeToString(DML_TENSOR_DATA_TYPE type);

    class Graph : public GraphBase {
      public:
        explicit Graph(Context* context);
        ~Graph() override = default;

        virtual MaybeError AddConstant(const op::Constant* constant) override;
        virtual MaybeError AddInput(const op::Input* input) override;
        virtual MaybeError AddOutput(std::string_view name, const OperandBase* output) override;
        virtual MaybeError AddBatchNorm(const op::BatchNorm* batchNorm) override;
        virtual MaybeError AddBinary(const op::Binary* binary) override;
        virtual MaybeError AddConv2d(const op::Conv2d* conv2d) override;
        virtual MaybeError AddConvTranspose2d(const op::ConvTranspose2d* convTranspose2d) override;
        virtual MaybeError AddPad(const op::Pad* pad) override;
        virtual MaybeError AddPool2d(const op::Pool2d* pool2d) override;
        virtual MaybeError AddReduce(const op::Reduce* reduce) override;
        virtual MaybeError AddResample2d(const op::Resample2d* resample2d) override;
        virtual MaybeError AddReshape(const op::Reshape* reshape) override;
        virtual MaybeError AddSlice(const op::Slice* slice) override;
        virtual MaybeError AddSplit(const op::Split* split) override;
        virtual MaybeError AddSqueeze(const op::Squeeze* squeeze) override;
        virtual MaybeError AddTranspose(const op::Transpose* transpose) override;
        virtual MaybeError AddUnary(const op::Unary* unary) override;
        virtual MaybeError AddGemm(const op::Gemm* Gemm) override;
        virtual MaybeError AddGru(const op::Gru* Gru) override;
        virtual MaybeError AddConcat(const op::Concat* concat) override;
        virtual MaybeError AddClamp(const op::Clamp* clamp) override;
        virtual MaybeError AddInstanceNorm(const op::InstanceNorm* instanceNorm) override;
        virtual MaybeError Finish() override;

        virtual MaybeError ComputeImpl(NamedInputsBase* inputs, NamedOutputsBase* outputs) override;

      private:
        MaybeError CompileImpl() override;

        ::dml::Expression BindingConstant(DML_TENSOR_DATA_TYPE dmlTensorType,
                                          ::dml::TensorDimensions dmlTensorDims,
                                          void const* value,
                                          size_t size
#ifdef WEBNN_ENABLE_GPU_BUFFER
                                          ,
                                          WGPUBuffer wgpuBuffer = nullptr
#endif
        );
        ::dml::Expression HardSwish(::dml::Expression& input);
        ::dml::Expression EmulateFusedActivation(FusionOperatorBase* activation,
                                                 ::dml::Expression& input);

        std::shared_ptr<::pydml::Device> mDevice;
        // The mutex is used to lock mDevice.
        std::mutex mMutex;
        std::unique_ptr<::dml::Graph> mGraph;
        std::map<const OperandBase*, ::dml::Expression> mExpression;
        std::vector<std::unique_ptr<::pydml::Binding>> mInputBindings;
        std::map<std::string, ::pydml::Binding*> mInputBindingMap;
        std::vector<std::unique_ptr<char>> mConstantBuffers;
        std::unordered_set<const OperandBase*> mConstantSet;
        std::vector<Ref<OperandBase>> mConstants;
        std::map<std::string, ::dml::Expression> mOutputExpressionMap;
        std::vector<std::unique_ptr<::pydml::Binding>> mOutputBindings;
        std::map<std::string, ::pydml::Binding*> mOutputBindingMap;
        std::unique_ptr<pydml::CompiledModel> mCompiledModel;
    };

}  // namespace webnn::native::dmlx

#endif  // WEBNN_NATIVE_DMLX_GRAPH_DMLX_H_
