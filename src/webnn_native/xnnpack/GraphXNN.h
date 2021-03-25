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

#ifndef WEBNN_NATIVE_XNNPACK_GRAPH_XNN_H_
#define WEBNN_NATIVE_XNNPACK_GRAPH_XNN_H_

#include <map>
#include <set>

#include <xnnpack.h>

#include "webnn_native/Graph.h"
#include "webnn_native/Operand.h"
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
#include "webnn_native/xnnpack/ContextXNN.h"

namespace webnn_native { namespace xnnpack {

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
        void CompileImpl(BuildGraphCallbackDelegate delegate) override;
        void ComputeImpl(NamedInputsBase* inputs,
                         MLComputeGraphCallback callback,
                         void* userdata,
                         NamedOutputsBase* outputs = nullptr) override;

        MLBuildGraphStatus CompileSyncImpl() override;
        MLComputeGraphStatus ComputeSyncImpl(NamedInputsBase* inputs,
                                             NamedOutputsBase* outputs) override;

        MLComputeGraphStatus GenericComputeImpl(NamedInputsBase* inputs,
                                                NamedOutputsBase* outputs,
                                                MLComputeGraphCallback callback = nullptr,
                                                void* userdata = nullptr);

        enum OperandType { INPUT, CONSTANT, BINARY, CLAMP, CONV2D, POOL2D, UNARY };
        struct OperandInfo {
            OperandInfo(OperandType opType) : opType(opType) {
            }
            OperandType opType;
            std::string name = "";
            xnn_datatype dataType = xnn_datatype_invalid;
            std::vector<size_t> dims = {};
            std::unique_ptr<char> buffer = nullptr;
        };

        pthreadpool_t GetThreadpool();
        size_t SizeOfOperandInfo(const std::shared_ptr<OperandInfo>& info);
        xnn_status CreateBuffer(std::shared_ptr<OperandInfo>& info,
                                const void* data = nullptr,
                                size_t length = 0);
        xnn_status CreateXnnOp(const op::Unary* unary);
        xnn_status CreateXnnOp(const op::Clamp* clamp);
        xnn_status CreateXnnOp(const op::Binary* binary);
        xnn_status CreateXnnOp(const op::Pool2d* pool2d);
        xnn_status CreateXnnOp(const op::Conv2d* conv2d,
                               const op::Binary* add = nullptr,
                               const op::Clamp* clamp = nullptr);

        enum XnnOpType {
            add_nd_f32,
            clamp_nc_f32,
            multiply_nd_f32,
            subtract_nd_f32,
            convolution2d_nhwc_f32,
            average_pooling2d_nhwc_f32,
            max_pooling2d_nhwc_f32
        };
        XnnOpType mXnnOperatorType;
        xnn_operator_t mXnnOperator;
        std::vector<std::shared_ptr<OperandInfo>> mConstants;
        std::vector<std::shared_ptr<OperandInfo>> mInputs;
        std::vector<std::shared_ptr<OperandInfo>> mOutputs;
        std::map<std::string, uint32_t> mExternalInputs;
        std::map<std::string, uint32_t> mExternalOutputs;

        // For graph building
        std::vector<const OperandBase*> mOperandsToBuild;
        std::map<const OperandBase*, std::shared_ptr<OperandInfo>> mOperandInfoMap;
    };

}}  // namespace webnn_native::xnnpack

#endif  // WEBNN_NATIVE_XNNPACK_GRAPH_XNN_H_
