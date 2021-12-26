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

#ifndef WEBNN_NATIVE_ONEDNN_MODEL_DNNL_H_
#define WEBNN_NATIVE_ONEDNN_MODEL_DNNL_H_

#include <map>
#include <set>

#include <dnnl.h>

#include "webnn_native/Graph.h"
#include "webnn_native/Operand.h"
#include "webnn_native/onednn/ContextDNNL.h"
#include "webnn_native/ops/Binary.h"
#include "webnn_native/ops/Clamp.h"
#include "webnn_native/ops/Constant.h"
#include "webnn_native/ops/Conv2d.h"
#include "webnn_native/ops/Input.h"
#include "webnn_native/ops/Pool2d.h"
#include "webnn_native/ops/Reshape.h"
#include "webnn_native/ops/Transpose.h"
#include "webnn_native/ops/Unary.h"

namespace webnn_native { namespace onednn {

    class Graph : public GraphBase {
      public:
        explicit Graph(Context* context);
        ~Graph() override;

        virtual MaybeError AddConstant(const op::Constant* constant) override;
        virtual MaybeError AddInput(const op::Input* input) override;
        virtual MaybeError AddOutput(const std::string& name, const OperandBase* output) override;
        virtual MaybeError AddBinary(const op::Binary* binary) override;
        virtual MaybeError AddConv2d(const op::Conv2d* conv2d) override;
        virtual MaybeError AddPool2d(const op::Pool2d* pool2d) override;
        virtual MaybeError AddUnary(const op::Unary* unary) override;
        virtual MaybeError AddClamp(const op::Clamp* clamp) override;
        virtual MaybeError Finish() override;

      private:
        dnnl_status_t AddConv2dImpl(const op::Conv2d* conv2d,
                                    const op::Binary* add = nullptr,
                                    const op::FusionClamp* clamp = nullptr);
        dnnl_status_t AddBinaryImpl(const op::Binary* binary);
        dnnl_status_t AddClampImpl(const op::Clamp* clamp);
        dnnl_status_t AddPool2dImpl(const op::Pool2d* pool2d);
        dnnl_status_t AddUnaryImpl(const op::Unary* unary);

        dnnl_status_t BuildPrimitives();

        MaybeError CompileImpl() override;
        MLComputeGraphStatus ComputeImpl(NamedInputsBase* inputs,
                                         NamedOutputsBase* outputs) override;
        dnnl_engine_t GetEngine();
        dnnl_status_t GetMemoryDesc(dnnl_memory_t memory, const dnnl_memory_desc_t** desc);
        dnnl_status_t ReorderIfNeeded(const dnnl_memory_desc_t* srcDesc,
                                      dnnl_memory_t srcMem,
                                      const dnnl_memory_desc_t* dstDesc,
                                      dnnl_memory_t* dstMem);
        dnnl_status_t ReorderToPlainFormat(dnnl_memory_t srcMem, dnnl_memory_t* dstMem);

        std::vector<dnnl_memory_t> mMemories;
        std::set<dnnl_memory_t> mConstantMemories;
        std::map<dnnl_memory_t, dnnl_memory_desc_t> mMemoryReinterprets;
        std::map<const OperandBase*, dnnl_memory_t> mOperandMemoryMap;
        std::map<std::string, dnnl_memory_t> mInputMemoryMap;
        std::map<std::string, dnnl_memory_t> mOutputMemoryMap;

        enum OperatorType { BINARY, CLAMP, CONV2D, POOL2D, UNARY };
        struct OperatorInfo {
            OperatorType opType;
            const OperatorBase* op;
        };
        // For op fusion
        std::vector<OperatorInfo> mOperandsToBuild;

        typedef struct {
            dnnl_primitive_t primitive;
            std::vector<dnnl_exec_arg_t> args;
        } Operation;

        std::vector<Operation> mOperations;

        dnnl_stream_t mStream;
    };

}}  // namespace webnn_native::onednn

#endif  // WEBNN_NATIVE_ONEDNN_MODEL_DNNL_H_
