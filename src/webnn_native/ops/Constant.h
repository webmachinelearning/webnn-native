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

#ifndef WEBNN_NATIVE_OPS_CONSTANT_H_
#define WEBNN_NATIVE_OPS_CONSTANT_H_

#include "common/Assert.h"
#include "webnn_native/Graph.h"
#include "webnn_native/Operand.h"

namespace webnn_native::op {

    class Constant final : public OperatorBase {
      public:
        Constant(GraphBuilderBase* builder,
                 const OperandDescriptor* desc,
                 const ArrayBufferView* arrayBuffer)
            : OperatorBase(builder), mBuffer(nullptr) {
            if (desc == nullptr || arrayBuffer == nullptr) {
                return;
            }
            mDimensions.assign(desc->dimensions, desc->dimensions + desc->dimensionsCount);
            mDescriptor.dimensions = mDimensions.data();
            mDescriptor.dimensionsCount = mDimensions.size();
            mDescriptor.type = desc->type;
#if defined(WEBNN_ENABLE_WIRE)
            // Prevent destroy from allocator memory after handling the command.
            mBuffer = malloc(arrayBuffer->byteLength);
            memcpy(mBuffer, static_cast<int8_t*>(arrayBuffer->buffer) + arrayBuffer->byteOffset,
                   arrayBuffer->byteLength);
#else
            mBuffer = static_cast<int8_t*>(arrayBuffer->buffer) + arrayBuffer->byteOffset;
#endif  // defined(WEBNN_ENABLE_WIRE)
            mByteLength = arrayBuffer->byteLength;
        }

#if defined(WEBNN_ENABLE_GPU_BUFFER)
        Constant(GraphBuilderBase* builder,
                 const OperandDescriptor* desc,
                 const GpuBufferView* view)
            : OperatorBase(builder), mBuffer(nullptr), mWGPUBuffer(nullptr) {
            if (desc == nullptr || view == nullptr) {
                return;
            }
            mDimensions.assign(desc->dimensions, desc->dimensions + desc->dimensionsCount);
            mDescriptor.dimensions = mDimensions.data();
            mDescriptor.dimensionsCount = mDimensions.size();
            mDescriptor.type = desc->type;
            mWGPUBuffer = reinterpret_cast<WGPUBuffer>(view->buffer);
            wgpuBufferReference(mWGPUBuffer);
            mByteOffset = view->offset;
            mByteLength = view->size;
        }
#endif

        ~Constant() override {
            if (mBuffer)
                free(mBuffer);
#if defined(WEBNN_ENABLE_GPU_BUFFER)
            if (mWGPUBuffer)
                wgpuBufferReference(mWGPUBuffer);
#endif
        }

        MaybeError AddToGraph(GraphBase* graph) const override {
            return graph->AddConstant(this);
        }

        MaybeError ValidateAndInferOutputInfo() override {
            // if (mBuffer == nullptr || mByteLength == 0) {
            //     return DAWN_VALIDATION_ERROR("Constant array buffer is invalid.");
            // }
            mOutputs[0]->SetType(mDescriptor.type);
            mOutputs[0]->SetShape(mDimensions);
            return {};
        }

        const OperandDescriptor* GetOperandDescriptor() const {
            return &mDescriptor;
        }

        void const* GetBuffer() const {
            return mBuffer;
        }

#if defined(WEBNN_ENABLE_GPU_BUFFER)
        WGPUBuffer GetWGPUBuffer() const {
            return mWGPUBuffer;
        }
#endif

        size_t GetByteLength() const {
            return mByteLength;
        }

        size_t GetByteOffset() const {
            return mByteOffset;
        }

      private:
        OperandDescriptor mDescriptor;
        std::vector<int32_t> mDimensions;
        void* mBuffer;
#if defined(WEBNN_ENABLE_GPU_BUFFER)
        WGPUBuffer mWGPUBuffer;
#endif
        size_t mByteLength;
        size_t mByteOffset;
    };

}  // namespace webnn_native::op

#endif  // WEBNN_NATIVE_OPS_CONSTANT_H_
