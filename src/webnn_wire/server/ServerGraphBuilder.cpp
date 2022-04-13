// Copyright 2019 The Webnn Authors
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

#include "webnn_wire/server/Server.h"

namespace webnn_wire::server {

    bool Server::DoGraphBuilderConstantInternal(ObjectId graphBuilderId,
                                                WNNOperandDescriptor const* desc,
                                                uint8_t const* buffer,
                                                size_t byteLength,
                                                size_t byteOffset,
                                                ObjectHandle result) {
        auto* graphBuilder = GraphBuilderObjects().Get(graphBuilderId);
        if (graphBuilder == nullptr) {
            return false;
        }

        // Create and register the operand object.
        auto* resultData = OperandObjects().Allocate(result.id);
        if (resultData == nullptr) {
            return false;
        }
        resultData->generation = result.generation;
        resultData->contextInfo = graphBuilder->contextInfo;
        if (resultData->contextInfo != nullptr) {
            if (!TrackContextChild(resultData->contextInfo, ObjectType::Operand, result.id)) {
                return false;
            }
        }
        WNNArrayBufferView value;
        value.buffer = const_cast<void*>(static_cast<const void*>(buffer));
        value.byteLength = byteLength;
        value.byteOffset = byteOffset;
        resultData->handle = mProcs.graphBuilderConstant(graphBuilder->handle, desc, &value);
        return true;
    }

    bool Server::DoGraphBuilderConstantWithGpuBufferInternal(ObjectId graphBuilderId,
                                                             WNNOperandDescriptor const* desc,
                                                             uint8_t const* buffer,
                                                             uint32_t id,
                                                             uint32_t generation,
                                                             size_t size,
                                                             size_t offset,
                                                             ObjectHandle result) {
        auto* graphBuilder = GraphBuilderObjects().Get(graphBuilderId);
        if (graphBuilder == nullptr) {
            return false;
        }

        // Create and register the operand object.
        auto* resultData = OperandObjects().Allocate(result.id);
        if (resultData == nullptr) {
            return false;
        }
        resultData->generation = result.generation;
        resultData->contextInfo = graphBuilder->contextInfo;
        if (resultData->contextInfo != nullptr) {
            if (!TrackContextChild(resultData->contextInfo, ObjectType::Operand, result.id)) {
                return false;
            }
        }

#if defined(WEBNN_ENABLE_GPU_BUFFER)
        WNNGpuBufferView value;
        value.buffer = GetWGPUBuffer(id, generation);
        value.id = id;
        value.generation = generation;
        value.size = size;
        value.offset = offset;
        resultData->handle =
            mProcs.graphBuilderConstantWithGpuBuffer(graphBuilder->handle, desc, &value);
#endif
        return true;
    }

    bool Server::DoGraphBuilderGruInternal(ObjectId graphBuilderId,
                                           ObjectId inputId,
                                           ObjectId weightId,
                                           ObjectId recurrentWeightId,
                                           int32_t steps,
                                           int32_t hiddenSize,
                                           WNNGruOptions const* options,
                                           ObjectHandle result) {
        auto* graphBuilder = GraphBuilderObjects().Get(graphBuilderId);
        auto* input = OperandObjects().Get(inputId);
        auto* weight = OperandObjects().Get(weightId);
        auto* recurrentWeight = OperandObjects().Get(recurrentWeightId);
        if (graphBuilder == nullptr || input == nullptr || weight == nullptr ||
            recurrentWeight == nullptr) {
            return false;
        }

        // Create and register the operandArray object.
        auto* resultData = OperandArrayObjects().Allocate(result.id);
        if (resultData == nullptr) {
            return false;
        }
        resultData->generation = result.generation;
        resultData->contextInfo = graphBuilder->contextInfo;
        if (resultData->contextInfo != nullptr) {
            if (!TrackContextChild(resultData->contextInfo, ObjectType::OperandArray, result.id)) {
                return false;
            }
        }
        resultData->handle =
            mProcs.graphBuilderGru(graphBuilder->handle, input->handle, weight->handle,
                                   recurrentWeight->handle, steps, hiddenSize, options);
        return true;
    }

    bool Server::DoGraphBuilderSplitInternal(ObjectId graphBuilderId,
                                             ObjectId inputId,
                                             uint32_t const* splits,
                                             uint32_t splitsCount,
                                             WNNSplitOptions const* options,
                                             ObjectHandle result) {
        auto* graphBuilder = GraphBuilderObjects().Get(graphBuilderId);
        auto* input = OperandObjects().Get(inputId);
        if (graphBuilder == nullptr || input == nullptr) {
            return false;
        }

        // Create and register the operandArray object.
        auto* resultData = OperandArrayObjects().Allocate(result.id);
        if (resultData == nullptr) {
            return false;
        }
        resultData->generation = result.generation;
        resultData->contextInfo = graphBuilder->contextInfo;
        if (resultData->contextInfo != nullptr) {
            if (!TrackContextChild(resultData->contextInfo, ObjectType::OperandArray, result.id)) {
                return false;
            }
        }
        resultData->handle = mProcs.graphBuilderSplit(graphBuilder->handle, input->handle, splits,
                                                      splitsCount, options);
        return true;
    }

}  // namespace webnn_wire::server
