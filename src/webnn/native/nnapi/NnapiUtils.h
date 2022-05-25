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

#ifndef WEBNN_NATIVE_NNAPI_UTILS_H_
#define WEBNN_NATIVE_NNAPI_UTILS_H_
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>

#include "webnn/native/Error.h"
#include "webnn/native/Graph.h"
#include "webnn/native/Operand.h"
#include "webnn/native/Operator.h"

#include "NeuralNetworksTypes.h"
#include "nnapi_implementation.h"
#include "webnn/native/nnapi/ErrorNnapi.h"

namespace webnn::native::nnapi {

    struct NodeInfo {
        wnn::OperandType type;
        std::vector<uint32_t> dimensions;
        std::string name;
        uint32_t opIndex;

        NodeInfo() : opIndex(INT32_MAX) {
        }

        size_t getDimsSize() {
            return std::accumulate(std::begin(dimensions), std::end(dimensions), 1,
                                   std::multiplies<size_t>());
        }

        size_t GetByteCount() {
            size_t count = std::accumulate(std::begin(dimensions), std::end(dimensions), 1,
                                           std::multiplies<size_t>());
            switch (type) {
                case wnn::OperandType::Float32:
                case wnn::OperandType::Uint32:
                case wnn::OperandType::Int32:
                    count *= 4;
                    break;
                case wnn::OperandType::Float16:
                    count *= 2;
                    break;
                default:
                    UNREACHABLE();
            }
            return count;
        }
    };

    inline int32_t ConvertToNnapiType(wnn::OperandType type) {
        int32_t nnapiType;
        switch (type) {
            case wnn::OperandType::Float32:
                nnapiType = ANEURALNETWORKS_TENSOR_FLOAT32;
                break;
            case wnn::OperandType::Int32:
                nnapiType = ANEURALNETWORKS_TENSOR_INT32;
                break;
            case wnn::OperandType::Float16:
                nnapiType = ANEURALNETWORKS_TENSOR_FLOAT16;
                break;
            case wnn::OperandType::Uint32:
                nnapiType = ANEURALNETWORKS_UINT32;
                break;
            default:
                UNREACHABLE();
        }
        return nnapiType;
    }

    inline MaybeError GetTensorDesc(const std::shared_ptr<NodeInfo>& node,
                                    ANeuralNetworksOperandType& tensorType) {
        if (node->dimensions.size() == 0)
            return DAWN_INTERNAL_ERROR("Invalid dimensions !!");

        tensorType.dimensions = &(node->dimensions[0]);
        tensorType.dimensionCount = node->dimensions.size();
        tensorType.scale = 0.0f;
        tensorType.zeroPoint = 0;
        tensorType.type = ConvertToNnapiType(node->type);

        return {};
    }
} // namespace webnn::native::nnapi

#endif
