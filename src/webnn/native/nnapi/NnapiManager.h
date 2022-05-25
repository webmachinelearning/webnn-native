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

#ifndef WEBNN_NATIVE_NN_MANAGER_H_
#define WEBNN_NATIVE_NN_MANAGER_H_

#include <unistd.h>
#include <map>
#include <set>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "NeuralNetworksTypes.h"
#include "NnapiUtils.h"
#include "nnapi_implementation.h"
#include "webnn/native/Error.h"
#include "webnn/native/nnapi/ErrorNnapi.h"

namespace webnn::native::nnapi {

    struct FdMem {
        int fd;
        ANeuralNetworksMemory* mem;
    };

    enum NNAPIComputeGraphStatus {
        NNAPIComputeGraphStatus_Error = 0,
        NNAPIComputeGraphStatus_Success = 1
    };
    class NnapiManager {
      public:
        explicit NnapiManager();

        ~NnapiManager() {
            for (auto const& element : mFdMemMap) {
                close(element.second.fd);
                mNnapi->ANeuralNetworksMemory_free(element.second.mem);
            }

            mNnapi->ANeuralNetworksCompilation_free(mNnCompilation);
            mNnapi->ANeuralNetworksModel_free(mNnModel);
        }

        MaybeError CreateOperandAndSetMemory(std::string name,
                                             const std::shared_ptr<NodeInfo>& node,
                                             const void* buffer);
        size_t SetInputMemory(int32_t index,
                              const ANeuralNetworksOperandType* type,
                              const ANeuralNetworksMemory* memory,
                              size_t offset,
                              size_t length);
        size_t SetOutputMemory(int32_t index,
                               const ANeuralNetworksOperandType* type,
                               const ANeuralNetworksMemory* memory,
                               size_t offset,
                               size_t length);
        MaybeError CreateScalarOperand(uint32_t type,
                                       const void* data,
                                       uint32_t& index,
                                       bool optional = false);
        MaybeError CreateInputOutputOperand(std::string name,
                                            const std::shared_ptr<NodeInfo>& node,
                                            bool input = true);
        MaybeError CreateOperand(const std::shared_ptr<NodeInfo>& node);
        MaybeError AddOperation(int32_t opCode,
                                size_t inputLen,
                                const uint32_t* input,
                                size_t outputLen,
                                const uint32_t* output);
        MaybeError SetVecOperand(int32_t index, const void* buffer, size_t length);
        MaybeError Compile(uint32_t inputCount,
                           const uint32_t* inputs,
                           uint32_t outputCount,
                           const uint32_t* outputs);
        NNAPIComputeGraphStatus ComputeAndWait();
        NNAPIComputeGraphStatus InitExecutionContext();
        void getFdNNMemory(uint32_t index, int& fd, ANeuralNetworksMemory*& mem) {
            fd = mFdMemMap[index].fd;
            mem = mFdMemMap[index].mem;
        }

      private:
        uint32_t GetOperandIdx() {
            return mOperandIndex++;
        }

        uint32_t mOperandIndex;
        const NnApi* mNnapi;
        ANeuralNetworksModel* mNnModel;
        ANeuralNetworksCompilation* mNnCompilation;
        ANeuralNetworksExecution* mNnExecution;
        ANeuralNetworksOperandType mInt32Operand, mBoolOperand, mFloat32Operand;
        std::map<uint32_t, struct FdMem> mFdMemMap;
    };
}  // namespace webnn::native::nnapi

#endif  // WEBNN_NATIVE_NN_MANAGER_H_
