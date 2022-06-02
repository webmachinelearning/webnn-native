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

#ifndef WEBNN_NATIVE_NAMED_INPUTS_H_
#define WEBNN_NATIVE_NAMED_INPUTS_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "common/Log.h"
#include "webnn/native/webnn_platform.h"

namespace webnn::native {

    class NamedInputsBase : public RefCounted {
      public:
        NamedInputsBase() = default;
        virtual ~NamedInputsBase() {
#if defined(WEBNN_ENABLE_GPU_BUFFER)
            for (auto& input : mInputs) {
                WGPUBuffer gpuBuffer =
                    reinterpret_cast<WGPUBuffer>(input.second.resource.gpuBufferView.buffer);
                if (gpuBuffer != nullptr) {
                    wgpuBufferRelease(gpuBuffer);
                }
            }
#endif
        }

        // WebNN API
        void APISet(char const* name, const Input* input) {
            mInputs[std::string(name)] = *input;
#if defined(WEBNN_ENABLE_WIRE)
            // Input data type is Arrary Buffer View.
            const ArrayBufferView arrayBufferView = input->resource.arrayBufferView;
            if (arrayBufferView.buffer != nullptr) {
                std::unique_ptr<char> buffer(new char[arrayBufferView.byteLength]);
                memcpy(buffer.get(), arrayBufferView.buffer, arrayBufferView.byteLength);

                mInputs[std::string(name)].resource.arrayBufferView.buffer = buffer.get();
                mInputsBuffer.push_back(std::move(buffer));
            } else if (input->resource.gpuBufferView.buffer != nullptr) {
#    if defined(WEBNN_ENABLE_GPU_BUFFER)
                GpuBufferView gpuBufferView = input->resource.gpuBufferView;
                WGPUBuffer gpuBuffer =
                    reinterpret_cast<WGPUBuffer>(input->resource.gpuBufferView.buffer);
                wgpuBufferReference(gpuBuffer);
                mInputs[std::string(name)].resource.gpuBufferView = gpuBufferView;
#    else
                UNREACHABLE();
#    endif
            }
            std::vector<int32_t> dimensions;
            dimensions.assign(input->dimensions, input->dimensions + input->dimensionsCount);
            // Prevent destroy from allocator memory after hanlding the command.
            mInputs[std::string(name)].dimensions = dimensions.data();
            mInputsDimensions.push_back(std::move(dimensions));
#endif  // defined(WEBNN_ENABLE_WIRE)
        }

        Input Get(char const* name) const {
            if (mInputs.find(std::string(name)) == mInputs.end()) {
                return Input();
            }
            return mInputs.at(std::string(name));
        }

        // Other methods
        const std::unordered_map<std::string, Input>& GetRecords() const {
            return mInputs;
        }

      private:
        // The tempary memory in Allocator will be released after handling the command, so the
        // buffer and dimensions pointer need to be copied to use in GraphComputeCmd.
        std::vector<std::unique_ptr<char>> mInputsBuffer;
        std::vector<std::vector<int32_t>> mInputsDimensions;

        std::unordered_map<std::string, Input> mInputs;
    };

}  // namespace webnn::native

#endif  // WEBNN_NATIVE_NAMED_INPUTS_H_
