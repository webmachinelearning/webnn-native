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

#ifndef WEBNN_NATIVE_NAMED_OUTPUTS_H_
#define WEBNN_NATIVE_NAMED_OUTPUTS_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/RefCounted.h"
#include "webnn/native/webnn_platform.h"

namespace webnn::native {

    class NamedOutputsBase : public RefCounted {
      public:
        NamedOutputsBase() = default;
        virtual ~NamedOutputsBase() {
#if defined(WEBNN_ENABLE_GPU_BUFFER)
            for (auto& output : mOutputs) {
                WGPUBuffer gpuBuffer =
                    reinterpret_cast<WGPUBuffer>(output.second.gpuBufferView.buffer);
                if (gpuBuffer != nullptr) {
                    wgpuBufferRelease(gpuBuffer);
                }
            }
#endif
        }

        // WebNN API
        void APISetOutput(char const* name, const Resource* resource) {
            mOutputs[std::string(name)] = *resource;
            if (resource->gpuBufferView.buffer != nullptr) {
#if defined(WEBNN_ENABLE_GPU_BUFFER)
                WGPUBuffer gpuBuffer = reinterpret_cast<WGPUBuffer>(resource->gpuBufferView.buffer);
                wgpuBufferReference(gpuBuffer);
#else
                UNREACHABLE();
#endif
            } else {
#if defined(WEBNN_ENABLE_WIRE)
                // malloc a memory to host the result of computing.
                std::unique_ptr<char> buffer(new char[resource->arrayBufferView.byteLength]);
                // Prevent destroy from allocator memory after hanlding the command.
                mOutputs[std::string(name)].arrayBufferView.buffer = buffer.get();
                mOutputsBuffer.push_back(std::move(buffer));
#endif  // defined(WEBNN_ENABLE_WIRE)
            }
        }

        void APIGetOutput(char const* name, ArrayBufferView* arrayBuffer) {
            if (mOutputs.find(std::string(name)) == mOutputs.end()) {
                return;
            }
            *arrayBuffer = mOutputs[std::string(name)].arrayBufferView;
        }

        Resource Get(char const* name) const {
            if (mOutputs.find(std::string(name)) == mOutputs.end()) {
                return Resource();
            }
            return mOutputs.at(std::string(name));
        }

        // Other methods
        const std::unordered_map<std::string, Resource>& GetRecords() const {
            return mOutputs;
        }

      private:
        // The tempary memory in Allocator will be released after handling the command, so malloc
        // the same size memory to hold the result from GraphComputeCmd.
        std::vector<std::unique_ptr<char>> mOutputsBuffer;

        std::unordered_map<std::string, Resource> mOutputs;
    };

}  // namespace webnn::native

#endif  // WEBNN_NATIVE_NAMED_OUTPUTS_H_
