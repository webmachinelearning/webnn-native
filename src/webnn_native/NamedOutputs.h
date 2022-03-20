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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common/RefCounted.h"
#include "webnn_native/webnn_platform.h"

namespace webnn_native {

    class NamedOutputsBase : public RefCounted {
      public:
        NamedOutputsBase() = default;
        virtual ~NamedOutputsBase() {
            for (auto& output : mOutputs) {
                WGPUBuffer gpuBuffer =
                    reinterpret_cast<WGPUBuffer>(output.second.gpuBufferView.buffer);
                if (gpuBuffer != nullptr) {
                    wgpuBufferRelease(gpuBuffer);
                }
            }
        }

        // WebNN API
        void Set(char const* name, const Resource* resource) {
            mOutputs[std::string(name)] = *resource;
#if defined(WEBNN_ENABLE_WIRE)
            ArrayBufferView arrayBufferView = resource->arrayBufferView;
            if (arrayBufferView.buffer != nullptr) {
                // malloc a memory to host the result of computing.
                std::unique_ptr<char> buffer(new char[arrayBufferView.byteLength]);
                // Prevent destroy from allocator memory after hanlding the command.
                mOutputs[std::string(name)].arrayBufferView.buffer = buffer.get();
                mOutputsBuffer.push_back(std::move(buffer));
            } else {
                WGPUBuffer gpuBuffer = reinterpret_cast<WGPUBuffer>(resource->gpuBufferView.buffer);
                wgpuBufferReference(gpuBuffer);
                mOutputs[std::string(name)].gpuBufferView = resource->gpuBufferView;
            }
#endif  // defined(WEBNN_ENABLE_WIRE)
        }

        // It's not support char** type in webnn.json to get name.
        void Get(size_t index, ArrayBufferView const* arrayBuffer) const {
            size_t i = 0;
            for (auto& namedOutput : mOutputs) {
                if (index == i) {
                    *const_cast<ArrayBufferView*>(arrayBuffer) = namedOutput.second.arrayBufferView;
                    return;
                }
                ++i;
            }
            UNREACHABLE();
        }

        Resource Get(char const* name) const {
            if (mOutputs.find(std::string(name)) == mOutputs.end()) {
                return Resource();
            }
            return mOutputs.at(std::string(name));
        }

        // Other methods
        const std::map<std::string, Resource>& GetRecords() const {
            return mOutputs;
        }

      private:
        // The tempary memory in Allocator will be released after handling the command, so malloc
        // the same size memory to hold the result from GraphComputeCmd.
        std::vector<std::unique_ptr<char>> mOutputsBuffer;

        std::map<std::string, Resource> mOutputs;
    };

}  // namespace webnn_native

#endif  // WEBNN_NATIVE_NAMED_OUTPUTS_H_
