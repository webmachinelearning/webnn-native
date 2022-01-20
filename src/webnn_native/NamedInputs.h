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

#include <map>
#include <memory>
#include <string>

#include "common/Log.h"
#include "webnn_native/webnn_platform.h"

namespace webnn_native {

    // TODO::Use WebGPU instead of ArrayBufferView
    class NamedInputsBase : public RefCounted {
      public:
        NamedInputsBase() = default;
        virtual ~NamedInputsBase() = default;

        // WebNN API
        void Set(char const* name, const Input* input) {
            std::vector<int32_t> dimensions;
            dimensions.assign(input->dimensions, input->dimensions + input->dimensionsCount);

            std::unique_ptr<char> buffer(new char[input->resource.byteLength]);
            memcpy(buffer.get(), input->resource.buffer, input->resource.byteLength);

            // Prevent destroy from allocator memory after hanlding the command.
            mInputs[std::string(name)] = *input;
            mInputs[std::string(name)].dimensions = dimensions.data();
            mInputs[std::string(name)].resource.buffer = buffer.get();

            mInputsBuffer.push_back(std::move(buffer));
            mInputsDimensions.push_back(std::move(dimensions));
        }

        Input Get(char const* name) const {
            if (mInputs.find(std::string(name)) == mInputs.end()) {
                return Input();
            }
            return mInputs.at(std::string(name));
        }

        // Other methods
        const std::map<std::string, Input>& GetRecords() const {
            return mInputs;
        }

      private:
        // The tempary memory in Allocator will be released after handling the command, so the
        // buffer and dimensions pointer need to be copied to use in GraphComputeCmd.
        std::vector<std::unique_ptr<char>> mInputsBuffer;
        std::vector<std::vector<int32_t>> mInputsDimensions;

        std::map<std::string, Input> mInputs;
    };

}  // namespace webnn_native

#endif  // WEBNN_NATIVE_NAMED_INPUTS_H_
