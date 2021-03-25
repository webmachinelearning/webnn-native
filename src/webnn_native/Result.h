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

#ifndef WEBNN_NATIVE_RESULT_H_
#define WEBNN_NATIVE_RESULT_H_

#include <vector>

#include "common/RefCounted.h"
#include "webnn_native/Forward.h"
#include "webnn_native/webnn_platform.h"

namespace webnn_native {

    class ResultBase : public RefCounted {
      public:
        explicit ResultBase(void* buffer, uint32_t buffer_size, std::vector<int32_t>& dimensions);
        virtual ~ResultBase() = default;

        // Dawn API
        const void* Buffer() const {
            return mBuffer;
        }
        uint32_t BufferSize() const {
            return mBufferSize;
        }
        const int32_t* Dimensions() const {
            return mDimensions.data();
        }
        uint32_t DimensionsSize() const {
            return mDimensions.size();
        }

      protected:
        void* mBuffer;
        uint32_t mBufferSize;
        std::vector<int32_t> mDimensions;
    };
}  // namespace webnn_native

#endif  // WEBNN_NATIVE_RESULT_H_