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

#ifndef WEBNN_NATIVE_DMLUTILS_H_
#define WEBNN_NATIVE_DMLUTILS_H_

#include <webnn/webnn_cpp.h>

namespace webnn_native { namespace dml { namespace utils {

    inline D3D12_HEAP_PROPERTIES CreateHeapProperties(
        D3D12_HEAP_TYPE type = D3D12_HEAP_TYPE_DEFAULT) {
        return {type, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
    };

    inline D3D12_RESOURCE_DESC CreateResourceDesc(
        UINT64 width,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
        return {D3D12_RESOURCE_DIMENSION_BUFFER, 0,    width, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                D3D12_TEXTURE_LAYOUT_ROW_MAJOR,  flags};
    };

    template <typename T>
    T RoundUpToMultiple(T value, T multiple) {
        static_assert(std::is_integral_v<T>);

        T remainder = value % multiple;
        if (remainder != 0) {
            value += multiple - remainder;
        }

        return value;
    }

}}}  // namespace webnn_native::dml::utils

#endif  // WEBNN_NATIVE_DML_UTILS_H_