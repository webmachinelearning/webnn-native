//* Copyright 2021 The Dawn Authors
//*
//* Licensed under the Apache License, Version 2.0 (the "License");
//* you may not use this file except in compliance with the License.
//* You may obtain a copy of the License at
//*
//*     http://www.apache.org/licenses/LICENSE-2.0
//*
//* Unless required by applicable law or agreed to in writing, software
//* distributed under the License is distributed on an "AS IS" BASIS,
//* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//* See the License for the specific language governing permissions and
//* limitations under the License.

#ifndef WEBNN_NATIVE_WEBNN_PLATFORM_AUTOGEN_H_
#define WEBNN_NATIVE_WEBNN_PLATFORM_AUTOGEN_H_

#include "webnn/webnn_cpp.h"
#include "webnn_native/Forward.h"

namespace webnn_native {

    template <typename T>
    struct EnumCount;

    {% for e in by_category["enum"] if e.contiguousFromZero %}
        template<>
        struct EnumCount<wnn::{{as_cppType(e.name)}}> {
            static constexpr uint32_t value = {{len(e.values)}};
        };
    {% endfor %}
}

#endif  // WEBNN_NATIVE_WEBNN_PLATFORM_AUTOGEN_H_