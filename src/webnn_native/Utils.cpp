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

#include "webnn_native/Utils.h"

#include "common/Assert.h"

namespace webnn_native { namespace utils {
    void ComputeImplicitPaddingForAutoPad(ml::AutoPad autoPad,
                                          int32_t dilation,
                                          int32_t inputSize,
                                          int32_t filterSize,
                                          int32_t stride,
                                          int32_t& paddingBegin,
                                          int32_t& paddingEnd) {
        int32_t outSize = (inputSize + stride - 1) / stride;
        int32_t dilatedFilter = (filterSize - 1) * dilation + 1;
        int32_t neededInput = (outSize - 1) * stride + dilatedFilter;
        int32_t totalPadding = neededInput > inputSize ? neededInput - inputSize : 0;
        switch (autoPad) {
            case ml::AutoPad::SameUpper:
                paddingBegin = totalPadding / 2;
                paddingEnd = (totalPadding + 1) / 2;
                break;
            case ml::AutoPad::SameLower:
                paddingBegin = (totalPadding + 1) / 2;
                paddingEnd = totalPadding / 2;
                break;
            default:
                DAWN_UNREACHABLE();
        }
    }
}}  // namespace webnn_native::utils