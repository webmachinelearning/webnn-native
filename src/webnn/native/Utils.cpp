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

#include "webnn/native/Utils.h"

#include "common/Assert.h"

namespace webnn::native::utils {
    void ComputeImplicitPaddingForAutoPad(wnn::AutoPad autoPad,
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
            case wnn::AutoPad::SameUpper:
                paddingBegin = totalPadding / 2;
                paddingEnd = (totalPadding + 1) / 2;
                break;
            case wnn::AutoPad::SameLower:
                paddingBegin = (totalPadding + 1) / 2;
                paddingEnd = totalPadding / 2;
                break;
            default:
                DAWN_UNREACHABLE();
        }
    }

    std::vector<int32_t> ComputeImplicitPaddingForAutoPad(const Conv2dOptions* options,
                                                          std::vector<int32_t> inputSize,
                                                          std::vector<int32_t> filterSize) {
        std::vector<int32_t> padding(4);
        utils::ComputeImplicitPaddingForAutoPad(options->autoPad, options->dilations[0],
                                                inputSize[0], filterSize[0], options->strides[0],
                                                padding[0], padding[1]);
        utils::ComputeImplicitPaddingForAutoPad(options->autoPad, options->dilations[1],
                                                inputSize[1], filterSize[1], options->strides[1],
                                                padding[2], padding[3]);
        return padding;
    }

    void ComputeImplicitPaddingForConvTranspose2dAutoPad(wnn::AutoPad autoPad,
                                                         int32_t dilation,
                                                         int32_t inputSize,
                                                         int32_t filterSize,
                                                         int32_t stride,
                                                         int32_t outputPadding,
                                                         int32_t& paddingBegin,
                                                         int32_t& paddingEnd) {
        int32_t outSize = inputSize * stride;
        int32_t totalPadding =
            stride * (inputSize - 1) + outputPadding + ((filterSize - 1) * dilation + 1) - outSize;
        switch (autoPad) {
            case wnn::AutoPad::SameUpper:
                paddingBegin = totalPadding / 2;
                paddingEnd = totalPadding - totalPadding / 2;
                break;
            case wnn::AutoPad::SameLower:
                paddingBegin = totalPadding - totalPadding / 2;
                paddingEnd = totalPadding / 2;
                break;
            default:
                DAWN_UNREACHABLE();
        }
    }

    std::vector<int32_t> ComputeImplicitPaddingForConvTranspose2dAutoPad(
        const ConvTranspose2dOptions* options,
        std::vector<int32_t> inputSize,
        std::vector<int32_t> filterSize) {
        std::vector<int32_t> padding(4);
        utils::ComputeImplicitPaddingForConvTranspose2dAutoPad(
            options->autoPad, options->dilations[0], inputSize[0], filterSize[0],
            options->strides[0], options->outputPadding[0], padding[0], padding[1]);
        utils::ComputeImplicitPaddingForConvTranspose2dAutoPad(
            options->autoPad, options->dilations[1], inputSize[1], filterSize[1],
            options->strides[1], options->outputPadding[1], padding[2], padding[3]);
        return padding;
    }

}  // namespace webnn::native::utils
