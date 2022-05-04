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

#ifndef WEBNN_NATIVE_NATIVEUTILS_H_
#define WEBNN_NATIVE_NATIVEUTILS_H_

#include <webnn/native/wnn_structs_autogen.h>
#include <webnn/webnn_cpp.h>
#include <vector>

namespace webnn::native::utils {
    template <typename T>
    void ComputeImplicitPaddingForAutoPad(wnn::AutoPad autoPad,
                                          T dilation,
                                          T inputSize,
                                          T filterSize,
                                          T stride,
                                          T& paddingBegin,
                                          T& paddingEnd) {
        T outSize = (inputSize + stride - 1) / stride;
        T dilatedFilter = (filterSize - 1) * dilation + 1;
        T neededInput = (outSize - 1) * stride + dilatedFilter;
        T totalPadding = neededInput > inputSize ? neededInput - inputSize : 0;
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

    template <typename S, typename T>
    std::vector<T> ComputeImplicitPaddingForAutoPad(const S* options,
                                                    std::vector<T> inputSize,
                                                    std::vector<T> filterSize) {
        std::vector<T> padding(4);
        utils::ComputeImplicitPaddingForAutoPad<T>(options->autoPad, options->dilations[0],
                                                   inputSize[0], filterSize[0], options->strides[0],
                                                   padding[0], padding[1]);
        utils::ComputeImplicitPaddingForAutoPad<T>(options->autoPad, options->dilations[1],
                                                   inputSize[1], filterSize[1], options->strides[1],
                                                   padding[2], padding[3]);
        return padding;
    }

    template <typename T>
    void ComputeImplicitPaddingForConvTranspose2dAutoPad(wnn::AutoPad autoPad,
                                                         T dilation,
                                                         T inputSize,
                                                         T filterSize,
                                                         T stride,
                                                         T outputPadding,
                                                         T& paddingBegin,
                                                         T& paddingEnd) {
        T outSize = inputSize * stride;
        T totalPadding =
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

    template <typename T>
    std::vector<T> ComputeImplicitPaddingForConvTranspose2dAutoPad(
        const ConvTranspose2dOptions* options,
        std::vector<T> inputSize,
        std::vector<T> filterSize) {
        std::vector<T> padding(4);
        utils::ComputeImplicitPaddingForConvTranspose2dAutoPad<T>(
            options->autoPad, options->dilations[0], inputSize[0], filterSize[0],
            options->strides[0], options->outputPadding[0], padding[0], padding[1]);
        utils::ComputeImplicitPaddingForConvTranspose2dAutoPad<T>(
            options->autoPad, options->dilations[1], inputSize[1], filterSize[1],
            options->strides[1], options->outputPadding[1], padding[2], padding[3]);
        return padding;
    }

}  // namespace webnn::native::utils

#endif  // WEBNN_NATIVE_OPERATOR_H_
