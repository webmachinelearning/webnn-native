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

#include <webnn/webnn_cpp.h>
#include <webnn_native/webnn_structs_autogen.h>
#include <vector>

namespace webnn_native::utils {
    void ComputeImplicitPaddingForAutoPad(wnn::AutoPad autoPad,
                                          int32_t dilation,
                                          int32_t inputSize,
                                          int32_t filterSize,
                                          int32_t stride,
                                          int32_t& paddingBegin,
                                          int32_t& paddingEnd);

    std::vector<int32_t> ComputeImplicitPaddingForAutoPad(const Conv2dOptions* options,
                                                          std::vector<int32_t> inputSize,
                                                          std::vector<int32_t> filterSize);

    void ComputeImplicitPaddingForConvTranspose2dAutoPad(wnn::AutoPad autoPad,
                                                         int32_t dilation,
                                                         int32_t inputSize,
                                                         int32_t filterSize,
                                                         int32_t stride,
                                                         int32_t outputPadding,
                                                         int32_t& paddingBegin,
                                                         int32_t& paddingEnd);

    std::vector<int32_t> ComputeImplicitPaddingForConvTranspose2dAutoPad(
        const ConvTranspose2dOptions* options,
        std::vector<int32_t> inputSize,
        std::vector<int32_t> filterSize);

}  // namespace webnn_native::utils

#endif  // WEBNN_NATIVE_OPERATOR_H_
