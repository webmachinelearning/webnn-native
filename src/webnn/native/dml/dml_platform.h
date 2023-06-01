// Copyright 2022 The WebNN-native Authors
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

#ifndef WEBNN_NATIVE_DML_DMLPLATFORM_H_
#define WEBNN_NATIVE_DML_DMLPLATFORM_H_

// This micro definition must be added before including "DirectML.h".
#define DML_TARGET_VERSION_USE_LATEST 1

#include <dxgi1_6.h>
#include <wrl\client.h>

#include "DirectML.h"

#define WEBNN_RETURN_IF_FAILED(EXPR)                                                              \
    do {                                                                                          \
        auto HR = EXPR;                                                                           \
        if (FAILED(HR)) {                                                                         \
            dawn::ErrorLog() << "Failed to do " << #EXPR << " Return HRESULT " << std::hex << HR; \
            return HR;                                                                            \
        }                                                                                         \
    } while (0)


using Microsoft::WRL::ComPtr;

#endif  // WEBNN_NATIVE_DML_DMLPLATFORM_H_
