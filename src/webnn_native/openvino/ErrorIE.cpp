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

#include "webnn_native/openvino/ErrorIE.h"

#include <sstream>
#include <string>

namespace webnn_native { namespace ie {

    MaybeError CheckStatusCodeImpl(IEStatusCode code, const char* context) {
        std::ostringstream errorMessage;
        errorMessage << context << " failed with status code " << code;

        switch (code) {
            case IEStatusCode::OK:
                break;
            case IEStatusCode::GENERAL_ERROR:
            case IEStatusCode::PARAMETER_MISMATCH:
            case IEStatusCode::NOT_FOUND:
            case IEStatusCode::OUT_OF_BOUNDS:
                return DAWN_VALIDATION_ERROR(errorMessage.str());
            default:
                return DAWN_INTERNAL_ERROR(errorMessage.str());
        }
        return {};
    }

}}  // namespace webnn_native::ie
