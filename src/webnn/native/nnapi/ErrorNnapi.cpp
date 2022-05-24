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

#include "ErrorNnapi.h"
#include <sstream>
#include <string>
#include "NeuralNetworksTypes.h"

namespace webnn::native::nnapi {

    MaybeError CheckStatusCodeImpl(int32_t code, const char* context) {
        std::ostringstream errorMessage;
        errorMessage << context << " status code : " << code;

        switch (code) {
            case ANEURALNETWORKS_NO_ERROR:
                break;
            default:
                return DAWN_INTERNAL_ERROR(errorMessage.str());
        }
        return {};
    }

    MaybeError CheckForNullNodeImpl(std::shared_ptr<NodeInfo> ptr, const char* context) {
        if (!ptr)
            return DAWN_INTERNAL_ERROR(context);
        return {};
    }

} // namespace webnn::native::nnapi
