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

#include "webnn_native/ops/Clamp.h"

#include <algorithm>

#include "common/Log.h"
#include "webnn_native/Error.h"

namespace webnn_native { namespace op {

    Clamp::Clamp(GraphBuilderBase* builder, OperandBase* input, ClampOptions const* options)
        : OperandBase(builder, {input}) {
        if (options != nullptr) {
            mOptions = *options;
            if (options->minValue != nullptr) {
                mInputs.push_back(options->minValue);
            }
            if (options->maxValue != nullptr) {
                mInputs.push_back(options->maxValue);
            }
        } else {
            mOptions.minValue = nullptr;
            mOptions.maxValue = nullptr;
        }
    }

}}  // namespace webnn_native::op