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

#include "webnn_native/ops/Resample.h"

#include <algorithm>

#include "common/Log.h"
#include "webnn_native/Error.h"

namespace webnn_native { namespace op {
    Resample::Resample(GraphBuilderBase* builder,
                       OperandBase* input,
                       ResampleOptions const* options)
        : OperatorBase(builder, {input}), mScales({}), mSizes({}) {
        mOptions.mode = options == nullptr ? ml::InterpolationMode::NearestNeighbor : options->mode;
        if (options != nullptr && options->scales != nullptr) {
            mScales.assign(options->scales, options->scales + options->scalesCount);
        }
        mOptions.scales = mScales.data();
        mOptions.scalesCount = mScales.size();

        if (options != nullptr && options->sizes != nullptr) {
            mSizes.assign(options->sizes, options->sizes + options->sizesCount);
        }
        mOptions.sizes = mSizes.data();
        mOptions.sizesCount = mSizes.size();
    }

    MaybeError Resample::Validate() {
        MaybeError maybeError = OperatorBase::Validate();
        if (maybeError.IsError()) {
            return maybeError;
        }

        // The input is 4-D tensor.
        if (mInputs[0]->Rank() != 4) {
            return DAWN_VALIDATION_ERROR("Input is not a 4D tensor.");
        }
        if (mOptions.scales == nullptr && mOptions.sizes == nullptr) {
            return DAWN_VALIDATION_ERROR("scales and sizes can't be both empty.");
        }
        // The scales is 4-D tensor.
        if (mOptions.scales != nullptr && mOptions.scalesCount != 4) {
            return DAWN_VALIDATION_ERROR("Argument scales is not a 4D tensor.");
        }
        // The sizes is 4-D tensor.
        if (mOptions.sizes != nullptr && mOptions.sizesCount != 4) {
            return DAWN_VALIDATION_ERROR("Argument scales is not a 4D tensor.");
        }

        return {};
    }

}}  // namespace webnn_native::op
