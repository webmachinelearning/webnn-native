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

#ifndef WEBNN_NATIVE_OPS_POOL2d_H_
#define WEBNN_NATIVE_OPS_POOL2d_H_

#include "webnn_native/Graph.h"
#include "webnn_native/Operand.h"

namespace webnn_native { namespace op {

    enum Pool2dType {
        kAveragePool2d = 0,
        kL2Pool2d,
        kMaxPool2d,
    };

    class Pool2d final : public OperandBase {
      public:
        Pool2d(GraphBuilderBase* builder,
               Pool2dType type,
               OperandBase* input,
               Pool2dOptions const* options);
        ~Pool2d() override = default;

        MaybeError AddToGraph(GraphBase* model) const override;
        MaybeError ValidateAndInferTypes() override;

        Pool2dOptions const* GetOptions() const;
        Pool2dType GetType() const;

      private:
        Pool2dOptions mOptions;
        std::vector<int32_t> mWindowDimensions;
        std::vector<int32_t> mPadding;
        std::vector<int32_t> mStride;
        std::vector<int32_t> mDilations;
        Pool2dType mOpType;
    };

}}  // namespace webnn_native::op

#endif  // WEBNN_NATIVE_OPS_POOL2d_H_
