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

#ifndef WEBNN_NATIVE_OPS_RESAMPLE2D_H_
#define WEBNN_NATIVE_OPS_RESAMPLE2D_H_

#include "webnn_native/Graph.h"
#include "webnn_native/Operand.h"

namespace webnn_native { namespace op {

    class Resample2d final : public OperatorBase {
      public:
        Resample2d(GraphBuilderBase* builder, OperandBase* input, Resample2dOptions const* options);
        ~Resample2d() override = default;

        MaybeError AddToGraph(GraphBase* graph) const override {
            return graph->AddResample2d(this);
        }
        MaybeError ValidateAndInferOutputInfo() override;

        Resample2dOptions const* GetOptions() const {
            return &mOptions;
        }
        std::vector<float> GetScales() const {
            return mScales;
        }
        std::vector<int32_t> GetAxes() const {
            return mAxes;
        }
        std::vector<int32_t> GetOutputShape() const {
            return mOutputs[0]->Shape();
        }

      private:
        MaybeError CalculateShape();
        Resample2dOptions mOptions;
        std::vector<float> mScales;
        std::vector<int32_t> mSizes;
        std::vector<int32_t> mAxes;
    };

}}  // namespace webnn_native::op

#endif  // WEBNN_NATIVE_OPS_RESAMPLE2D_H_
