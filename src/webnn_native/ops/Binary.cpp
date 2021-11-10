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

#include "webnn_native/ops/Binary.h"

#include "webnn_native/Error.h"

namespace webnn_native { namespace op {

    MaybeError BroadcastShape(std::vector<int32_t> shapeA,
                              std::vector<int32_t> shapeB,
                              std::vector<int32_t>& newShape,
                              size_t skipAxes = 0) {
        // The rank of the output tensor is the maximum rank of the input tensors.
        auto rankA = shapeA.size(), rankB = shapeB.size();
        auto rankOutput = rankA >= rankB ? rankA : rankB;
        newShape.resize(rankOutput);
        DAWN_ASSERT(rankA >= skipAxes && rankB >= skipAxes);
        // For each dimension of the output tensor, its size is the maximum size along that
        // dimension of the input tensors.
        for (size_t i = 0; i < rankOutput; ++i) {
            // Skip some axes from the right side when broadcasting.
            if (i >= skipAxes) {
                auto dimA = i < rankA ? shapeA[rankA - i - 1] : 1;
                auto dimB = i < rankB ? shapeB[rankB - i - 1] : 1;
                if (dimA != dimB && dimA != 1 && dimB != 1) {
                    return DAWN_VALIDATION_ERROR("Shapes are incompatible, broadcasting failed.");
                }
                newShape[rankOutput - i - 1] = dimA > dimB ? dimA : dimB;
            }
        }
        return {};
    }

    MaybeError Binary::CaculateMatMulShape() {
        auto inputShapeA = mInputs[0]->Shape(), inputShapeB = mInputs[1]->Shape();
        auto rankA = inputShapeA.size(), rankB = inputShapeB.size();
        std::vector<int32_t> outputShape;
        if (rankA == 1 && rankB == 1) {
            if (inputShapeA != inputShapeB) {
                return DAWN_VALIDATION_ERROR(
                    "The two 1D inputs of Matmul should have the same shape.");
            }
            outputShape = {1};
        }
        if (rankA == 2 && rankB == 1) {
            if (inputShapeA[1] != inputShapeB[0]) {
                return DAWN_VALIDATION_ERROR("The input shapes are incompatible.");
            }
            outputShape = {inputShapeA[0], 1};
        }
        if (rankA == 1 && rankB == 2) {
            if (inputShapeA[0] != inputShapeB[0]) {
                return DAWN_VALIDATION_ERROR("The input shapes are incompatible.");
            }
            outputShape = {1, inputShapeB[1]};
        }
        if (rankA >= 2 && rankB >= 2) {
            if (inputShapeA[rankA - 1] != inputShapeB[rankB - 2]) {
                return DAWN_VALIDATION_ERROR("The input shapes are incompatible.");
            }
            auto maybeError = BroadcastShape(inputShapeA, inputShapeB, outputShape, 2);
            if (maybeError.IsError()) {
                return maybeError;
            }
            outputShape[outputShape.size() - 1] = inputShapeB[rankB - 1];
            outputShape[outputShape.size() - 2] = inputShapeA[rankA - 2];
        }
        mOutputs[0]->SetShape(std::move(outputShape));
        return {};
    }

    MaybeError Binary::CaculateElementWiseBinaryShape() {
        auto inputShapeA = mInputs[0]->Shape(), inputShapeB = mInputs[1]->Shape();
        std::vector<int32_t> outputShape;
        auto maybeError = BroadcastShape(inputShapeA, inputShapeB, outputShape);
        if (maybeError.IsError()) {
            return maybeError;
        }
        mOutputs[0]->SetShape(std::move(outputShape));
        return {};
    }

    MaybeError Binary::ValidateAndInferOutputInfo() {
        MaybeError maybeError = OperatorBase::ValidateAndInferOutputInfo();
        if (maybeError.IsError()) {
            return maybeError;
        }

        Ref<OperandBase> a = mInputs[0];
        Ref<OperandBase> b = mInputs[1];
        if (a->Type() != b->Type()) {
            return DAWN_VALIDATION_ERROR("Argument types are inconsistent.");
        }
        if (mOpType == kMatMul) {
            return CaculateMatMulShape();
        } else {
            return CaculateElementWiseBinaryShape();
        }
    }

}}  // namespace webnn_native::op
