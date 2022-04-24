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

#include "webnn_native/mlas/GraphMLAS.h"

#include <mlas.h>

#include <numeric>

#include "common/Assert.h"
#include "common/Log.h"
#include "webnn_native/ErrorData.h"
#include "webnn_native/NamedInputs.h"
#include "webnn_native/NamedOutputs.h"
#include "webnn_native/Operand.h"
#include "webnn_native/Utils.h"

#define VERBOSE 0

namespace webnn::native::mlas {

    void* AlignedAlloc(size_t size) {
        if (size <= 0)
            return nullptr;
        void* p;
        size_t alignment = MlasGetPreferredBufferAlignment();
#if _MSC_VER
        p = _aligned_malloc(size, alignment);
#elif defined(_LIBCPP_SGX_CONFIG)
        p = memalign(alignment, size);
#else
        if (posix_memalign(&p, alignment, size) != 0) {
            return nullptr;
        }
#endif
        return p;
    }

    void AlignedFree(void* p) {
#if _MSC_VER
        _aligned_free(p);
#else
        free(p);
#endif
    }

    class Memory : public RefCounted {
      public:
        explicit Memory(wnn::OperandType type,
                        const std::vector<int32_t>& dims,
                        bool blockedLayout = false)
            : mType(type), mDimensions(dims), mBuffer(nullptr), mBlockedLayout(blockedLayout) {
        }

        ~Memory() {
            if (mBuffer) {
                AlignedFree(mBuffer);
            }
        };

        bool Allocate() {
            size_t elementNum = std::accumulate(mDimensions.begin(), mDimensions.end(), (size_t)1,
                                                std::multiplies<size_t>{});
            size_t elementSize;
            switch (mType) {
                case wnn::OperandType::Float32:
                    elementSize = sizeof(float);
                    break;
                case wnn::OperandType::Float16:
                    elementSize = sizeof(int16_t);
                    break;
                case wnn::OperandType::Int32:
                    elementSize = sizeof(int32_t);
                    break;
                case wnn::OperandType::Uint32:
                    elementSize = sizeof(uint32_t);
                    break;
                case wnn::OperandType::Int8:
                    elementSize = sizeof(int8_t);
                    break;
                case wnn::OperandType::Uint8:
                    elementSize = sizeof(uint8_t);
                    break;
                default:
                    return false;
            }
            mByteLength = elementNum * elementSize;
            mBuffer = AlignedAlloc(mByteLength);
            return mBuffer != nullptr;
        }

        wnn::OperandType GetType() {
            return mType;
        }
        std::vector<int32_t> GetDimensions() {
            return mDimensions;
        }
        void* GetBuffer() {
            return mBuffer;
        }
        size_t GetByteLength() {
            return mByteLength;
        }
        bool IsBlockedLayout() {
            return mBlockedLayout;
        }

      private:
        wnn::OperandType mType;
        std::vector<int32_t> mDimensions;
        void* mBuffer;
        size_t mByteLength;
        bool mBlockedLayout;
    };

    class Kernel : public RefCounted {
      public:
        Kernel() = default;
        virtual ~Kernel() = default;

        virtual void Compute(MLAS_THREADPOOL* threadPool = nullptr) = 0;
    };

    class Clamp : public Kernel {
      public:
        Clamp(const Ref<Memory>& input,
              const Ref<Memory>& output,
              size_t elementNum,
              MLAS_ACTIVATION actition)
            : mInput(input), mOutput(output), mElementNum(elementNum), mActivation(actition) {
        }
        virtual ~Clamp() = default;

        virtual void Compute(MLAS_THREADPOOL* threadPool = nullptr) {
            const float* input = reinterpret_cast<const float*>(mInput->GetBuffer());
            float* output = reinterpret_cast<float*>(mOutput->GetBuffer());
            memcpy(output, input, mElementNum * sizeof(float));
            MlasActivation(&mActivation, output, nullptr, 1, mElementNum, mElementNum);
        }

      private:
        Ref<Memory> mInput;
        Ref<Memory> mOutput;
        size_t mElementNum;
        MLAS_ACTIVATION mActivation;
    };

    class Unary : public Kernel {
      public:
        Unary(op::UnaryOpType opType,
              const Ref<Memory>& input,
              const Ref<Memory>& output,
              size_t elementNum,
              MLAS_ACTIVATION activation)
            : mOpType(opType),
              mInput(input),
              mOutput(output),
              mElementNum(elementNum),
              mActivation(activation) {
        }
        virtual ~Unary() = default;

        virtual void Compute(MLAS_THREADPOOL* threadPool = nullptr) {
            const float* input = reinterpret_cast<const float*>(mInput->GetBuffer());
            float* output = reinterpret_cast<float*>(mOutput->GetBuffer());
            if (mOpType == op::UnaryOpType::kSigmoid) {
                MlasComputeLogistic(input, output, mElementNum);
            } else if (mOpType == op::UnaryOpType::kSoftmax) {
                MlasComputeSoftmax(input, output, mInput->GetDimensions()[0],
                                   mInput->GetDimensions()[1], false, threadPool);
            } else if (mOpType == op::UnaryOpType::kExp) {
                MlasComputeExp(input, output, mElementNum);
            } else if (mOpType == op::UnaryOpType::kTanh) {
                MlasComputeTanh(input, output, mElementNum);
            } else if (mOpType == op::UnaryOpType::kRelu ||
                       mOpType == op::UnaryOpType::kHardSwish ||
                       mOpType == op::UnaryOpType::kLeakyRelu) {
                memcpy(output, input, mElementNum * sizeof(float));
                MlasActivation(&mActivation, output, nullptr, 1, mElementNum, mElementNum);
            } else {
                DAWN_UNREACHABLE();
            }
        }

      private:
        op::UnaryOpType mOpType;
        Ref<Memory> mInput;
        Ref<Memory> mOutput;
        size_t mElementNum;
        MLAS_ACTIVATION mActivation;
    };

    class ReorderInput : public Kernel {
      public:
        ReorderInput(const Ref<Memory>& input,
                     const Ref<Memory>& output,
                     size_t inputChannels,
                     size_t inputSize)
            : mInput(input),
              mOutput(output),
              mInputChannels(inputChannels),
              mInputSize(inputSize){};

        virtual ~ReorderInput() = default;

        virtual void Compute(MLAS_THREADPOOL* threadPool = nullptr) {
            const float* input = reinterpret_cast<const float*>(mInput->GetBuffer());
            float* output = reinterpret_cast<float*>(mOutput->GetBuffer());
#if (VERBOSE)
            dawn::InfoLog() << "MlasReorderInputNchw";
            dawn::InfoLog() << "    input: " << input << " output: " << output;
            dawn::InfoLog() << "    input channels: " << mInputChannels;
            dawn::InfoLog() << "    input size: " << mInputSize;
#endif
            MlasReorderInputNchw(input, output, mInputChannels, mInputSize);
        }

      private:
        Ref<Memory> mInput;
        Ref<Memory> mOutput;
        size_t mInputChannels;
        size_t mInputSize;
    };

    class ReorderOutput : public Kernel {
      public:
        ReorderOutput(const Ref<Memory>& input,
                      const Ref<Memory>& output,
                      const std::vector<int64_t>& outputShape)
            : mInput(input), mOutput(output), mOutputShape(outputShape) {
        }

        virtual ~ReorderOutput() = default;

        virtual void Compute(MLAS_THREADPOOL* threadPool = nullptr) {
            const float* input = reinterpret_cast<const float*>(mInput->GetBuffer());
            float* output = reinterpret_cast<float*>(mOutput->GetBuffer());
#if (VERBOSE)
            dawn::InfoLog() << "MlasReorderOutputNchw";
            dawn::InfoLog() << "    input: " << input << " output: " << output;
            dawn::InfoLog() << "    output shape: [" << mOutputShape[0] << ", " << mOutputShape[1]
                            << ", " << mOutputShape[2] << ", " << mOutputShape[3] << "]";
#endif
            MlasReorderOutputNchw(mOutputShape.data(), input, output);
        }

      private:
        Ref<Memory> mInput;
        Ref<Memory> mOutput;
        std::vector<int64_t> mOutputShape;
    };

    class Conv2d : public Kernel {
      public:
        Conv2d(bool nchwcConv,
               const Ref<Memory>& input,
               const Ref<Memory>& filter,
               const Ref<Memory>& bias,
               const Ref<Memory>& output,
               const std::vector<int64_t>& inputShape,
               const std::vector<int64_t>& kernelShape,
               const std::vector<int64_t>& dilationShape,
               const std::vector<int64_t>& padding,
               const std::vector<int64_t>& strideShape,
               const std::vector<int64_t>& outputShape,
               size_t groupCount,
               MLAS_ACTIVATION activation)
            : nchwcConv(nchwcConv),
              mInput(input),
              mFilter(filter),
              mBias(bias),
              mOutput(output),
              mInputShape(inputShape),
              mKernelShape(kernelShape),
              mDilationShape(dilationShape),
              mPadding(padding),
              mStrideShape(strideShape),
              mOutputShape(outputShape),
              mGroupCount(groupCount),
              mActivation(activation),
              mZeroMode(true) {
        }

        virtual ~Conv2d() = default;

        bool Prepare(MLAS_THREADPOOL* threadPool = nullptr) {
            DAWN_ASSERT(!nchwcConv);
            size_t workingBufferSize;
            size_t dimensions = 2;
            size_t batchCount = mInputShape[0];
            size_t inputChannels = mInputShape[1];
            size_t outputChannels = mOutputShape[1];
            std::vector<int64_t> inputShape = {mInputShape[2], mInputShape[3]};
            std::vector<int64_t> outputShape = {mOutputShape[2], mOutputShape[3]};
            MlasConvPrepare(&mParameters, dimensions, batchCount, mGroupCount,
                            inputChannels / mGroupCount, inputShape.data(), mKernelShape.data(),
                            mDilationShape.data(), mPadding.data(), mStrideShape.data(),
                            outputShape.data(), outputChannels / mGroupCount, &mActivation,
                            &workingBufferSize, threadPool);
            if (workingBufferSize > 0) {
                mWorkingBuffer =
                    AcquireRef(new Memory(wnn::OperandType::Float32, {int32_t(workingBufferSize)}));
                if (!mWorkingBuffer->Allocate()) {
                    dawn::ErrorLog() << "Failed to allocate working buffer";
                    return false;
                }
            }
            return true;
        }

        virtual void Compute(MLAS_THREADPOOL* threadPool = nullptr) {
            const float* input = reinterpret_cast<const float*>(mInput->GetBuffer());
            const float* filter = reinterpret_cast<const float*>(mFilter->GetBuffer());
            const float* bias =
                mBias.Get() ? reinterpret_cast<const float*>(mBias->GetBuffer()) : nullptr;
            float* output = reinterpret_cast<float*>(mOutput->GetBuffer());
            if (!nchwcConv) {
                float* workingBuffer = mWorkingBuffer.Get()
                                           ? reinterpret_cast<float*>(mWorkingBuffer->GetBuffer())
                                           : nullptr;
                MlasConv(&mParameters, input, filter, bias, workingBuffer, output, threadPool);
            } else {
                MlasNchwcConv(mInputShape.data(), mKernelShape.data(), mDilationShape.data(),
                              mPadding.data(), mStrideShape.data(), mOutputShape.data(),
                              mGroupCount, input, filter, bias, output, &mActivation, mZeroMode,
                              threadPool);
            }
#if (VERBOSE)
            nchwcConv ? dawn::InfoLog() << "MlasNchwcConv" : dawn::InfoLog() << "MlasConv";
            dawn::InfoLog() << "    input: " << input << " output: " << output;
            dawn::InfoLog() << "    input shape: [" << mInputShape[0] << ", " << mInputShape[1]
                            << ", " << mInputShape[2] << ", " << mInputShape[3] << "]";
            dawn::InfoLog() << "    kernel shape: [" << mKernelShape[0] << ", " << mKernelShape[1]
                            << "]";
            dawn::InfoLog() << "    output shape: [" << mOutputShape[0] << ", " << mOutputShape[1]
                            << ", " << mOutputShape[2] << ", " << mOutputShape[3] << "]";
            dawn::InfoLog() << "    group count: " << mGroupCount;
            dawn::InfoLog() << "    activation: " << mActivation.ActivationKind;
            dawn::InfoLog() << "    zero mode: " << mZeroMode;
#endif
        }

      private:
        friend class Graph;
        bool nchwcConv;
        Ref<Memory> mInput;
        Ref<Memory> mFilter;
        Ref<Memory> mBias;
        Ref<Memory> mWorkingBuffer;
        Ref<Memory> mOutput;
        MLAS_CONV_PARAMETERS mParameters;
        std::vector<int64_t> mInputShape;
        std::vector<int64_t> mKernelShape;
        std::vector<int64_t> mDilationShape;
        std::vector<int64_t> mPadding;
        std::vector<int64_t> mStrideShape;
        std::vector<int64_t> mOutputShape;
        size_t mGroupCount;
        MLAS_ACTIVATION mActivation;
        bool mZeroMode;
    };

    class Pool2d : public Kernel {
      public:
        Pool2d(MLAS_POOLING_KIND kind,
               bool global,
               const Ref<Memory>& input,
               const Ref<Memory>& output,
               const std::vector<int64_t>& inputShape,
               const std::vector<int64_t>& kernelShape,
               const std::vector<int64_t>& dilationShape,
               const std::vector<int64_t>& padding,
               const std::vector<int64_t>& strideShape,
               const std::vector<int64_t>& outputShape)
            : mKind(kind),
              mGlobal(global),
              mInput(input),
              mOutput(output),
              mInputShape(inputShape),
              mKernelShape(kernelShape),
              mDilationShape(dilationShape),
              mPadding(padding),
              mStrideShape(strideShape),
              mOutputShape(outputShape) {
        }

        virtual ~Pool2d() = default;

        virtual void Compute(MLAS_THREADPOOL* threadPool = nullptr) {
            const float* input = reinterpret_cast<const float*>(mInput->GetBuffer());
            float* output = reinterpret_cast<float*>(mOutput->GetBuffer());
            MlasNchwcPool(mKind, mInputShape.data(), mGlobal ? nullptr : mKernelShape.data(),
                          mGlobal ? nullptr : mDilationShape.data(),
                          mGlobal ? nullptr : mPadding.data(),
                          mGlobal ? nullptr : mStrideShape.data(), mOutputShape.data(), input,
                          output, threadPool);
#if (VERBOSE)
            dawn::InfoLog() << "MlasNchwcPool";
            dawn::InfoLog() << "    kind: " << mKind;
            dawn::InfoLog() << "    global: " << mGlobal;
            dawn::InfoLog() << "    input: " << input << " output: " << output;
            dawn::InfoLog() << "    input shape: [" << mInputShape[0] << ", " << mInputShape[1]
                            << ", " << mInputShape[2] << ", " << mInputShape[3] << "]";
            dawn::InfoLog() << "    kernel shape: [" << mKernelShape[0] << ", " << mKernelShape[1]
                            << "]";
            dawn::InfoLog() << "    output shape: [" << mOutputShape[0] << ", " << mOutputShape[1]
                            << ", " << mOutputShape[2] << ", " << mOutputShape[3] << "]";
#endif
        }

      private:
        friend class Graph;
        MLAS_POOLING_KIND mKind;
        bool mGlobal;
        Ref<Memory> mInput;
        Ref<Memory> mOutput;
        std::vector<int64_t> mInputShape;
        std::vector<int64_t> mKernelShape;
        std::vector<int64_t> mDilationShape;
        std::vector<int64_t> mPadding;
        std::vector<int64_t> mStrideShape;
        std::vector<int64_t> mOutputShape;
    };

    Graph::Graph(Context* context) : GraphBase(context) {
    }

    Graph::~Graph() {
    }

    MaybeError Graph::AddConstant(const op::Constant* constant) {
        const OperandBase* operand = constant->PrimaryOutput();
        Ref<Memory> memory = AcquireRef(new Memory(operand->Type(), operand->Shape()));
        if (!memory->Allocate()) {
            return DAWN_INTERNAL_ERROR("Failed to allocate memory.");
        }
        memcpy(memory->GetBuffer(), constant->GetBuffer(), constant->GetByteLength());
        mMemoryMap.insert(std::make_pair(operand, memory));
#if (VERBOSE)
        dawn::InfoLog() << "add constant memory: " << memory.Get();
#endif
        return {};
    }

    MaybeError Graph::AddInput(const op::Input* input) {
        const OperandBase* operand = input->PrimaryOutput();
        Ref<Memory> memory = AcquireRef(new Memory(operand->Type(), operand->Shape()));
        if (!memory->Allocate()) {
            return DAWN_INTERNAL_ERROR("Failed to allocate memory.");
        }
        mMemoryMap.insert(std::make_pair(operand, memory));
        mInputs.insert(std::make_pair(input->GetName(), memory));
#if (VERBOSE)
        dawn::InfoLog() << "add input memory: " << memory.Get();
#endif
        return {};
    }

    MaybeError Graph::AddOutput(std::string_view name, const OperandBase* output) {
        DAWN_ASSERT(mMemoryMap.find(output) != mMemoryMap.end());
        Ref<Memory> memory = mMemoryMap.at(output);
        if (memory->IsBlockedLayout()) {
            // ReorderOutput
            const size_t rank = output->Shape().size();
            if (rank != 4) {
                return DAWN_INTERNAL_ERROR("NCHWc memory layout only supports rank 4.");
            }
            int32_t channels = output->Shape()[1];
            DAWN_ASSERT(channels <= memory->GetDimensions()[1]);
            Ref<Memory> nchwMemory = AcquireRef(new Memory(output->Type(), output->Shape()));
            if (!nchwMemory->Allocate()) {
                return DAWN_INTERNAL_ERROR("Failed to allocate output memory.");
            }
            std::vector<int64_t> outputShape = {output->Shape()[0], output->Shape()[1],
                                                output->Shape()[2], output->Shape()[3]};
            mKernels.push_back(AcquireRef(new ReorderOutput(memory, nchwMemory, outputShape)));
            memory = nchwMemory;
        }
        mOutputs.insert(std::make_pair(name.data(), memory));
        return {};
    }

    MaybeError Graph::AddClamp(const op::Clamp* clamp) {
        const OperandBase* inputOperand = clamp->Inputs()[0].Get();
        if (inputOperand->Type() != wnn::OperandType::Float32) {
            return DAWN_INTERNAL_ERROR("Only support float32");
        }
        DAWN_ASSERT(mMemoryMap.find(inputOperand) != mMemoryMap.end());
        Ref<Memory> inputMemory = mMemoryMap.at(inputOperand);
        const OperandBase* outputOperand = clamp->PrimaryOutput();
        Ref<Memory> outputMemory =
            AcquireRef(new Memory(outputOperand->Type(), outputOperand->Shape()));
        if (!outputMemory->Allocate()) {
            return DAWN_INTERNAL_ERROR("Failed to allocate output memory");
        }
        mMemoryMap.insert(std::make_pair(outputOperand, outputMemory));
        std::vector<int32_t> dimensions = inputOperand->Shape();
        size_t elementNum = std::accumulate(dimensions.begin(), dimensions.end(), (size_t)1,
                                            std::multiplies<size_t>{});
        MLAS_ACTIVATION activation;
        activation.ActivationKind = MlasClipActivation;
        activation.Parameters.Clip.minimum = clamp->GetMinValue();
        activation.Parameters.Clip.maximum = clamp->GetMaxValue();
        mKernels.push_back(
            AcquireRef(new Clamp(inputMemory, outputMemory, elementNum, activation)));
        return {};
    }

    MaybeError Graph::AddBinary(const op::Binary* binary) {
        if (binary->GetType() != op::BinaryOpType::kAdd) {
            return DAWN_UNIMPLEMENTED_ERROR("Binary op is unimplemented.");
        }
        const OperandBase* a = binary->Inputs()[0].Get();
        if (a->Type() != wnn::OperandType::Float32) {
            return DAWN_INTERNAL_ERROR("Only support float32 input.");
        }
        const OperandBase* b = binary->Inputs()[1].Get();
        if (b->Type() != wnn::OperandType::Float32) {
            return DAWN_INTERNAL_ERROR("Only support float32 input.");
        }
        if (a->Shape() != b->Shape()) {
            return DAWN_INTERNAL_ERROR("Shapes don't match.");
        }
        DAWN_ASSERT(mMemoryMap.find(a) != mMemoryMap.end());
        Ref<Memory> aMemory = mMemoryMap.at(a);
        if (!aMemory->IsBlockedLayout()) {
            return DAWN_INTERNAL_ERROR("Only support blocked memory.");
        }
        DAWN_ASSERT(mMemoryMap.find(b) != mMemoryMap.end());
        Ref<Memory> bMemory = mMemoryMap.at(b);
        if (!bMemory->IsBlockedLayout()) {
            return DAWN_INTERNAL_ERROR("Only support blocked memory.");
        }
#if (VERBOSE)
        dawn::InfoLog() << "Add add";
        dawn::InfoLog() << "    a: " << a->Operator();
        dawn::InfoLog() << "    b: " << b->Operator();
#endif
        Ref<Conv2d> aConv2d;
        if (mConv2dKernels.find(a->Operator()) != mConv2dKernels.end()) {
            aConv2d = mConv2dKernels.at(a->Operator());
        }
        Ref<Conv2d> bConv2d;
        if (mConv2dKernels.find(b->Operator()) != mConv2dKernels.end()) {
            bConv2d = mConv2dKernels.at(b->Operator());
        }
        if (aConv2d.Get() == nullptr && bConv2d.Get() == nullptr) {
            return DAWN_INTERNAL_ERROR("At least one operand should be conv2d.");
        }
        Ref<Conv2d> conv2d;
        if (aConv2d.Get() != nullptr && bConv2d.Get() != nullptr) {
            size_t aKernelIndex = 0;
            for (; aKernelIndex < mKernels.size(); ++aKernelIndex) {
                if (mKernels[aKernelIndex].Get() == aConv2d.Get()) {
                    break;
                }
            }
            size_t bKernelIndex = 0;
            for (; bKernelIndex < mKernels.size(); ++bKernelIndex) {
                if (mKernels[bKernelIndex].Get() == bConv2d.Get()) {
                    break;
                }
            }
            if (aKernelIndex > bKernelIndex) {
                aConv2d->mOutput = bConv2d->mOutput;
                conv2d = aConv2d;
            } else {
                bConv2d->mOutput = aConv2d->mOutput;
                conv2d = bConv2d;
            }
        } else if (aConv2d.Get() != nullptr) {
            aConv2d->mOutput = bMemory;
            conv2d = aConv2d;
        } else {
            bConv2d->mOutput = aMemory;
            conv2d = bConv2d;
        }
        conv2d->mZeroMode = false;
        const OperandBase* output = binary->PrimaryOutput();
        mMemoryMap.insert(std::make_pair(output, conv2d->mOutput));
        return {};
    }

    MaybeError Graph::AddConv2d(const op::Conv2d* conv2d) {
        const Conv2dOptions* options = conv2d->GetOptions();
        if (options->inputLayout != wnn::InputOperandLayout::Nchw) {
            return DAWN_INTERNAL_ERROR("Only support nchw input layout");
        }
        if (options->filterLayout != wnn::Conv2dFilterOperandLayout::Oihw) {
            return DAWN_INTERNAL_ERROR("Only support iohw filter layout");
        }
        const OperandBase* inputOperand = conv2d->Inputs()[0].Get();
        if (inputOperand->Type() != wnn::OperandType::Float32) {
            return DAWN_INTERNAL_ERROR("Only support float32 input");
        }
        const OperandBase* filterOperand = conv2d->Inputs()[1].Get();
        if (filterOperand->Type() != wnn::OperandType::Float32) {
            return DAWN_INTERNAL_ERROR("Only support float32 filter");
        }
        size_t groupCount = options->groups;
        size_t batchCount = inputOperand->Shape()[0];
        size_t inputChannels = inputOperand->Shape()[1];
        size_t outputChannels = filterOperand->Shape()[0];
        int32_t inputHeight = inputOperand->Shape()[2];
        int32_t inputWidth = inputOperand->Shape()[3];
        int32_t filterHeight = filterOperand->Shape()[2];
        int32_t filterWidth = filterOperand->Shape()[3];
        std::vector<int64_t> inputShape = {inputOperand->Shape()[0], inputOperand->Shape()[1],
                                           inputOperand->Shape()[2], inputOperand->Shape()[3]};
        std::vector<int64_t> kernelShape = {filterOperand->Shape()[2], filterOperand->Shape()[3]};
        std::vector<int64_t> dilationShape = {options->dilations[0], options->dilations[1]};
        int32_t paddingBeginningHeight = options->padding[0],
                paddingEndingHeight = options->padding[1],
                paddingBeginningWidth = options->padding[2],
                paddingEndingWidth = options->padding[3];
        if (options->autoPad != wnn::AutoPad::Explicit) {
            utils::ComputeImplicitPaddingForAutoPad(options->autoPad, options->dilations[0],
                                                    inputHeight, filterHeight, options->strides[0],
                                                    paddingBeginningHeight, paddingEndingHeight);
            utils::ComputeImplicitPaddingForAutoPad(options->autoPad, options->dilations[1],
                                                    inputWidth, filterWidth, options->strides[1],
                                                    paddingBeginningWidth, paddingEndingWidth);
        }
        std::vector<int64_t> padding = {paddingBeginningHeight, paddingEndingHeight,
                                        paddingBeginningWidth, paddingEndingWidth};
        std::vector<int64_t> strideShape = {options->strides[0], options->strides[1]};
        const OperandBase* outputOperand = conv2d->PrimaryOutput();
        std::vector<int64_t> outputShape = {outputOperand->Shape()[0], outputOperand->Shape()[1],
                                            outputOperand->Shape()[2], outputOperand->Shape()[3]};

        size_t nchwcBlockSize = MlasNchwcGetBlockSize();
        bool nchwcConv = nchwcBlockSize > 1 ? true : false;
        bool reorderInput = true;
        bool reorderFilterOIHWBo = false;
        int64_t filterInputChannels = filterOperand->Shape()[1];
        ;
        int64_t nchwcGroupCount = groupCount;

        // The current implementation of ReorderInput requires the channel count to be
        // aligned to this value.
        constexpr int64_t channelAlignment = 4;
        const int64_t nchwcInputChannels =
            (inputChannels + nchwcBlockSize - 1) & ~(nchwcBlockSize - 1);
        const int64_t nchwcOutputChannels =
            (outputChannels + nchwcBlockSize - 1) & ~(nchwcBlockSize - 1);

        if (nchwcConv) {
            if (groupCount > 1) {
                if ((outputChannels % channelAlignment) != 0) {
                    nchwcConv = false;
                }
                if (filterInputChannels == 1 && outputChannels == groupCount) {
                    // Depthwise convolution.
                    reorderFilterOIHWBo = true;
                    nchwcGroupCount = nchwcOutputChannels;
                } else if (((inputChannels % nchwcBlockSize) != 0) ||
                           ((outputChannels % groupCount) != 0) ||
                           (((outputChannels / groupCount) % nchwcBlockSize) != 0)) {
                    nchwcConv = false;
                }
            } else {
                if (static_cast<size_t>(inputChannels) < nchwcBlockSize) {
                    // Use NCHW input buffer directly.
                    reorderFilterOIHWBo = true;
                    reorderInput = false;
                } else {
                    if ((inputChannels % channelAlignment) != 0) {
                        nchwcConv = false;
                    }
                    filterInputChannels =
                        (inputChannels + nchwcBlockSize - 1) & ~(nchwcBlockSize - 1);
                }
            }
        }

        DAWN_ASSERT(mMemoryMap.find(inputOperand) != mMemoryMap.end());
        Ref<Memory> inputMemory = mMemoryMap.at(inputOperand);
        if (nchwcConv && reorderInput) {
            if (!inputMemory->IsBlockedLayout()) {
                Ref<Memory> reorderInputMemory = inputMemory;
                std::vector<int32_t> reorderedOutputShape = {
                    static_cast<int32_t>(batchCount), static_cast<int32_t>(nchwcInputChannels),
                    inputHeight, inputWidth};
                Ref<Memory> reorderOutputMemory =
                    AcquireRef(new Memory(inputOperand->Type(), reorderedOutputShape, true));
                if (!reorderOutputMemory->Allocate()) {
                    return DAWN_INTERNAL_ERROR("Failed to allocate reorder output memory.");
                }
                size_t inputSize = inputHeight * inputWidth;
                mKernels.push_back(AcquireRef(new ReorderInput(
                    reorderInputMemory, reorderOutputMemory, inputChannels, inputSize)));
                inputMemory = reorderOutputMemory;
                inputShape[1] = nchwcInputChannels;
            } else {
                inputShape[1] = inputMemory->GetDimensions()[1];
            }
        }

        DAWN_ASSERT(mMemoryMap.find(filterOperand) != mMemoryMap.end());
        Ref<Memory> filterMemory = mMemoryMap.at(filterOperand);
        if (nchwcConv && !filterMemory->IsBlockedLayout()) {
            std::vector<int32_t> reorderedFilterShape = {static_cast<int32_t>(nchwcOutputChannels),
                                                         static_cast<int32_t>(filterInputChannels),
                                                         filterHeight, filterWidth};
            Ref<Memory> reorderedFilterMemory =
                AcquireRef(new Memory(filterOperand->Type(), reorderedFilterShape, true));
            if (!reorderedFilterMemory->Allocate()) {
                return DAWN_INTERNAL_ERROR("Failed to allocate reorder output memory.");
            }
            std::vector<int64_t> filterShape = {
                filterOperand->Shape()[0], filterOperand->Shape()[1], filterOperand->Shape()[2],
                filterOperand->Shape()[3]};
            const float* filterData = reinterpret_cast<const float*>(filterMemory->GetBuffer());
            float* reorderdFilterData =
                reinterpret_cast<float*>(reorderedFilterMemory->GetBuffer());
            if (reorderFilterOIHWBo) {
                MlasReorderFilterOIHWBo(filterShape.data(), filterData, reorderdFilterData);
            } else {
                MlasReorderFilterOIHWBiBo(filterShape.data(), filterData, reorderdFilterData);
            }
            filterMemory = reorderedFilterMemory;
        }
        Ref<Memory> biasMemory;
        if (options->bias) {
            const OperandBase* biasOperand = conv2d->Inputs()[2].Get();
            if (biasOperand->Type() != wnn::OperandType::Float32) {
                return DAWN_INTERNAL_ERROR("Only support float32 bias");
            }
            DAWN_ASSERT(mMemoryMap.find(biasOperand) != mMemoryMap.end());
            biasMemory = mMemoryMap.at(biasOperand);
            if (nchwcConv && !biasMemory->IsBlockedLayout()) {
                std::vector<int32_t> alignedBiasShape = {static_cast<int32_t>(nchwcOutputChannels)};
                Ref<Memory> alignedBiasMemory =
                    AcquireRef(new Memory(biasOperand->Type(), alignedBiasShape, true));
                if (!alignedBiasMemory->Allocate()) {
                    return DAWN_INTERNAL_ERROR("Failed to allocate reorder output memory.");
                }
                memcpy(alignedBiasMemory->GetBuffer(), biasMemory->GetBuffer(),
                       biasMemory->GetByteLength());
                biasMemory = alignedBiasMemory;
            }
        }

        MLAS_ACTIVATION activation;
        activation.ActivationKind = MlasIdentityActivation;
        if (options->activation) {
            switch (options->activation->GetFusionType()) {
                case FusionType::Clamp:
                    activation.ActivationKind = MlasClipActivation;
                    activation.Parameters.Clip.minimum =
                        reinterpret_cast<op::FusionClamp*>(options->activation)->GetMinValue();
                    activation.Parameters.Clip.maximum =
                        reinterpret_cast<op::FusionClamp*>(options->activation)->GetMaxValue();
                    break;
                case FusionType::HardSwish:
                    activation.ActivationKind = MlasHardSigmoidActivation;
                    activation.Parameters.HardSigmoid.alpha = 1.0 / 6.0;
                    activation.Parameters.HardSigmoid.beta = 0.5;
                    break;
                case FusionType::Relu:
                    activation.ActivationKind = MlasReluActivation;
                    break;
                case FusionType::Sigmoid:
                    activation.ActivationKind = MlasLogisticActivation;
                    break;
                case FusionType::LeakyRelu:
                    activation.ActivationKind = MlasLeakyReluActivation;
                    activation.Parameters.LeakyRelu.alpha =
                        reinterpret_cast<op::FusionLeakyRelu*>(options->activation)->GetAlpha();
                    break;
                default:
                    return DAWN_INTERNAL_ERROR("Unsupported fused activation");
            }
        }

        Ref<Memory> outputMemory;
        if (!nchwcConv) {
            outputMemory = AcquireRef(new Memory(outputOperand->Type(), outputOperand->Shape()));
        } else {
            std::vector<int32_t> nchwcOutputShape = {
                outputOperand->Shape()[0], static_cast<int32_t>(nchwcOutputChannels),
                outputOperand->Shape()[2], outputOperand->Shape()[3]};
            outputMemory = AcquireRef(new Memory(outputOperand->Type(), nchwcOutputShape, true));
            outputShape[1] = nchwcOutputChannels;
        }
        if (!outputMemory->Allocate()) {
            return DAWN_INTERNAL_ERROR("Failed to allocate output memory");
        }
        mMemoryMap.insert(std::make_pair(outputOperand, outputMemory));

        Ref<Conv2d> kernel = AcquireRef(new Conv2d(
            nchwcConv, inputMemory, filterMemory, biasMemory, outputMemory, inputShape, kernelShape,
            dilationShape, padding, strideShape, outputShape, nchwcGroupCount, activation));
        if (!nchwcConv) {
            if (!kernel->Prepare(reinterpret_cast<Context*>(GetContext())->GetThreadPool())) {
                return DAWN_INTERNAL_ERROR("Failed to prepare conv2d.");
            }
        }
#if (VERBOSE)
        dawn::InfoLog() << "Add conv2d " << conv2d << " kernel " << kernel.Get();
        dawn::InfoLog() << "    input memory: " << inputMemory.Get();
        dawn::InfoLog() << "    output memory: " << outputMemory.Get();
#endif
        mKernels.push_back(kernel);
        mConv2dKernels.insert(std::make_pair(conv2d, kernel));
        return {};
    }

    MaybeError Graph::AddPool2d(const op::Pool2d* pool2d) {
        const Pool2dOptions* options = pool2d->GetOptions();
        if (options->layout != wnn::InputOperandLayout::Nchw) {
            return DAWN_INTERNAL_ERROR("Only support nchw input layout");
        }
        const OperandBase* inputOperand = pool2d->Inputs()[0].Get();
        if (inputOperand->Type() != wnn::OperandType::Float32) {
            return DAWN_INTERNAL_ERROR("Only support float32 input");
        }
        size_t nchwcBlockSize = MlasNchwcGetBlockSize();
        bool nchwcPool = nchwcBlockSize > 1 ? true : false;

        MLAS_POOLING_KIND kind;
        if (pool2d->GetType() == op::Pool2dType::kAveragePool2d) {
            kind = MlasAveragePoolingIncludePad;
        } else if (pool2d->GetType() == op::Pool2dType::kMaxPool2d) {
            kind = MlasMaximumPooling;
        } else {
            return DAWN_INTERNAL_ERROR("Pool type is unsupported");
        }
        size_t batchCount = inputOperand->Shape()[0];
        size_t inputChannels = inputOperand->Shape()[1];
        std::vector<int64_t> inputShape = {inputOperand->Shape()[0], inputOperand->Shape()[1],
                                           inputOperand->Shape()[2], inputOperand->Shape()[3]};
        int32_t inputHeight = inputOperand->Shape()[2];
        int32_t inputWidth = inputOperand->Shape()[3];
        bool globalPooling = options->windowDimensions == nullptr;
        std::vector<int64_t> kernelShape = {
            options->windowDimensions ? options->windowDimensions[0] : inputOperand->Shape()[2],
            options->windowDimensions ? options->windowDimensions[1] : inputOperand->Shape()[3]};
        std::vector<int64_t> dilationShape = {options->dilations[0], options->dilations[1]};
        int32_t paddingBeginningHeight = options->padding[0],
                paddingEndingHeight = options->padding[1],
                paddingBeginningWidth = options->padding[2],
                paddingEndingWidth = options->padding[3];
        if (options->autoPad != wnn::AutoPad::Explicit) {
            utils::ComputeImplicitPaddingForAutoPad(
                options->autoPad, options->dilations[0], inputHeight, kernelShape[0],
                options->strides[0], paddingBeginningHeight, paddingEndingHeight);
            utils::ComputeImplicitPaddingForAutoPad(options->autoPad, options->dilations[1],
                                                    inputWidth, kernelShape[1], options->strides[1],
                                                    paddingBeginningWidth, paddingEndingWidth);
        }
        std::vector<int64_t> padding = {paddingBeginningHeight, paddingEndingHeight,
                                        paddingBeginningWidth, paddingEndingWidth};
        std::vector<int64_t> strideShape = {options->strides[0], options->strides[1]};

        bool reorderInput = true;
        // The current implementation of ReorderInput requires the channel count to be
        // aligned to this value.
        constexpr int64_t channelAlignment = 4;
        const int64_t nchwcChannels = (inputChannels + nchwcBlockSize - 1) & ~(nchwcBlockSize - 1);

        if (static_cast<size_t>(inputChannels) < nchwcBlockSize) {
            // Use NCHW input buffer directly.
            reorderInput = false;
        } else {
            if ((inputChannels % channelAlignment) != 0) {
                nchwcPool = false;
            }
        }

        if (!nchwcPool) {
            return DAWN_INTERNAL_ERROR("Only support nchwc pool");
        }

        DAWN_ASSERT(mMemoryMap.find(inputOperand) != mMemoryMap.end());
        Ref<Memory> inputMemory = mMemoryMap.at(inputOperand);
        if (nchwcPool && reorderInput) {
            if (!inputMemory->IsBlockedLayout()) {
                Ref<Memory> reorderInputMemory = inputMemory;
                std::vector<int32_t> reorderedOutputShape = {static_cast<int32_t>(batchCount),
                                                             static_cast<int32_t>(nchwcChannels),
                                                             inputHeight, inputWidth};
                Ref<Memory> reorderOutputMemory =
                    AcquireRef(new Memory(inputOperand->Type(), reorderedOutputShape, true));
                if (!reorderOutputMemory->Allocate()) {
                    return DAWN_INTERNAL_ERROR("Failed to allocate reorder output memory.");
                }
                size_t inputSize = inputHeight * inputWidth;
                mKernels.push_back(AcquireRef(new ReorderInput(
                    reorderInputMemory, reorderOutputMemory, inputChannels, inputSize)));
                inputMemory = reorderOutputMemory;
                inputShape[1] = nchwcChannels;
            } else {
                inputShape[1] = inputMemory->GetDimensions()[1];
            }
        }

        const OperandBase* outputOperand = pool2d->PrimaryOutput();
        std::vector<int64_t> outputShape = {outputOperand->Shape()[0], nchwcChannels,
                                            outputOperand->Shape()[2], outputOperand->Shape()[3]};
        Ref<Memory> outputMemory;
        std::vector<int32_t> nchwcOutputShape = {
            outputOperand->Shape()[0], inputMemory->GetDimensions()[1], outputOperand->Shape()[2],
            outputOperand->Shape()[3]};
        outputMemory = AcquireRef(new Memory(outputOperand->Type(), nchwcOutputShape, true));
        if (!outputMemory->Allocate()) {
            return DAWN_INTERNAL_ERROR("Failed to allocate output memory");
        }
        mMemoryMap.insert(std::make_pair(outputOperand, outputMemory));
        Ref<Pool2d> kernel =
            AcquireRef(new Pool2d(kind, globalPooling, inputMemory, outputMemory, inputShape,
                                  kernelShape, dilationShape, padding, strideShape, outputShape));
#if (VERBOSE)
        dawn::InfoLog() << "Add pool2d " << pool2d << " kernel " << kernel.Get();
#endif
        mKernels.push_back(kernel);
        return {};
    }

    MaybeError Graph::AddUnary(const op::Unary* unary) {
        op::UnaryOpType opType = unary->GetType();
        if (opType == op::UnaryOpType::kExp || opType == op::UnaryOpType::kHardSwish ||
            opType == op::UnaryOpType::kLeakyRelu || opType == op::UnaryOpType::kRelu ||
            opType == op::UnaryOpType::kSigmoid || opType == op::UnaryOpType::kSoftmax ||
            opType == op::UnaryOpType::kTanh) {
            const OperandBase* inputOperand = unary->Inputs()[0].Get();
            if (inputOperand->Type() != wnn::OperandType::Float32) {
                return DAWN_INTERNAL_ERROR("Only support float32");
            }
            DAWN_ASSERT(mMemoryMap.find(inputOperand) != mMemoryMap.end());
            Ref<Memory> inputMemory = mMemoryMap.at(inputOperand);
            const OperandBase* outputOperand = unary->PrimaryOutput();
            Ref<Memory> outputMemory =
                AcquireRef(new Memory(outputOperand->Type(), outputOperand->Shape()));
            if (!outputMemory->Allocate()) {
                return DAWN_INTERNAL_ERROR("Failed to allocate output memory");
            }
            mMemoryMap.insert(std::make_pair(outputOperand, outputMemory));
            std::vector<int32_t> dimensions = inputOperand->Shape();
            size_t elementNum = std::accumulate(dimensions.begin(), dimensions.end(), (size_t)1,
                                                std::multiplies<size_t>{});
            MLAS_ACTIVATION activation;
            if (opType == op::UnaryOpType::kRelu) {
                activation.ActivationKind = MlasReluActivation;
            } else if (opType == op::UnaryOpType::kHardSwish) {
                activation.ActivationKind = MlasHardSigmoidActivation;
                activation.Parameters.HardSigmoid.alpha = 1.0 / 6.0;
                activation.Parameters.HardSigmoid.beta = 0.5;
            } else if (opType == op::UnaryOpType::kLeakyRelu) {
                activation.ActivationKind = MlasLeakyReluActivation;
                activation.Parameters.LeakyRelu.alpha =
                    reinterpret_cast<const op::LeakyRelu*>(unary)->GetAlpha();
            }
            mKernels.push_back(
                AcquireRef(new Unary(opType, inputMemory, outputMemory, elementNum, activation)));
        } else {
            return DAWN_UNIMPLEMENTED_ERROR("Unsupported unary op");
        }
        return {};
    }

    MaybeError Graph::Finish() {
        return {};
    }

    MaybeError Graph::CompileImpl() {
        return {};
    }

    MaybeError Graph::ComputeImpl(NamedInputsBase* inputs, NamedOutputsBase* outputs) {
        for (auto& [name, input] : inputs->GetRecords()) {
            Ref<Memory> inputMemory = mInputs.at(name);
            auto& resource = input.resource.arrayBufferView;
            DAWN_INVALID_IF(inputMemory->GetByteLength() < resource.byteLength,
                            "The size of input memory is less than input buffer.");
            memcpy(inputMemory->GetBuffer(),
                   static_cast<int8_t*>(resource.buffer) + resource.byteOffset,
                   resource.byteLength);
        }

        for (auto& kernel : mKernels) {
            kernel->Compute(reinterpret_cast<Context*>(GetContext())->GetThreadPool());
        }

        std::vector<std::string> outputNames;
        for (auto& [name, _] : outputs->GetRecords()) {
            outputNames.push_back(name);
        }

        for (size_t i = 0; i < outputNames.size(); ++i) {
            std::string outputName = outputNames[i];
            Ref<Memory> outputMemory = mOutputs.at(outputName);
            const ArrayBufferView& output = outputs->GetRecords().at(outputName).arrayBufferView;
            DAWN_INVALID_IF(output.byteLength < outputMemory->GetByteLength(),
                            "The size of output buffer is less than output memory.");
            memcpy(static_cast<int8_t*>(output.buffer) + output.byteOffset,
                   outputMemory->GetBuffer(), output.byteLength);
        }
        return {};
    }

}  // namespace webnn::native::mlas
