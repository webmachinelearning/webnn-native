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

#ifndef WEBNN_NATIVE_EXAMPLES_SAMPLE_UTILS_H_
#define WEBNN_NATIVE_EXAMPLES_SAMPLE_UTILS_H_
#define TIME_TYPE std::chrono::duration<double, std::milli>

#include <webnn/webnn.h>
#include <webnn/webnn_cpp.h>
#include <condition_variable>
#include <mutex>
#include <vector>

#include "common/Log.h"
#include "common/RefCounted.h"
#include "third_party/cnpy/cnpy.h"
#include "third_party/stb/stb_image.h"
#include "third_party/stb/stb_image_resize.h"

class ExampleBase {
  public:
    ExampleBase() = default;
    virtual ~ExampleBase() = default;

    virtual bool ParseAndCheckExampleOptions(int argc, const char* argv[]);

    std::string mImagePath;
    std::string mWeightsPath;
    std::string mLabelPath;
    int mNIter = 1;
    std::string mLayout = "nchw";
    bool mNormalization = false;
    size_t mModelHeight;
    size_t mModelWidth;
    size_t mModelChannels;
    std::vector<float> mMean = {0, 0, 0};  // Average values of pixels on channels.
    std::vector<float> mStd = {1, 1, 1};   // Variance values of pixels on channels.
    std::string mChannelScheme = "RGB";
    std::vector<int32_t> mOutputShape;
    std::string mDevicePreference = "default";
    std::string mPowerPreference = "default";
    bool mFused = true;
};

wnn::Context CreateCppContext(wnn::ContextOptions const* options = nullptr);
wnn::NamedInputs CreateCppNamedInputs();
wnn::NamedOutputs CreateCppNamedOutputs();
wnn::OperatorArray CreateCppOperatorArray();
void DoFlush();

bool Expected(float output, float expected);

namespace utils {

    uint32_t SizeOfShape(const std::vector<int32_t>& dims);

    enum FusedActivation { NONE, RELU, RELU6, SIGMOID, LEAKYRELU, TANH };

    wnn::ClampOptions CreateClampOptions(const wnn::GraphBuilder& builder,
                                         const std::vector<int32_t>& minShape,
                                         const std::vector<float>& minValue,
                                         const std::vector<int32_t>& maxShape,
                                         const std::vector<float>& maxValue);

    const wnn::FusionOperator CreateActivationOperator(
        const wnn::GraphBuilder& builder,
        FusedActivation activation = FusedActivation::NONE,
        const void* options = nullptr);

    const wnn::Operand CreateActivationOperand(const wnn::GraphBuilder& builder,
                                               const wnn::Operand& input,
                                               FusedActivation activation,
                                               const void* options = nullptr);

    wnn::Operand BuildInput(const wnn::GraphBuilder& builder,
                            std::string name,
                            const std::vector<int32_t>& dimensions,
                            wnn::OperandType type = wnn::OperandType::Float32);

    wnn::Operand BuildConstant(const wnn::GraphBuilder& builder,
                               const std::vector<int32_t>& dimensions,
                               const void* value,
                               size_t size,
                               wnn::OperandType type = wnn::OperandType::Float32);

    template <typename T>
    struct Conv2dBaseOptions {
      public:
        std::vector<int32_t> padding;
        std::vector<int32_t> strides;
        std::vector<int32_t> dilations;
        wnn::AutoPad autoPad = wnn::AutoPad::Explicit;
        int32_t groups = 1;
        wnn::InputOperandLayout inputLayout = wnn::InputOperandLayout::Nchw;
        wnn::Operand bias;
        wnn::FusionOperator activation;

        T& GetBaseOptions() {
            if (!padding.empty()) {
                mOptions.paddingCount = padding.size();
                mOptions.padding = padding.data();
            }
            if (!strides.empty()) {
                mOptions.stridesCount = strides.size();
                mOptions.strides = strides.data();
            }
            if (!dilations.empty()) {
                mOptions.dilationsCount = dilations.size();
                mOptions.dilations = dilations.data();
            }
            mOptions.groups = groups;
            mOptions.autoPad = autoPad;
            mOptions.inputLayout = inputLayout;
            mOptions.bias = bias;
            mOptions.activation = activation;

            return mOptions;
        }

      protected:
        T mOptions;
    };

    struct Conv2dOptions final : public Conv2dBaseOptions<wnn::Conv2dOptions> {
      public:
        wnn::Conv2dFilterOperandLayout filterLayout = wnn::Conv2dFilterOperandLayout::Oihw;

        const wnn::Conv2dOptions* AsPtr() {
            mOptions = GetBaseOptions();
            mOptions.filterLayout = filterLayout;

            return &mOptions;
        }
    };

    struct ConvTranspose2dOptions final : public Conv2dBaseOptions<wnn::ConvTranspose2dOptions> {
      public:
        std::vector<int32_t> outputPadding;
        std::vector<int32_t> outputSizes;
        wnn::ConvTranspose2dFilterOperandLayout filterLayout =
            wnn::ConvTranspose2dFilterOperandLayout::Iohw;

        const wnn::ConvTranspose2dOptions* AsPtr() {
            mOptions = GetBaseOptions();
            if (!outputPadding.empty()) {
                mOptions.outputPaddingCount = outputPadding.size();
                mOptions.outputPadding = outputPadding.data();
            }
            if (!outputSizes.empty()) {
                mOptions.outputSizesCount = outputSizes.size();
                mOptions.outputSizes = outputSizes.data();
            }
            mOptions.filterLayout = filterLayout;

            return &mOptions;
        }
    };

    struct SliceOptions {
        std::vector<int32_t> axes;
        const wnn::SliceOptions* AsPtr() {
            if (!axes.empty()) {
                mOptions.axesCount = axes.size();
                mOptions.axes = axes.data();
            }

            return &mOptions;
        }

      private:
        wnn::SliceOptions mOptions;
    };

    struct Pool2dOptions {
      public:
        std::vector<int32_t> windowDimensions;
        std::vector<int32_t> padding;
        std::vector<int32_t> strides;
        std::vector<int32_t> dilations;
        std::vector<int32_t> outputSizes;
        wnn::AutoPad autoPad = wnn::AutoPad::Explicit;
        wnn::InputOperandLayout layout = wnn::InputOperandLayout::Nchw;
        wnn::RoundingType roundinyType = wnn::RoundingType::Floor;

        const wnn::Pool2dOptions* AsPtr() {
            if (!windowDimensions.empty()) {
                mOptions.windowDimensionsCount = windowDimensions.size();
                mOptions.windowDimensions = windowDimensions.data();
            }
            if (!padding.empty()) {
                mOptions.paddingCount = padding.size();
                mOptions.padding = padding.data();
            }
            if (!strides.empty()) {
                mOptions.stridesCount = strides.size();
                mOptions.strides = strides.data();
            }
            if (!dilations.empty()) {
                mOptions.dilationsCount = dilations.size();
                mOptions.dilations = dilations.data();
            }
            if (!outputSizes.empty()) {
                mOptions.outputSizesCount = outputSizes.size();
                mOptions.outputSizes = outputSizes.data();
            }
            mOptions.layout = layout;
            mOptions.autoPad = autoPad;
            mOptions.roundingType = roundinyType;
            return &mOptions;
        }

      private:
        wnn::Pool2dOptions mOptions;
    };

    typedef struct {
        const std::string name;
        const wnn::Operand operand;
    } NamedOperand;

    wnn::Graph Build(const wnn::GraphBuilder& builder, const std::vector<NamedOperand>& outputs);

    template <typename T>
    struct NamedInput {
        const std::string name;
        const std::vector<T>& resource;
    };

    template <typename T>
    struct NamedOutput {
        const std::string name;
        std::vector<T>& resource;
    };

    template <typename T>
    wnn::ComputeGraphStatus Compute(const wnn::Graph& graph,
                                    const std::vector<NamedInput<T>>& inputs,
                                    const std::vector<NamedOutput<T>>& outputs) {
        if (graph.GetHandle() == nullptr) {
            dawn::ErrorLog() << "The graph is invaild.";
            return wnn::ComputeGraphStatus::Error;
        }

        // The `mlInputs` local variable to hold the input data util computing the graph.
        std::vector<wnn::Input> mlInputs;
        mlInputs.reserve(inputs.size());
        wnn::NamedInputs namedInputs = CreateCppNamedInputs();
        for (auto& input : inputs) {
            wnn::Input wnninput = {};
            wnninput.resource.arrayBufferView = {(void*)input.resource.data(),
                                                 input.resource.size() * sizeof(float)};
            mlInputs.push_back(wnninput);
            namedInputs.Set(input.name.c_str(), &mlInputs.back());
        }
        DAWN_ASSERT(outputs.size() > 0);
        // The `mlOutputs` local variable to hold the output data util computing the graph.
        std::vector<wnn::Resource> mlOutputs;
        mlOutputs.reserve(outputs.size());
        wnn::NamedOutputs namedOutputs = CreateCppNamedOutputs();
        for (auto& output : outputs) {
            wnn::Resource resource = {};
            resource.arrayBufferView.buffer = output.resource.data();
            resource.arrayBufferView.byteLength = output.resource.size() * sizeof(float);
            mlOutputs.push_back(resource);
            namedOutputs.Set(output.name.c_str(), &mlOutputs.back());
        }
        wnn::ComputeGraphStatus status = graph.Compute(namedInputs, namedOutputs);
        DoFlush();

        return status;
    }

    wnn::ComputeGraphStatus Compute(const wnn::Graph& graph,
                                    const std::vector<NamedInput<float>>& inputs,
                                    const std::vector<NamedOutput<float>>& outputs);

    template <class T>
    bool CheckValue(const std::vector<T>& value, const std::vector<T>& expectedValue) {
        if (value.size() != expectedValue.size()) {
            dawn::ErrorLog() << "The size of output data is expected as " << expectedValue.size()
                             << ", but got " << value.size();
            return false;
        }
        for (size_t i = 0; i < value.size(); ++i) {
            if (!Expected(value[i], expectedValue[i])) {
                dawn::ErrorLog() << "The output value at index " << i << " is expected as "
                                 << expectedValue[i] << ", but got " << value[i];
                return false;
            }
        }
        return true;
    }

    class Async {
      public:
        Async() : mDone(false) {
        }
        ~Async() = default;
        void Wait();
        void Finish();

      private:
        std::condition_variable mCondVar;
        std::mutex mMutex;
        bool mDone;
    };

    std::vector<std::string> ReadTopKLabel(const std::vector<size_t>& topKIndex,
                                           const std::string& labelPath);

    void SelectTopKData(std::vector<float>& outputData,
                        std::vector<size_t>& topKIndex,
                        std::vector<float>& topKData);

    void PrintResult(const std::vector<float>& output, const std::string& labelPath = "");

    bool LoadAndPreprocessImage(const ExampleBase* example, std::vector<float>& processedPixels);

    void ShowUsage();

    void PrintExexutionTime(
        std::vector<std::chrono::duration<double, std::milli>> executionTimeVector);

    const wnn::ContextOptions CreateContextOptions(const std::string& devicePreference = "default",
                                                   const std::string& powerPreference = "default");
}  // namespace utils

#endif  // WEBNN_NATIVE_EXAMPLES_SAMPLE_UTILS_H_