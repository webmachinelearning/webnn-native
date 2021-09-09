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
    std::string mDevice = "default";
    bool mFused = true;
};

ml::Context CreateCppContext(ml::ContextOptions const* options = nullptr);

bool Expected(float output, float expected);

namespace utils {

    uint32_t SizeOfShape(const std::vector<int32_t>& dims);

    enum FusedActivation { NONE, RELU, RELU6, SIGMOID, LEAKYRELU };

    ml::ClampOptions CreateClampOptions(const ml::GraphBuilder& builder,
                                        const std::vector<int32_t>& minShape,
                                        const std::vector<float>& minValue,
                                        const std::vector<int32_t>& maxShape,
                                        const std::vector<float>& maxValue);

    const ml::Operator CreateActivationOperator(const ml::GraphBuilder& builder,
                                                FusedActivation activation = FusedActivation::NONE,
                                                const void* options = nullptr);

    const ml::Operand CreateActivationOperand(const ml::GraphBuilder& builder,
                                              const ml::Operand& input,
                                              FusedActivation activation,
                                              const void* options = nullptr);

    ml::Operand BuildInput(const ml::GraphBuilder& builder,
                           std::string name,
                           const std::vector<int32_t>& dimensions,
                           ml::OperandType type = ml::OperandType::Float32);

    ml::Operand BuildConstant(const ml::GraphBuilder& builder,
                              const std::vector<int32_t>& dimensions,
                              const void* value,
                              size_t size,
                              ml::OperandType type = ml::OperandType::Float32);

    struct Conv2dOptions {
      public:
        std::vector<int32_t> padding;
        std::vector<int32_t> strides;
        std::vector<int32_t> dilations;
        std::vector<int32_t> outputPadding;
        std::vector<int32_t> outputSizes;
        ml::AutoPad autoPad = ml::AutoPad::Explicit;
        bool transpose = false;
        int32_t groups = 1;
        ml::InputOperandLayout inputLayout = ml::InputOperandLayout::Nchw;
        ml::FilterOperandLayout filterLayout = ml::FilterOperandLayout::Oihw;
        ml::Operand bias;
        ml::Operator activation;

        const ml::Conv2dOptions* AsPtr() {
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
            if (!outputPadding.empty()) {
                mOptions.outputPaddingCount = outputPadding.size();
                mOptions.outputPadding = outputPadding.data();
            }
            if (!outputSizes.empty()) {
                mOptions.outputSizesCount = outputSizes.size();
                mOptions.outputSizes = outputSizes.data();
            }
            mOptions.transpose = transpose;
            mOptions.groups = groups;
            mOptions.autoPad = autoPad;
            mOptions.inputLayout = inputLayout;
            mOptions.filterLayout = filterLayout;
            mOptions.bias = bias;
            mOptions.activation = activation;

            return &mOptions;
        }

      private:
        ml::Conv2dOptions mOptions;
    };

    struct Pool2dOptions {
      public:
        std::vector<int32_t> windowDimensions;
        std::vector<int32_t> padding;
        std::vector<int32_t> strides;
        std::vector<int32_t> dilations;
        ml::AutoPad autoPad = ml::AutoPad::Explicit;
        ml::InputOperandLayout layout = ml::InputOperandLayout::Nchw;

        const ml::Pool2dOptions* AsPtr() {
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
            mOptions.layout = layout;
            mOptions.autoPad = autoPad;
            return &mOptions;
        }

      private:
        ml::Pool2dOptions mOptions;
    };

    typedef struct {
        const std::string name;
        const ml::Operand operand;
    } NamedOperand;

    ml::Graph Build(const ml::GraphBuilder& builder, const std::vector<NamedOperand>& outputs);

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
    ml::ComputeGraphStatus Compute(const ml::Graph& graph,
                                   const std::vector<NamedInput<T>>& inputs,
                                   const std::vector<NamedOutput<T>>& outputs) {
        if (graph.GetHandle() == nullptr) {
            dawn::ErrorLog() << "The graph is invaild.";
            return ml::ComputeGraphStatus::Error;
        }

        // The `mlInputs` local variable to hold the input data util computing the graph.
        std::vector<ml::Input> mlInputs;
        mlInputs.reserve(inputs.size());
        ml::NamedInputs namedInputs = ml::CreateNamedInputs();
        for (auto& input : inputs) {
            const ml::ArrayBufferView resource = {(void*)input.resource.data(),
                                                  input.resource.size() * sizeof(float)};
            mlInputs.push_back({resource});
            namedInputs.Set(input.name.c_str(), &mlInputs.back());
        }
        DAWN_ASSERT(outputs.size() > 0);
        // The `mlOutputs` local variable to hold the output data util computing the graph.
        std::vector<ml::ArrayBufferView> mlOutputs;
        mlOutputs.reserve(outputs.size());
        ml::NamedOutputs namedOutputs = ml::CreateNamedOutputs();
        for (auto& output : outputs) {
            const ml::ArrayBufferView resource = {output.resource.data(),
                                                  output.resource.size() * sizeof(float)};
            mlOutputs.push_back(resource);
            namedOutputs.Set(output.name.c_str(), &mlOutputs.back());
        }
        return graph.Compute(namedInputs, namedOutputs);
    }

    ml::ComputeGraphStatus Compute(const ml::Graph& graph,
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

    const ml::ContextOptions CreateContextOptions(const std::string& device = "default");
}  // namespace utils

#endif  // WEBNN_NATIVE_EXAMPLES_SAMPLE_UTILS_H_
