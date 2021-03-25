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

uint32_t product(const std::vector<int32_t>& dims);

ml::Context CreateCppContext();

bool Expected(float output, float expected);

namespace utils {

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
        int32_t groups = 1;
        ml::AutoPad autoPad = ml::AutoPad::Explicit;
        ml::InputOperandLayout inputLayout = ml::InputOperandLayout::Nchw;
        ml::FilterOperandLayout filterLayout = ml::FilterOperandLayout::Oihw;

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
            mOptions.groups = groups;
            mOptions.autoPad = autoPad;
            mOptions.inputLayout = inputLayout;
            mOptions.filterLayout = filterLayout;
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
        const std::string& name;
        const ml::Operand& operand;
    } NamedOperand;

    ml::Graph AwaitBuild(const ml::GraphBuilder& builder, const std::vector<NamedOperand>& outputs);

    typedef struct {
        const std::string name;
        const ml::Input input;
    } NamedInput;

    typedef struct {
        const std::string name;
        const ml::Output output;
    } NamedOutput;

    ml::NamedResults AwaitCompute(const ml::Graph& compilation,
                                  const std::vector<NamedInput>& inputs,
                                  const std::vector<NamedOutput>& outputs = {});

    bool CheckShape(const ml::Result& result, const std::vector<int32_t>& expectedShape);

    template <class T>
    bool CheckValue(const ml::Result& result, const std::vector<T>& expectedValue) {
        if (result.GetHandle() == nullptr) {
            return false;
        }
        size_t size = result.BufferSize() / sizeof(T);
        if (size != expectedValue.size()) {
            dawn::ErrorLog() << "The size of output data is expected as " << expectedValue.size()
                             << ", but got " << size;
            return false;
        }
        for (size_t i = 0; i < result.BufferSize() / sizeof(T); ++i) {
            T value = static_cast<const T*>(result.Buffer())[i];
            if (!Expected(value, expectedValue[i])) {
                dawn::ErrorLog() << "The output value at index " << i << " is expected as "
                                 << expectedValue[i] << ", but got " << value;
                return false;
            }
        }
        return true;
    }

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

    struct ImagePreprocessOptions {
        bool nchw = true;
        bool normalization = false;
        size_t modelHeight;
        size_t modelWidth;
        size_t modelChannels;
        size_t modelSize;
        std::vector<float> mean = {0, 0, 0};  // Average values of pixels on channels.
        std::vector<float> std = {1, 1, 1};   // Variance values of pixels on channels.
        std::string channelScheme = "RGB";
    };

    std::vector<std::string> ReadTopKLabel(const std::vector<size_t>& topKIndex,
                                           const std::string& labelPath);

    void SelectTopKData(std::vector<float>& outputData,
                        std::vector<size_t>& topKIndex,
                        std::vector<float>& topKData);

    void PrintResult(ml::Result output, const std::string& labelPath = "");

    float* LoadAndPreprocessImage(const std::string& imagePath,
                                  const ImagePreprocessOptions& options);
}  // namespace utils

#endif  // WEBNN_NATIVE_EXAMPLES_SAMPLE_UTILS_H_
