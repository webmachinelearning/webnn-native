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

uint32_t product(const std::vector<int32_t>& dims);

webnn::NeuralNetworkContext CreateCppNeuralNetworkContext();

webnn::NamedInputs CreateCppNamedInputs();

webnn::NamedOperands CreateCppNamedOperands();

webnn::NamedOutputs CreateCppNamedOutputs();

void DumpMemoryLeaks();

bool Expected(float output, float expected);

namespace utils {

    webnn::Operand BuildInput(const webnn::ModelBuilder& builder,
                              std::string name,
                              const std::vector<int32_t>& dimensions,
                              webnn::OperandType type = webnn::OperandType::Float32);

    webnn::Operand BuildConstant(const webnn::ModelBuilder& builder,
                                 const std::vector<int32_t>& dimensions,
                                 const void* value,
                                 size_t size,
                                 webnn::OperandType type = webnn::OperandType::Float32);

    struct Conv2dOptions {
      public:
        std::vector<int32_t> padding;
        std::vector<int32_t> strides;
        std::vector<int32_t> dilations;
        int32_t groups = 1;
        webnn::OperandLayout layout = webnn::OperandLayout::Nchw;

        const webnn::Conv2dOptions* AsPtr() {
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
            mOptions.layout = layout;
            return &mOptions;
        }

      private:
        webnn::Conv2dOptions mOptions;
    };

    struct Pool2dOptions {
      public:
        std::vector<int32_t> windowDimensions;
        std::vector<int32_t> padding;
        std::vector<int32_t> strides;
        std::vector<int32_t> dilations;
        webnn::OperandLayout layout = webnn::OperandLayout::Nchw;

        const webnn::Pool2dOptions* AsPtr() {
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
            return &mOptions;
        }

      private:
        webnn::Pool2dOptions mOptions;
    };

    typedef struct {
        const std::string& name;
        const webnn::Operand& operand;
    } NamedOutput;

    webnn::Model CreateModel(const webnn::ModelBuilder& builder,
                             const std::vector<NamedOutput>& outputs);

    webnn::Compilation AwaitCompile(const webnn::Model& model,
                                    webnn::CompilationOptions const* options = nullptr);

    typedef struct {
        const std::string& name;
        const webnn::Input& input;
    } NamedInput;

    webnn::NamedResults AwaitCompute(const webnn::Compilation& compilation,
                                     const std::vector<NamedInput>& inputs);

    bool CheckShape(const webnn::Result& result, const std::vector<int32_t>& expectedShape);

    template <class T>
    bool CheckValue(const webnn::Result& result, const std::vector<T>& expectedValue) {
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
}  // namespace utils

#endif  // WEBNN_NATIVE_EXAMPLES_SAMPLE_UTILS_H_
