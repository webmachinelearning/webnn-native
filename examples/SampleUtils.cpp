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

#include "SampleUtils.h"

#include <webnn/webnn.h>
#include <webnn/webnn_cpp.h>
#include <webnn/webnn_proc.h>
#include <webnn_native/WebnnNative.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

#include "common/Assert.h"
#include "common/Log.h"

uint32_t product(const std::vector<int32_t>& dims) {
    uint32_t prod = 1;
    for (size_t i = 0; i < dims.size(); ++i)
        prod *= dims[i];
    return prod;
}

ml::Context CreateCppContext() {
    WebnnProcTable backendProcs = webnn_native::GetProcs();
    webnnProcSetProcs(&backendProcs);
    MLContext context = webnn_native::CreateContext();
    if (context) {
        return ml::Context::Acquire(context);
    }
    return ml::Context();
}

bool Expected(float output, float expected) {
    return (fabs(output - expected) < 0.005f);
}

namespace utils {

    ml::Operand BuildInput(const ml::GraphBuilder& builder,
                           std::string name,
                           const std::vector<int32_t>& dimensions,
                           ml::OperandType type) {
        ml::OperandDescriptor desc = {type, dimensions.data(), (uint32_t)dimensions.size()};
        return builder.Input(name.c_str(), &desc);
    }

    ml::Operand BuildConstant(const ml::GraphBuilder& builder,
                              const std::vector<int32_t>& dimensions,
                              const void* value,
                              size_t size,
                              ml::OperandType type) {
        ml::OperandDescriptor desc = {type, dimensions.data(), (uint32_t)dimensions.size()};
        return builder.Constant(&desc, value, size);
    }

    ml::Graph AwaitBuild(const ml::GraphBuilder& builder,
                         const std::vector<NamedOperand>& outputs) {
        typedef struct {
            Async async;
            ml::Graph graph;
        } BuildData;

        BuildData buildData;
        ml::NamedOperands namedOperands = ml::CreateNamedOperands();
        for (auto& output : outputs) {
            namedOperands.Set(output.name.c_str(), output.operand);
        }
        builder.Build(
            namedOperands,
            [](MLBuildGraphStatus status, MLGraph impl, char const* message, void* userData) {
                BuildData* buildDataPtr = reinterpret_cast<BuildData*>(userData);
                DAWN_ASSERT(buildDataPtr);
                if (status != MLBuildGraphStatus_Success) {
                    dawn::ErrorLog() << "Compute failed: " << message;
                } else {
                    buildDataPtr->graph = buildDataPtr->graph.Acquire(impl);
                }
                buildDataPtr->async.Finish();
                return;
            },
            &buildData);
        buildData.async.Wait();
        return buildData.graph;
    }

    ml::NamedResults AwaitCompute(const ml::Graph& graph,
                                  const std::vector<NamedInput>& inputs,
                                  const std::vector<NamedOutput>& outputs) {
        if (graph.GetHandle() == nullptr) {
            return ml::NamedResults();
        }
        typedef struct {
            Async async;
            ml::NamedResults results;
        } ComputeData;

        ComputeData computeData;
        ml::NamedInputs namedInputs = ml::CreateNamedInputs();
        for (auto& input : inputs) {
            namedInputs.Set(input.name.c_str(), &input.input);
        }
        ml::NamedOutputs namedOutputs;
        if (outputs.size() > 0) {
            namedOutputs = ml::CreateNamedOutputs();
            for (auto& output : outputs) {
                namedOutputs.Set(output.name.c_str(), &output.output);
            }
        }
        graph.Compute(
            namedInputs,
            [](MLComputeGraphStatus status, MLNamedResults impl, char const* message,
               void* userData) {
                ComputeData* computeDataPtr = reinterpret_cast<ComputeData*>(userData);
                DAWN_ASSERT(computeDataPtr);
                if (status != MLComputeGraphStatus_Success) {
                    dawn::ErrorLog() << "Compute failed: " << message;
                } else {
                    computeDataPtr->results = computeDataPtr->results.Acquire(impl);
                }
                computeDataPtr->async.Finish();
                return;
            },
            &computeData, namedOutputs ? namedOutputs : nullptr);
        computeData.async.Wait();
        return computeData.results;
    }

    bool CheckShape(const ml::Result& result, const std::vector<int32_t>& expectedShape) {
        if (result.GetHandle() == nullptr) {
            return false;
        }
        if (expectedShape.size() != result.DimensionsSize()) {
            dawn::ErrorLog() << "The output rank is expected as " << expectedShape.size()
                             << ", but got " << result.DimensionsSize();
            return false;
        } else {
            for (size_t i = 0; i < result.DimensionsSize(); ++i) {
                int32_t dimension = result.Dimensions()[i];
                if (!Expected(expectedShape[i], dimension)) {
                    dawn::ErrorLog() << "The output dimension of axis " << i << " is expected as "
                                     << expectedShape[i] << ", but got " << dimension;
                    return false;
                }
            }
        }
        return true;
    }

    void Async::Wait() {
        // Wait for async callback.
        std::unique_lock<std::mutex> lock(mMutex);
        bool& done = mDone;
        mCondVar.wait(lock, [&done] { return done; });
        mDone = false;
    }

    void Async::Finish() {
        std::lock_guard<std::mutex> lock(mMutex);
        mDone = true;
        mCondVar.notify_one();
        return;
    }

    std::vector<std::string> ReadTopKLabel(const std::vector<size_t>& topKIndex,
                                           const std::string& labelPath) {
        if (labelPath.empty()) {
            return {};
        }
        std::vector<std::string> topKLabel, labeList;
        std::ifstream file;
        file.open(labelPath);
        if (!file) {
            dawn::ErrorLog() << "Failed to open label file at " << labelPath << ".";
            return {};
        }
        std::string line;
        while (getline(file, line)) {
            labeList.push_back(line);
        }
        file.close();

        if (labeList.size() >= topKIndex.size()) {
            for (size_t i = 0; i < topKIndex.size(); ++i) {
                topKLabel.push_back(labeList[topKIndex[i]]);
            }
        }
        return topKLabel;
    }

    const size_t TOP_NUMBER = 3;
    void SelectTopKData(std::vector<float>& outputData,
                        std::vector<size_t>& topKIndex,
                        std::vector<float>& topKData) {
        std::vector<size_t> indexes(outputData.size());
        std::iota(std::begin(indexes), std::end(indexes), 0);
        std::partial_sort(
            std::begin(indexes), std::begin(indexes) + TOP_NUMBER, std::end(indexes),
            [&outputData](unsigned l, unsigned r) { return outputData[l] > outputData[r]; });
        std::sort(outputData.rbegin(), outputData.rend());

        for (size_t i = 0; i < TOP_NUMBER; ++i) {
            topKIndex[i] = indexes[i];
            topKData[i] = outputData[i];
        }
    }

    void PrintResult(ml::Result output, const std::string& labelPath) {
        const float* outputBuffer = static_cast<const float*>(output.Buffer());
        std::vector<float> outputData(outputBuffer,
                                      outputBuffer + output.BufferSize() / sizeof(float));
        std::vector<size_t> topKIndex(TOP_NUMBER);
        std::vector<float> topKData(TOP_NUMBER);
        SelectTopKData(outputData, topKIndex, topKData);
        std::vector<std::string> topKLabel = ReadTopKLabel(topKIndex, labelPath);
        std::cout << std::endl << "Prediction Result:" << std::endl;
        std::cout << "#"
                  << "   "
                  << "Probability"
                  << "   "
                  << "Label" << std::endl;
        std::cout.precision(2);
        for (size_t i = 0; i < TOP_NUMBER; ++i) {
            std::cout << i << "   ";
            int w = 10 - std::to_string(100 * topKData[i]).find(".");
            std::cout << std::left << std::fixed << 100 * topKData[i] << "%" << std::setw(w) << " ";
            if (topKLabel.empty()) {
                std::cout << std::left << topKIndex[i] << std::endl;
            } else {
                std::cout << std::left << topKLabel[i] << std::endl;
            }
        }
        std::cout << std::endl;
    }

    float* LoadAndPreprocessImage(const std::string& imagePath,
                                  const ImagePreprocessOptions& options) {
        // Read an image.
        int imageWidth, imageHeight, imageChannels = 0;
        uint8_t* inputPixels =
            stbi_load(imagePath.c_str(), &imageWidth, &imageHeight, &imageChannels, 0);
        if (inputPixels == 0) {
            return nullptr;
        }
        // Resize the image with model's input size
        const size_t imageSize = imageHeight * imageWidth * imageChannels;
        float* floatPixels = (float*)malloc(imageSize * sizeof(float));
        for (size_t i = 0; i < imageSize; ++i) {
            floatPixels[i] = inputPixels[i];
        }
        float* resizedPixels = (float*)malloc(options.modelSize * sizeof(float));
        stbir_resize_float(floatPixels, imageWidth, imageHeight, 0, resizedPixels,
                           options.modelWidth, options.modelHeight, 0, options.modelChannels);

        // Reoder the image to NCHW/NHWC layout.
        float* processedPixels = (float*)malloc(options.modelSize * sizeof(float));
        for (size_t c = 0; c < options.modelChannels; ++c) {
            for (size_t h = 0; h < options.modelHeight; ++h) {
                for (size_t w = 0; w < options.modelWidth; ++w) {
                    float value = resizedPixels[h * options.modelWidth * options.modelChannels +
                                                w * options.modelChannels + c];
                    options.normalization ? value = value / 255 : value;
                    if (options.nchw) {
                        processedPixels[c * options.modelHeight * options.modelWidth +
                                        h * options.modelWidth + w] =
                            (value - options.mean[c]) / options.std[c];
                    } else {
                        processedPixels[h * options.modelWidth * options.modelChannels +
                                        w * options.modelChannels + c] =
                            (value - options.mean[c]) / options.std[c];
                    }
                }
            }
        }
        free(resizedPixels);
        free(floatPixels);
        return processedPixels;
    }

}  // namespace utils
