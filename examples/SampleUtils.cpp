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

static std::unique_ptr<webnn_native::Instance> instance;
ml::Context CreateCppContext(ml::ContextOptions const* options) {
    instance = std::make_unique<webnn_native::Instance>();
    WebnnProcTable backendProcs = webnn_native::GetProcs();
    webnnProcSetProcs(&backendProcs);
    MLContext context = instance->CreateContext(options);
    if (context) {
        return ml::Context::Acquire(context);
    }
    return ml::Context();
}

bool ExampleBase::ParseAndCheckExampleOptions(int argc, const char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp("-h", argv[i]) == 0) {
            utils::ShowUsage();
            return false;
        }
        if (strcmp("-i", argv[i]) == 0 && i + 1 < argc) {
            mImagePath = argv[i + 1];
        } else if (strcmp("-m", argv[i]) == 0 && i + 1 < argc) {
            mWeightsPath = argv[i + 1];
        } else if (strcmp("-l", argv[i]) == 0 && i + 1 < argc) {
            mLayout = argv[i + 1];
        } else if (strcmp("-n", argv[i]) == 0 && i + 1 < argc) {
            mNIter = atoi(argv[i + 1]);
        } else if (strcmp("-d", argv[i]) == 0 && i + 1 < argc) {
            mDevice = argv[i + 1];
        }
    }

    if (mImagePath.empty() || mWeightsPath.empty() || (mLayout != "nchw" && mLayout != "nhwc") ||
        mNIter < 1 || (mDevice != "gpu" && mDevice != "cpu" && mDevice != "default")) {
        dawn::ErrorLog() << "Invalid options.";
        utils::ShowUsage();
        return false;
    }
    return true;
}

bool Expected(float output, float expected) {
    return (fabs(output - expected) < 0.005f);
}

namespace utils {

    uint32_t SizeOfShape(const std::vector<int32_t>& dims) {
        uint32_t prod = 1;
        for (size_t i = 0; i < dims.size(); ++i)
            prod *= dims[i];
        return prod;
    }

    const ml::FusionOperator CreateActivationOperator(const ml::GraphBuilder& builder,
                                                      FusedActivation activation,
                                                      const void* options) {
        ml::FusionOperator activationOperator;
        switch (activation) {
            case FusedActivation::RELU:
                activationOperator = builder.ReluOperator();
                break;
            case FusedActivation::RELU6: {
                auto clampOptions = reinterpret_cast<ml::ClampOptions const*>(options);
                activationOperator = builder.ClampOperator(clampOptions);
                break;
            }
            case FusedActivation::SIGMOID:
                activationOperator = builder.SigmoidOperator();
                break;
            case FusedActivation::TANH:
                activationOperator = builder.TanhOperator();
                break;
            case FusedActivation::LEAKYRELU: {
                auto leakyReluOptions = reinterpret_cast<ml::LeakyReluOptions const*>(options);
                activationOperator = builder.LeakyReluOperator(leakyReluOptions);
                break;
            }
            default:
                dawn::ErrorLog() << "The activation is unsupported";
                DAWN_ASSERT(0);
        }
        return activationOperator;
    }

    const ml::Operand CreateActivationOperand(const ml::GraphBuilder& builder,
                                              const ml::Operand& input,
                                              FusedActivation activation,
                                              const void* options) {
        ml::Operand activationOperand;
        switch (activation) {
            case FusedActivation::RELU:
                activationOperand = builder.Relu(input);
                break;
            case FusedActivation::RELU6: {
                auto clampOptions = reinterpret_cast<ml::ClampOptions const*>(options);
                activationOperand = builder.Clamp(input, clampOptions);
                break;
            }
            case FusedActivation::SIGMOID:
                activationOperand = builder.Sigmoid(input);
                break;
            case FusedActivation::TANH:
                activationOperand = builder.Tanh(input);
                break;
            case FusedActivation::LEAKYRELU: {
                auto leakyReluOptions = reinterpret_cast<ml::LeakyReluOptions const*>(options);
                activationOperand = builder.LeakyRelu(input, leakyReluOptions);
                break;
            }
            default:
                dawn::ErrorLog() << "The activation is unsupported";
                DAWN_ASSERT(0);
        }
        return activationOperand;
    }

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
        ml::ArrayBufferView arrayBuffer = {const_cast<void*>(value), size};
        return builder.Constant(&desc, &arrayBuffer);
    }

    ml::Graph Build(const ml::GraphBuilder& builder, const std::vector<NamedOperand>& outputs) {
        ml::NamedOperands namedOperands = ml::CreateNamedOperands();
        for (auto& output : outputs) {
            namedOperands.Set(output.name.c_str(), output.operand);
        }
        return builder.Build(namedOperands);
    }

    ml::ComputeGraphStatus Compute(const ml::Graph& graph,
                                   const std::vector<NamedInput<float>>& inputs,
                                   const std::vector<NamedOutput<float>>& outputs) {
        return Compute<float>(graph, inputs, outputs);
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

    void PrintResult(const std::vector<float>& output, const std::string& labelPath) {
        std::vector<size_t> topKIndex(TOP_NUMBER);
        std::vector<float> topKData(TOP_NUMBER);
        SelectTopKData(const_cast<std::vector<float>&>(output), topKIndex, topKData);
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

    bool LoadAndPreprocessImage(const ExampleBase* example, std::vector<float>& processedPixels) {
        // Read an image.
        int imageWidth, imageHeight, imageChannels = 0;
        uint8_t* inputPixels =
            stbi_load(example->mImagePath.c_str(), &imageWidth, &imageHeight, &imageChannels, 0);
        if (inputPixels == 0) {
            dawn::ErrorLog() << "Failed to load and preprocess the image at "
                             << example->mImagePath;
            return false;
        }
        // Resize the image with model's input size
        const size_t imageSize = imageHeight * imageWidth * imageChannels;
        float* floatPixels = (float*)malloc(imageSize * sizeof(float));
        for (size_t i = 0; i < imageSize; ++i) {
            floatPixels[i] = inputPixels[i];
        }
        float* resizedPixels = (float*)malloc(example->mModelHeight * example->mModelWidth *
                                              example->mModelChannels * sizeof(float));
        stbir_resize_float(floatPixels, imageWidth, imageHeight, 0, resizedPixels,
                           example->mModelWidth, example->mModelHeight, 0, example->mModelChannels);

        // Reoder the image to NCHW/NHWC layout.
        for (size_t c = 0; c < example->mModelChannels; ++c) {
            for (size_t h = 0; h < example->mModelHeight; ++h) {
                for (size_t w = 0; w < example->mModelWidth; ++w) {
                    float value = resizedPixels[h * example->mModelWidth * example->mModelChannels +
                                                w * example->mModelChannels + c];
                    example->mNormalization ? value = value / 255 : value;
                    if (example->mLayout == "nchw") {
                        processedPixels[c * example->mModelHeight * example->mModelWidth +
                                        h * example->mModelWidth + w] =
                            (value - example->mMean[c]) / example->mStd[c];
                    } else {
                        processedPixels[h * example->mModelWidth * example->mModelChannels +
                                        w * example->mModelChannels + c] =
                            (value - example->mMean[c]) / example->mStd[c];
                    }
                }
            }
        }
        free(resizedPixels);
        free(floatPixels);
        return true;
    }

    void ShowUsage() {
        std::cout << std::endl;
        std::cout << "Example Options:" << std::endl;
        std::cout << "    -h                      "
                  << "Print this message." << std::endl;
        std::cout << "    -i \"<path>\"             "
                  << "Required. Path to an image." << std::endl;
        std::cout << "    -m \"<path>\"             "
                  << "Required. Path to the .npy files with trained weights/biases." << std::endl;
        std::cout
            << "    -l \"<layout>\"           "
            << "Optional. Specify the layout: \"nchw\" or \"nhwc\". The default value is \"nchw\"."
            << std::endl;
        std::cout << "    -n \"<integer>\"          "
                  << "Optional. Number of iterations. The default value is 1, and should not be "
                     "less than 1."
                  << std::endl;
        std::cout << "    -d \"<device>\"           "
                  << "Optional. Specify a target device: \"cpu\" or \"gpu\" or "
                     "\"default\" to infer on. The default value is \"default\"."
                  << std::endl;
    }

    void PrintExexutionTime(std::vector<TIME_TYPE> executionTime) {
        size_t nIter = executionTime.size();
        if (executionTime.size() > 1) {
            std::sort(executionTime.begin(), executionTime.end());
            TIME_TYPE medianExecutionTime =
                nIter % 2 != 0 ? executionTime[floor(nIter / 2)]
                               : (executionTime[nIter / 2 - 1] + executionTime[nIter / 2]) / 2;
            dawn::InfoLog() << "Median Execution Time of " << nIter
                            << " Iterations: " << medianExecutionTime.count() << " ms";
        } else {
            dawn::InfoLog() << "Execution Time: " << executionTime[0].count() << " ms";
        }
    }

    const ml::ContextOptions CreateContextOptions(const std::string& device) {
        ml::ContextOptions options;
        if (device == "cpu") {
            options.devicePreference = ml::DevicePreference::Cpu;
        } else if (device == "gpu") {
            options.devicePreference = ml::DevicePreference::Gpu;
        } else if (device == "default") {
            options.devicePreference = ml::DevicePreference::Default;
        } else {
            dawn::ErrorLog()
                << "Invalid options, only support devices: \"cpu\", \"gpu\" and \"default\".";
            DAWN_ASSERT(0);
        }
        return options;
    }
}  // namespace utils
