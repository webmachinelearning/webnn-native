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

#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "common/Log.h"
#include "examples/LeNet/LeNet.h"
#include "examples/LeNet/MnistUbyte.h"
#include "examples/SampleUtils.h"

const size_t TOP_NUMBER = 3;

void SelectTopKData(std::vector<float> outputData,
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

void PrintResult(ml::Result output) {
    const float* outputBuffer = static_cast<const float*>(output.Buffer());
    std::vector<float> outputData(outputBuffer, outputBuffer + output.BufferSize() / sizeof(float));
    std::vector<size_t> topKIndex(TOP_NUMBER);
    std::vector<float> topKData(TOP_NUMBER);
    SelectTopKData(outputData, topKIndex, topKData);

    std::cout << std::endl << "Prediction Result:" << std::endl;
    std::cout << "#"
              << "   "
              << "Label"
              << " "
              << "Probability" << std::endl;
    std::cout.precision(2);
    for (size_t i = 0; i < TOP_NUMBER; ++i) {
        std::cout << i << "   ";
        std::cout << std::left << std::setw(5) << std::fixed << topKIndex[i] << " ";
        std::cout << std::left << std::fixed << 100 * topKData[i] << "%" << std::endl;
    }
    std::cout << std::endl;
}

void ShowUsage() {
    std::cout << std::endl;
    std::cout << "LeNet [OPTIONs]" << std::endl << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "    -h                      "
              << "Print this message." << std::endl;
    std::cout << "    -i \"<path>\"             "
              << "Required. Path to an image." << std::endl;
    std::cout << "    -m \"<path>\"             "
              << "Required. Path to a .bin file with trained weights." << std::endl;
}

int main(int argc, const char* argv[]) {
    DumpMemoryLeaks();

    std::string imagePath, modelPath;
    for (int i = 1; i < argc; ++i) {
        if (strcmp("-h", argv[i]) == 0) {
            ShowUsage();
            return 0;
        }
        if (strcmp("-i", argv[i]) == 0 && i + 1 < argc) {
            imagePath = argv[i + 1];
        } else if (strcmp("-m", argv[i]) == 0 && i + 1 < argc) {
            modelPath = argv[i + 1];
        }
    }

    if (imagePath.empty() || modelPath.empty()) {
        dawn::ErrorLog() << "Invalid options.";
        ShowUsage();
        return -1;
    }

    MnistUbyte reader(imagePath);
    if (!reader.DataInitialized()) {
        dawn::ErrorLog() << "The input image is invalid.";
        return -1;
    }
    if (reader.Size() != 28 * 28) {
        dawn::ErrorLog() << "The expected size of the input image is 784 (28 * 28), but got "
                         << reader.Size() << ".";
        return -1;
    }

    LeNet lenet;
    if (!lenet.Load(modelPath)) {
        dawn::ErrorLog() << "Failed to load LeNet.";
        return -1;
    }
    std::vector<float> input(reader.GetData().get(), reader.GetData().get() + reader.Size());
    ml::Result result = lenet.Compute(input.data(), input.size() * sizeof(float));
    if (!result) {
        dawn::ErrorLog() << "Failed to compute LeNet.";
        return -1;
    }
    PrintResult(result);
    dawn::InfoLog() << "Done.";
}
