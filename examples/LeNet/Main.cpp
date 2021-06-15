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

#include "examples/LeNet/LeNet.h"

#include <algorithm>
#include <iostream>

#include "common/Log.h"
#include "examples/LeNet/MnistUbyte.h"

void ShowUsage() {
    std::cout << std::endl;
    std::cout << "LeNet [OPTION]" << std::endl << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "    -h                      "
              << "Print this message." << std::endl;
    std::cout << "    -i \"<path>\"             "
              << "Required. Path to an image." << std::endl;
    std::cout << "    -m \"<path>\"             "
              << "Required. Path to a .bin file with trained weights/biases." << std::endl;
    std::cout
        << "    -n \"<integer>\"          "
        << "Optional. Number of iterations. The default value is 1, and should not be less than 1."
        << std::endl;
}

int main(int argc, const char* argv[]) {
    std::string imagePath, modelPath;
    int nIter = 1;
    for (int i = 1; i < argc; ++i) {
        if (strcmp("-h", argv[i]) == 0) {
            ShowUsage();
            return 0;
        }
        if (strcmp("-i", argv[i]) == 0 && i + 1 < argc) {
            imagePath = argv[i + 1];
        } else if (strcmp("-m", argv[i]) == 0 && i + 1 < argc) {
            modelPath = argv[i + 1];
        } else if (strcmp("-n", argv[i]) == 0 && i + 1 < argc) {
            nIter = atoi(argv[i + 1]);
        }
    }

    if (imagePath.empty() || modelPath.empty() || nIter < 1) {
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

    const std::chrono::time_point<std::chrono::high_resolution_clock> compilationStartTime =
        std::chrono::high_resolution_clock::now();
    LeNet lenet;
    if (!lenet.Load(modelPath)) {
        dawn::ErrorLog() << "Failed to load LeNet.";
        return -1;
    }

    const std::chrono::duration<double, std::milli> compilationElapsedTime =
        std::chrono::high_resolution_clock::now() - compilationStartTime;
    dawn::InfoLog() << "Compilation Time: " << compilationElapsedTime.count() << " ms";
    ml::Result result;
    std::vector<float> input(reader.GetData().get(), reader.GetData().get() + reader.Size());
    std::vector<std::chrono::duration<double, std::milli>> executionTimeVector;

    for (int i = 0; i < nIter; ++i) {
        std::chrono::time_point<std::chrono::high_resolution_clock> executionStartTime =
            std::chrono::high_resolution_clock::now();
        result = lenet.Compute(input.data(), input.size() * sizeof(float));
        if (!result) {
            dawn::ErrorLog() << "Failed to compute LeNet.";
            return -1;
        }
        executionTimeVector.push_back(std::chrono::high_resolution_clock::now() -
                                      executionStartTime);
    }

    if (nIter > 1) {
        std::sort(executionTimeVector.begin(), executionTimeVector.end());
        std::chrono::duration<double, std::milli> medianExecutionTime =
            nIter % 2 != 0
                ? executionTimeVector[floor(nIter / 2)]
                : (executionTimeVector[nIter / 2 - 1] + executionTimeVector[nIter / 2]) / 2;
        dawn::InfoLog() << "Median Execution Time of " << nIter
                        << " Iterations: " << medianExecutionTime.count() << " ms";
    } else {
        dawn::InfoLog() << "Execution Time: " << executionTimeVector[0].count() << " ms";
    }
    utils::PrintResult(result);
    dawn::InfoLog() << "Done.";
}
