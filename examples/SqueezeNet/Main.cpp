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

#include "examples/SqueezeNet/SqueezeNet.h"

#include <algorithm>
#include <iostream>
#include <string>

void ShowUsage() {
    std::cout << std::endl;
    std::cout << "SqueezeNet [OPTION]" << std::endl << std::endl;
    std::cout << "Options:" << std::endl;
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
    std::cout
        << "    -n \"<integer>\"          "
        << "Optional. Number of iterations. The default value is 1, and should not be less than 1."
        << std::endl;
}

int main(int argc, const char* argv[]) {
    std::string imagePath, weightsPath, layout = "nchw";
    int nIter = 1;
    for (int i = 1; i < argc; ++i) {
        if (strcmp("-h", argv[i]) == 0) {
            ShowUsage();
            return 0;
        }
        if (strcmp("-i", argv[i]) == 0 && i + 1 < argc) {
            imagePath = argv[i + 1];
        } else if (strcmp("-m", argv[i]) == 0 && i + 1 < argc) {
            weightsPath = argv[i + 1];
        } else if (strcmp("-l", argv[i]) == 0 && i + 1 < argc) {
            layout = argv[i + 1];
        } else if (strcmp("-n", argv[i]) == 0 && i + 1 < argc) {
            nIter = atoi(argv[i + 1]);
        }
    }

    if (imagePath.empty() || weightsPath.empty() || (layout != "nchw" && layout != "nhwc") ||
        nIter < 1) {
        dawn::ErrorLog() << "Invalid options.";
        ShowUsage();
        return -1;
    }

    utils::ImagePreprocessOptions preOptions;
    preOptions.nchw = layout == "nchw";
    preOptions.modelHeight = 224;
    preOptions.modelWidth = 224;
    preOptions.modelChannels = 3;
    preOptions.modelSize =
        preOptions.modelHeight * preOptions.modelWidth * preOptions.modelChannels;

    preOptions.normalization = preOptions.nchw ? true : false;
    preOptions.nchw ? preOptions.mean = {0.485, 0.456, 0.406}
                    : preOptions.mean = {127.5, 127.5, 127.5};
    preOptions.nchw ? preOptions.std = {0.229, 0.224, 0.225}
                    : preOptions.std = {127.5, 127.5, 127.5};

    float* processedPixels = utils::LoadAndPreprocessImage(imagePath, preOptions);
    if (processedPixels == nullptr) {
        dawn::ErrorLog() << "Failed to load and preprocess the image at " << imagePath;
        return -1;
    }

    const std::chrono::time_point<std::chrono::high_resolution_clock> compilationStartTime =
        std::chrono::high_resolution_clock::now();
    // Create a model with weights and biases from .npy files.
    SqueezeNet squeezenet(preOptions.nchw);
    if (preOptions.nchw) {
        if (!squeezenet.LoadNCHW(weightsPath)) {
            dawn::ErrorLog() << "Failed to load SqueezeNet for NCHW.";
            return -1;
        }
    } else {
        if (!squeezenet.LoadNHWC(weightsPath)) {
            dawn::ErrorLog() << "Failed to load SqueezeNet for NHWC.";
            return -1;
        }
    }

    const std::chrono::duration<double, std::milli> compilationElapsedTime =
        std::chrono::high_resolution_clock::now() - compilationStartTime;
    dawn::InfoLog() << "Compilation Time: " << compilationElapsedTime.count() << " ms";
    ml::Result result;
    std::vector<float> input(processedPixels, processedPixels + preOptions.modelSize);
    std::vector<std::chrono::duration<double, std::milli>> executionTimeVector;

    for (int i = 0; i < nIter; ++i) {
        std::chrono::time_point<std::chrono::high_resolution_clock> executionStartTime =
            std::chrono::high_resolution_clock::now();
        result = squeezenet.Compute(input.data(), input.size() * sizeof(float));
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

    free(processedPixels);
    std::string labelPath =
        preOptions.nchw ? "examples/labels/labels1000.txt" : "examples/labels/labels1001.txt";
    utils::PrintResult(result, labelPath);
    dawn::InfoLog() << "Done.";
}