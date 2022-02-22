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

#include "examples/ResNet/ResNet.h"

int main(int argc, const char* argv[]) {
    // Set input options for the example.
    ResNet resnet;
    if (!resnet.ParseAndCheckExampleOptions(argc, argv)) {
        return -1;
    }

    // Pre-process the input image.
    std::vector<float> processedPixels(resnet.mModelHeight * resnet.mModelWidth *
                                       resnet.mModelChannels);
    if (!utils::LoadAndPreprocessImage(&resnet, processedPixels)) {
        return -1;
    }

    // Create a graph with weights and biases from .npy files.
    const wnn::ContextOptions options =
        utils::CreateContextOptions(resnet.mDevicePreference, resnet.mPowerPreference);
    wnn::Context context = CreateCppContext(&options);
    context.SetUncapturedErrorCallback(
        [](WNNErrorType type, char const* message, void* userData) {
            if (type != WNNErrorType_NoError) {
                dawn::ErrorLog() << "Error type is " << type << ", message is " << message;
            }
        },
        &resnet);
    wnn::GraphBuilder builder = wnn::CreateGraphBuilder(context);
    wnn::Operand output =
        resnet.mLayout == "nchw" ? resnet.LoadNCHW(builder) : resnet.LoadNHWC(builder);

    // Build the graph.
    const std::chrono::time_point<std::chrono::high_resolution_clock> compilationStartTime =
        std::chrono::high_resolution_clock::now();
    wnn::Graph graph = utils::Build(builder, {{"output", output}});
    if (!graph) {
        dawn::ErrorLog() << "Failed to build graph.";
        return -1;
    }

    const TIME_TYPE compilationElapsedTime =
        std::chrono::high_resolution_clock::now() - compilationStartTime;
    dawn::InfoLog() << "Compilation Time: " << compilationElapsedTime.count() << " ms";

    // Compute the graph.
    std::vector<float> result(utils::SizeOfShape(resnet.mOutputShape));
    // Do the first inference for warming up if nIter > 1.
    if (resnet.mNIter > 1) {
        wnn::ComputeGraphStatus status =
            utils::Compute(graph, {{"input", processedPixels}}, {{"output", result}});
        DAWN_ASSERT(status == wnn::ComputeGraphStatus::Success);
    }

    std::vector<TIME_TYPE> executionTime;
    for (int i = 0; i < resnet.mNIter; ++i) {
        std::chrono::time_point<std::chrono::high_resolution_clock> executionStartTime =
            std::chrono::high_resolution_clock::now();
        wnn::ComputeGraphStatus status =
            utils::Compute(graph, {{"input", processedPixels}}, {{"output", result}});
        DAWN_ASSERT(status == wnn::ComputeGraphStatus::Success);
        executionTime.push_back(std::chrono::high_resolution_clock::now() - executionStartTime);
    }

    // Print the result.
    utils::PrintExexutionTime(executionTime);
    utils::PrintResult(result, resnet.mLabelPath);
    dawn::InfoLog() << "Done.";
}