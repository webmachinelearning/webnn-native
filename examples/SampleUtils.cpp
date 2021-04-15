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

#include "common/Assert.h"
#include "common/Log.h"

#include <webnn/webnn.h>
#include <webnn/webnn_cpp.h>
#include <webnn/webnn_proc.h>
#include <webnn_native/WebnnNative.h>
#if defined(_WIN32)
#    include <crtdbg.h>
#endif

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

void DumpMemoryLeaks() {
#if defined(_WIN32) && defined(_DEBUG)
    // Send all reports to STDOUT.
    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDOUT);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDOUT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDOUT);
    // Perform automatic leak checking at program exit through a call to _CrtDumpMemoryLeaks and
    // generate an error report if the application failed to free all the memory it allocated.
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
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
                             const std::vector<NamedOutput>& outputs) {
        typedef struct {
            Async async;
            ml::Graph graph;
        } BuildData;

        BuildData buildData;
        ml::NamedOperands namedOperands = ml::CreateNamedOperands();
        for (auto& output : outputs) {
            namedOperands.Set(output.name.c_str(), output.operand);
        }
        builder.Build(namedOperands,
            [](MLBuildStatus status, MLGraph impl, char const* message,
               void* userData) {
                BuildData* buildDataPtr = reinterpret_cast<BuildData*>(userData);
                DAWN_ASSERT(buildDataPtr);
                if (status != MLBuildStatus_Success) {
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
                                     const std::vector<NamedInput>& inputs) {
        typedef struct {
            Async async;
            ml::NamedResults results;
        } ComputeData;

        ComputeData computeData;
        ml::NamedInputs namedInputs = ml::CreateNamedInputs();
        for (auto& input : inputs) {
            namedInputs.Set(input.name.c_str(), &input.input);
        }
        graph.Compute(
            namedInputs,
            [](MLComputeStatus status, MLNamedResults impl, char const* message,
               void* userData) {
                ComputeData* computeDataPtr = reinterpret_cast<ComputeData*>(userData);
                DAWN_ASSERT(computeDataPtr);
                if (status != MLComputeStatus_Success) {
                    dawn::ErrorLog() << "Compute failed: " << message;
                } else {
                    computeDataPtr->results = computeDataPtr->results.Acquire(impl);
                }
                computeDataPtr->async.Finish();
                return;
            },
            &computeData, nullptr);
        computeData.async.Wait();
        return computeData.results;
    }

    bool CheckShape(const ml::Result& result, const std::vector<int32_t>& expectedShape) {
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

}  // namespace utils
