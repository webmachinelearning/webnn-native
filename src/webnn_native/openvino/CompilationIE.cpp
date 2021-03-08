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

#include "webnn_native/openvino/CompilationIE.h"

#include <vector>

#include "common/Log.h"
#include "webnn_native/Error.h"
#include "webnn_native/ErrorData.h"
#include "webnn_native/NamedResults.h"
#include "webnn_native/Operand.h"
#include "webnn_native/Result.h"
#include "webnn_native/openvino/ErrorIE.h"
#include "webnn_native/openvino/ienn_symbol_table/ienn_symbol_table.h"

#define DAWN_CALLBACK_TRY(code, messages)                                     \
    {                                                                         \
        MaybeError maybeError = CheckStatusCode(code, messages);              \
        if (maybeError.IsError()) {                                           \
            std::unique_ptr<ErrorData> error = maybeError.AcquireError();     \
            callback(status, nullptr, error->GetMessage().c_str(), userdata); \
            return;                                                           \
        }                                                                     \
    }                                                                         \
    for (;;)                                                                  \
    break

namespace webnn_native { namespace ie {

    class Result : public ResultBase {
      public:
        using ResultBase::Reference;
        ~Result() {
            ie_compilation_free_buffer(&mBuffer);
        }
    };

    Compilation::Compilation(Ref<Model> model) : mModel(model) {
    }

    Compilation::~Compilation() {
        IE(ie_compilation_free)(mIeCompilation);
    }

    void Compilation::Compile(WebnnCompileCallback callback,
                              void* userdata,
                              CompilationOptions const* options) {
        // We may leverage https://dawn-review.googlesource.com/c/dawn/+/36360 to
        // implement async compilation as standle-alone component.
        WebnnCompileStatus status = WebnnCompileStatus_Error;
        // Create compilation for IE backend.
        IEStatusCode code =
            IE(ie_create_compilation)(mModel->GetInferenceEngineModel(), &mIeCompilation);
        DAWN_CALLBACK_TRY(code, "IE create compilation");
        status = WebnnCompileStatus_Success;
        callback(status, reinterpret_cast<WebnnCompilation>(this), nullptr, userdata);
    }

    void Compilation::ComputeImpl(NamedInputsBase* inputs,
                                  WebnnComputeCallback callback,
                                  void* userdata,
                                  NamedOutputsBase* outputs) {
        WebnnComputeStatus status = WebnnComputeStatus_Error;
        // Set input data to nGraph.
        for (auto& input : inputs->GetRecords()) {
            ie_operand_t ieOperand;
            ieOperand.name = const_cast<char*>(mModel->mInputIdMap[input.first].c_str());
            IEStatusCode code = IE(ie_compilation_set_input)(
                mIeCompilation, &ieOperand, input.second->buffer, input.second->size);
            DAWN_CALLBACK_TRY(code, "IE set input");
        }

        // Compute the compiled model.
        IEStatusCode code = IE(ie_compilation_compute)(mIeCompilation);
        DAWN_CALLBACK_TRY(code, "IE compute model");
        // Get Data from nGraph with output.
        Ref<NamedResultsBase> results = AcquireRef(new NamedResultsBase());
        size_t outputNumber = mModel->GetOutputsNumber();
        for (size_t i = 0; i < outputNumber; ++i) {
            std::string outputId = mModel->GetOutputId(i);
            void* outputBuffer;
            size_t bufferLength;
            IEStatusCode code = IE(ie_compilation_get_buffer)(mIeCompilation, outputId.data(),
                                                              &outputBuffer, &bufferLength);
            DAWN_CALLBACK_TRY(code, "IE get buffer");
            ie_dimensions_t ieDimensions;
            code =
                IE(ie_compilation_get_dimensions)(mIeCompilation, outputId.data(), &ieDimensions);
            DAWN_CALLBACK_TRY(code, "IE get dimensions");
            std::vector<int32_t> dimensions(ieDimensions.dims,
                                            ieDimensions.dims + ieDimensions.ranks);
            code = IE(ie_compilation_free_dimensions)(&ieDimensions);
            Ref<ResultBase> result =
                AcquireRef(new Result::ResultBase(outputBuffer, bufferLength, dimensions));
            std::string outputName = mModel->mOutputNameMap[outputId];
            results->Set(outputName.c_str(), result.Detach());
            if (outputs != nullptr) {
                const Output* output = outputs->GetRecords().at(outputName);
                ie_operand_t ieOperand;
                ieOperand.name = const_cast<char*>(outputId.c_str());
                IEStatusCode code = IE(ie_compilation_get_output)(mIeCompilation, &ieOperand,
                                                                  output->buffer, output->size);
                DAWN_CALLBACK_TRY(code, "IE get output");
            }
        }
        status = WebnnComputeStatus_Success;
        callback(status, reinterpret_cast<WebnnNamedResults>(results.Detach()), nullptr, userdata);
        return;
    }

}}  // namespace webnn_native::ie
