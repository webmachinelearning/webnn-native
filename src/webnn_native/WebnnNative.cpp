// Copyright 2018 The Dawn Authors
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

#include "webnn_native/WebnnNative.h"

#include <memory>

#include "common/Assert.h"
#include "webnn_native/GraphBuilder.h"

#if defined(_WIN32)
#    include <crtdbg.h>
#endif

// Contains the entry-points into webnn_native
namespace webnn_native {

    namespace {

        void DumpMemoryLeaks() {
#if defined(_WIN32) && defined(_DEBUG)
            // Send all reports to STDOUT.
            _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
            _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDOUT);
            _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
            _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDOUT);
            _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
            _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDOUT);
            // Perform automatic leak checking at program exit through a call to _CrtDumpMemoryLeaks
            // and generate an error report if the application failed to free all the memory it
            // allocated.
            _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
        }
    }  // namespace

    const WebnnProcTable& GetProcsAutogen();

    const WebnnProcTable& GetProcs() {
        DumpMemoryLeaks();
        return GetProcsAutogen();
    }

    namespace null {
        ContextBase* Create(MLContextOptions const* options);
    }
    namespace ie {
        ContextBase* Create(MLContextOptions const* options);
    }
    namespace dml {
        ContextBase* Create(MLContextOptions const* options);
    }
    namespace onednn {
        ContextBase* Create();
    }
    namespace xnnpack {
        ContextBase* Create();
    }

    // Should put the default null backend at the end.
    MLContext CreateContext(MLContextOptions const* options) {
#if defined(WEBNN_ENABLE_BACKEND_OPENVINO)
        return reinterpret_cast<MLContext>(ie::Create(options));
#elif defined(WEBNN_ENABLE_BACKEND_DML)
        return reinterpret_cast<MLContext>(dml::Create(options));
#elif defined(WEBNN_ENABLE_BACKEND_ONEDNN)
        return reinterpret_cast<MLContext>(onednn::Create());
#elif defined(WEBNN_ENABLE_BACKEND_XNNPACK)
        return reinterpret_cast<MLContext>(xnnpack::Create());
#elif defined(WEBNN_ENABLE_BACKEND_NULL)
        return reinterpret_cast<MLContext>(null::Create(options));
#else
        return nullptr;
#endif
    }

}  // namespace webnn_native
