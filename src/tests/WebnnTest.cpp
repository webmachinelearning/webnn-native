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

#include "tests/WebnnTest.h"

static WebnnTestEnvironment* gTestEnv = nullptr;

void InitWebnnEnd2EndTestEnvironment() {
    gTestEnv = new WebnnTestEnvironment();
    testing::AddGlobalTestEnvironment(gTestEnv);
}

static void ErrorCallback(WebnnErrorType type, char const* message, void* userdata) {
    if (type != WebnnErrorType_NoError) {
        dawn::ErrorLog() << "error type is " << type << ", messages are " << message;
    }
}

const webnn::NeuralNetworkContext& WebnnTest::GetContext() {
    return gTestEnv->GetContext();
}

void WebnnTestEnvironment::SetUp() {
    mContext = CreateCppNeuralNetworkContext();
    mContext.SetUncapturedErrorCallback(ErrorCallback, nullptr);
    DAWN_ASSERT(mContext);
}

const webnn::NeuralNetworkContext& WebnnTestEnvironment::GetContext() {
    return mContext;
}