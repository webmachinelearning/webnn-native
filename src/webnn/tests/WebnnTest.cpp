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

#include "webnn/tests/WebnnTest.h"

static WebnnTestEnvironment* gTestEnv = nullptr;

void InitWebnnEnd2EndTestEnvironment(wnn::ContextOptions const* options) {
    gTestEnv = new WebnnTestEnvironment(options);
    testing::AddGlobalTestEnvironment(gTestEnv);
}

const wnn::Context& WebnnTest::GetContext() {
    return gTestEnv->GetContext();
}

void WebnnTest::SetUp() {
    const wnn::Context& context = GetContext();
    context.SetUncapturedErrorCallback(ErrorCallback, this);
}

WebnnTest::~WebnnTest() {
    const wnn::Context& context = GetContext();
    context.SetUncapturedErrorCallback(ErrorCallback, nullptr);
}

void WebnnTest::TearDown() {
    ASSERT_FALSE(mExpectError);
}

void WebnnTest::StartExpectContextError() {
    mExpectError = true;
    mError = false;
}
bool WebnnTest::EndExpectContextError() {
    mExpectError = false;
    return mError;
}

std::string WebnnTest::GetLastErrorMessage() const {
    return mErrorMessage;
}

void WebnnTest::ErrorCallback(WNNErrorType type, char const* message, void* userdata) {
    ASSERT(type != WNNErrorType_NoError);
    auto self = static_cast<WebnnTest*>(userdata);
    if (self) {
        self->mErrorMessage = message;

        ASSERT_TRUE(self->mExpectError) << "Got unexpected error: " << message;
        ASSERT_FALSE(self->mError) << "Got two errors in expect block";
        self->mError = true;
    } else {
        ASSERT_TRUE(type == WNNErrorType_NoError) << "Got unexpected error: " << message;
    }
}

void WebnnTestEnvironment::SetUp() {
    mContext = CreateCppContext(mOptions);
    DAWN_ASSERT(mContext);
}

const wnn::Context& WebnnTestEnvironment::GetContext() {
    return mContext;
}
