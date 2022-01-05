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

#include "tests/unittests/validation/ValidationTest.h"

#include "common/Assert.h"
#include "webnn/webnn.h"
#include "webnn/webnn_proc.h"

void ValidationTest::SetUp() {
    instance = std::make_unique<webnn_native::Instance>();
    MLContext context = instance->CreateTestContext();
    WebnnProcTable backendProcs = webnn_native::GetProcs();
    ASSERT_NE(&backendProcs, nullptr);
    webnnProcSetProcs(&backendProcs);
    // GTest will not run test body if fail to create context.
    ASSERT_TRUE(context != nullptr);
    mContext = ml::Context::Acquire(context);
    mContext.SetUncapturedErrorCallback(ErrorCallback, this);
    mBuilder = ml::CreateGraphBuilder(mContext);
}

ValidationTest::~ValidationTest() {
}

void ValidationTest::TearDown() {
    ASSERT_FALSE(mExpectError);
}

void ValidationTest::StartExpectContextError() {
    mExpectError = true;
    mError = false;
}
bool ValidationTest::EndExpectContextError() {
    mExpectError = false;
    return mError;
}

std::string ValidationTest::GetLastErrorMessage() const {
    return mErrorMessage;
}

void ValidationTest::ErrorCallback(MLErrorType type, char const* message, void* userdata) {
    ASSERT(type != MLErrorType_NoError);
    auto self = static_cast<ValidationTest*>(userdata);
    self->mErrorMessage = message;

    ASSERT_TRUE(self->mExpectError) << "Got unexpected error: " << message;
    ASSERT_FALSE(self->mError) << "Got two errors in expect block";
    self->mError = true;
}
