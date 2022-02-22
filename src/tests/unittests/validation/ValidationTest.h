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

#ifndef TESTS_UNITTESTS_VALIDATIONTEST_H_
#define TESTS_UNITTESTS_VALIDATIONTEST_H_

#include "common/Log.h"
#include "gtest/gtest.h"
#include "webnn/webnn_cpp.h"
#include "webnn_native/WebnnNative.h"

#define ASSERT_CONTEXT_ERROR(statement)                          \
    StartExpectContextError();                                   \
    statement;                                                   \
    if (!EndExpectContextError()) {                              \
        FAIL() << "Expected context error in:\n " << #statement; \
    }                                                            \
    do {                                                         \
    } while (0)

class ValidationTest : public testing::Test {
  public:
    ~ValidationTest() override;

    void SetUp() override;
    void TearDown() override;

    void StartExpectContextError();
    bool EndExpectContextError();
    std::string GetLastErrorMessage() const;

  protected:
    std::unique_ptr<webnn_native::Instance> instance;
    wnn::Context mContext;
    wnn::GraphBuilder mBuilder;

  private:
    static void ErrorCallback(WNNErrorType type, const char* message, void* userdata);
    std::string mErrorMessage;
    bool mExpectError = false;
    bool mError = false;
};

#endif  // TESTS_UNITTESTS_VALIDATIONTEST_H_
