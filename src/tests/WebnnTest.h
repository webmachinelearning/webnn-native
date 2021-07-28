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

#ifndef TESTS_WEBNN_TEST_H_
#define TESTS_WEBNN_TEST_H_

#include "examples/SampleUtils.h"
#include "gtest/gtest.h"

class WebnnTest : public testing::Test {
  protected:
    ~WebnnTest() override;

    void SetUp() override;
    void TearDown() override;

    const ml::Context& GetContext();
    void StartExpectContextError();
    bool EndExpectContextError();
    std::string GetLastErrorMessage() const;

  private:
    static void ErrorCallback(MLErrorType type, const char* message, void* userdata);
    std::string mErrorMessage;
    bool mExpectError = false;
    bool mError = false;
};

void InitWebnnEnd2EndTestEnvironment(ml::ContextOptions const* options = nullptr);

class WebnnTestEnvironment : public testing::Environment {
  public:
    explicit WebnnTestEnvironment(ml::ContextOptions const* options) : mOptions(options) {
    }
    void SetUp() override;

    const ml::Context& GetContext();

  protected:
    ml::ContextOptions const* mOptions;
    ml::Context mContext;
};

#endif  // TESTS_WEBNN_TEST_H_
