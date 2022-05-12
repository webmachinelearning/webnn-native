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

#include <gtest/gtest.h>

#include "mocks/ContextMock.h"

namespace webnn::native { namespace {

    using ::testing::Test;

    class ContextMockTests : public Test {
      protected:
        ContextMock contextMock;
    };

    TEST_F(ContextMockTests, CreateGraph) {
        EXPECT_CALL(contextMock, CreateGraphImpl).Times(1);
        EXPECT_TRUE(contextMock.CreateGraph() == nullptr);
    }

}}  // namespace webnn::native::
