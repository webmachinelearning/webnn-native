// Copyright 2017 The Dawn Authors
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

#ifndef WEBNN_NATIVE_FORWARD_H_
#define WEBNN_NATIVE_FORWARD_H_

#include <cstdint>

template <typename T>
class Ref;

namespace webnn::native {

    class ContextBase;
    class GraphBase;
    class GraphBuilderBase;
    class InstanceBase;
    class NamedInputsBase;
    class NamedOperandsBase;
    class NamedOutputsBase;
    class NamedResultsBase;
    class OperandArrayBase;
    class OperatorArrayBase;
    class OperandBase;
    class OperatorBase;
    class FusionOperatorBase;
    class ResultBase;

}  // namespace webnn::native

#endif  // WEBNN_NATIVE_FORWARD_H_
