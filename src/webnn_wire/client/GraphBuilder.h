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

#ifndef WEBNN_WIRE_CLIENT_GRAPH_BUILDER_H_
#define WEBNN_WIRE_CLIENT_GRAPH_BUILDER_H_

#include <webnn/webnn.h>

#include "webnn_wire/WireClient.h"
#include "webnn_wire/client/ObjectBase.h"

#include <map>

namespace webnn_wire::client {

    class GraphBuilder final : public ObjectBase {
      public:
        using ObjectBase::ObjectBase;

        WNNOperand Constant(WNNOperandDescriptor const* desc, WNNArrayBufferView const* value);
        WNNOperand ConstantWithGpuBuffer(WNNOperandDescriptor const* desc,
                                         WNNGpuBufferView const* value);
        WNNOperandArray Gru(WNNOperand input,
                            WNNOperand weight,
                            WNNOperand recurrentWeight,
                            int32_t steps,
                            int32_t hiddenSize,
                            WNNGruOptions const* options);
        WNNOperandArray Split(WNNOperand input,
                              uint32_t const* splits,
                              uint32_t splitsCount,
                              WNNSplitOptions const* options);
    };

}  // namespace webnn_wire::client

#endif  // WEBNN_WIRE_CLIENT_GRAPH_BUILDER_H_
