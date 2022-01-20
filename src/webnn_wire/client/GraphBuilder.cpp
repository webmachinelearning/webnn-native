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

#include "webnn_wire/client/GraphBuilder.h"

#include "webnn_wire/WireCmd_autogen.h"
#include "webnn_wire/client/ApiObjects_autogen.h"
#include "webnn_wire/client/Client.h"

namespace webnn_wire { namespace client {

    MLOperand GraphBuilder::Constant(MLOperandDescriptor const* desc,
                                     MLArrayBufferView const* value) {
        GraphBuilderConstantInternalCmd cmd;
        cmd.graphBuilderId = this->id;
        cmd.desc = desc;
        cmd.buffer = static_cast<const uint8_t*>(value->buffer);
        cmd.byteLength = value->byteLength;
        cmd.byteOffset = value->byteOffset;

        // Create the Operand and send the building constant command.
        auto* allocation = client->OperandAllocator().New(client);
        Operand* operand = allocation->object.get();
        cmd.result = ObjectHandle{operand->id, allocation->generation};
        client->SerializeCommand(cmd);

        return ToAPI(operand);
    }

}}  // namespace webnn_wire::client
