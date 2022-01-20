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

#include "webnn_wire/client/OperandArray.h"

#include "webnn_wire/WireCmd_autogen.h"
#include "webnn_wire/client/Client.h"

namespace webnn_wire { namespace client {

    size_t OperandArray::Size() {
        OperandArraySizeCmd cmd;
        cmd.operandArrayId = this->id;

        client->SerializeCommand(cmd);

        // TODO: Implement return command to get the size from wire server.
        DAWN_ASSERT(0);
        return 0;
    }

}}  // namespace webnn_wire::client
