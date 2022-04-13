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

namespace webnn_wire::client {

    WNNOperand GraphBuilder::Constant(WNNOperandDescriptor const* desc,
                                      WNNArrayBufferView const* value) {
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

    WNNOperand GraphBuilder::ConstantWithGpuBuffer(WNNOperandDescriptor const* desc,
                                                   WNNGpuBufferView const* value) {
        GraphBuilderConstantWithGpuBufferInternalCmd cmd;
        cmd.graphBuilderId = this->id;
        cmd.desc = desc;
        cmd.buffer = static_cast<const uint8_t*>(value->buffer);
        cmd.id = value->id;
        cmd.generation = value->generation;
        cmd.byteLength = value->size;
        cmd.byteOffset = value->offset;

        // Create the Operand and send the building constant command.
        auto* allocation = client->OperandAllocator().New(client);
        Operand* operand = allocation->object.get();
        cmd.result = ObjectHandle{operand->id, allocation->generation};
        client->SerializeCommand(cmd);

        return ToAPI(operand);
    }

    // Override GraphBuilderGruCmd to set the size of result OperandArray in client,
    // otherwise WNNOperandArray.Size() need to wait Server return a command with the size.
    WNNOperandArray GraphBuilder::Gru(WNNOperand input,
                                      WNNOperand weight,
                                      WNNOperand recurrentWeight,
                                      int32_t steps,
                                      int32_t hiddenSize,
                                      WNNGruOptions const* options) {
        GraphBuilderGruInternalCmd cmd;
        cmd.graphBuilderId = this->id;

        auto* allocation = client->OperandArrayAllocator().New(client);
        OperandArray* operandArray = allocation->object.get();
        operandArray->SetSize(options->returnSequence ? 2 : 1);

        cmd.result = ObjectHandle{operandArray->id, allocation->generation};
        cmd.inputId = FromAPI(input)->id;
        cmd.weightId = FromAPI(weight)->id;
        cmd.recurrentWeightId = FromAPI(recurrentWeight)->id;
        cmd.steps = steps;
        cmd.hiddenSize = hiddenSize;
        cmd.options = options;

        client->SerializeCommand(cmd);

        return ToAPI(operandArray);
    }

    // Override GraphBuilderSplitCmd to set the size of result OperandArray in client,
    // otherwise WNNOperandArray.Size() need to wait Server return a command with the size.
    WNNOperandArray GraphBuilder::Split(WNNOperand input,
                                        uint32_t const* splits,
                                        uint32_t splitsCount,
                                        WNNSplitOptions const* options) {
        GraphBuilderSplitInternalCmd cmd;
        cmd.graphBuilderId = this->id;

        auto* allocation = client->OperandArrayAllocator().New(client);
        OperandArray* operandArray = allocation->object.get();
        operandArray->SetSize(splitsCount == 1 ? splits[0] : splitsCount);

        cmd.result = ObjectHandle{operandArray->id, allocation->generation};
        cmd.inputId = FromAPI(input)->id;
        cmd.splits = splits;
        cmd.splitsCount = splitsCount;
        cmd.options = options;

        client->SerializeCommand(cmd);

        return ToAPI(operandArray);
    }

}  // namespace webnn_wire::client
