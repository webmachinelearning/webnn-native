//* Copyright 2019 The Dawn Authors
//* Copyright 2021 The WebNN-native Authors
//*
//* Licensed under the Apache License, Version 2.0 (the "License");
//* you may not use this file except in compliance with the License.
//* You may obtain a copy of the License at
//*
//*     http://www.apache.org/licenses/LICENSE-2.0
//*
//* Unless required by applicable law or agreed to in writing, software
//* distributed under the License is distributed on an "AS IS" BASIS,
//* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//* See the License for the specific language governing permissions and
//* limitations under the License.

#include "common/Assert.h"
#include "webnn/wire/client/Client.h"

#include <string>

namespace webnn::wire::client {
    {% for command in cmd_records["return command"] %}
        bool Client::Handle{{command.name.CamelCase()}}(const volatile char** commands, size_t* size) {
            Return{{command.name.CamelCase()}}Cmd cmd;
            DeserializeResult deserializeResult = cmd.Deserialize(commands, size, &mAllocator);

            if (deserializeResult == DeserializeResult::FatalError) {
                return false;
            }

            {% for member in command.members if member.handle_type %}
                {% set Type = member.handle_type.name.CamelCase() %}
                {% set name = as_varName(member.name) %}

                {% if member.type.dict_name == "ObjectHandle" %}
                    {{Type}}* {{name}} = {{Type}}Allocator().GetObject(cmd.{{name}}.id);
                    uint32_t {{name}}Generation = {{Type}}Allocator().GetGeneration(cmd.{{name}}.id);
                    if ({{name}}Generation != cmd.{{name}}.generation) {
                        {{name}} = nullptr;
                    }
                {% endif %}
            {% endfor %}

            return Do{{command.name.CamelCase()}}(
                {%- for member in command.members -%}
                    {%- if member.handle_type -%}
                        {{as_varName(member.name)}}
                    {%- else -%}
                        cmd.{{as_varName(member.name)}}
                    {%- endif -%}
                    {%- if not loop.last -%}, {% endif %}
                {%- endfor -%}
            );
        }
    {% endfor %}

    const volatile char* Client::HandleCommandsImpl(const volatile char* commands, size_t size) {
        while (size >= sizeof(CmdHeader) + sizeof(ReturnWireCmd)) {
            // Start by chunked command handling, if it is done, then it means the whole buffer
            // was consumed by it, so we return a pointer to the end of the commands.
            switch (HandleChunkedCommands(commands, size)) {
                case ChunkedCommandsResult::Consumed:
                    return commands + size;
                case ChunkedCommandsResult::Error:
                    return nullptr;
                case ChunkedCommandsResult::Passthrough:
                    break;
            }

            ReturnWireCmd cmdId = *reinterpret_cast<const volatile ReturnWireCmd*>(commands + sizeof(CmdHeader));
            bool success = false;
            switch (cmdId) {
                {% for command in cmd_records["return command"] %}
                    {% set Suffix = command.name.CamelCase() %}
                    case ReturnWireCmd::{{Suffix}}:
                        success = Handle{{Suffix}}(&commands, &size);
                        break;
                {% endfor %}
                default:
                    success = false;
            }

            if (!success) {
                return nullptr;
            }
            mAllocator.Reset();
        }

        if (size != 0) {
            return nullptr;
        }

        return commands;
    }
}  // namespace webnn::wire::client
