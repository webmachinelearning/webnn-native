//* Copyright 2017 The Dawn Authors
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

#ifndef WEBNN_NATIVE_WEBNN_STRUCTS_H_
#define WEBNN_NATIVE_WEBNN_STRUCTS_H_

#include "webnn/webnn_cpp.h"
#include "webnn_native/Forward.h"

namespace webnn_native {

{% macro render_cpp_default_value(member) -%}
    {%- if member.annotation in ["*", "const*", "const*const*"] and member.optional -%}
        {{" "}}= nullptr
    {%- elif member.type.category == "object" and member.optional -%}
        {{" "}}= nullptr
    {%- elif member.type.category in ["enum", "bitmask"] and member.default_value != None -%}
        {{" "}}= ml::{{as_cppType(member.type.name)}}::{{as_cppEnum(Name(member.default_value))}}
    {%- elif member.type.category == "native" and member.default_value != None -%}
        {{" "}}= {{member.default_value}}
    {%- else -%}
        {{assert(member.default_value == None)}}
    {%- endif -%}
{%- endmacro %}

    struct ChainedStruct {
        ChainedStruct const * nextInChain = nullptr;
    };

    {% for type in by_category["structure"] %}
        {% if type.chained %}
            struct {{as_cppType(type.name)}} : ChainedStruct {
                {{as_cppType(type.name)}}() {
                    sType = ml::SType::{{type.name.CamelCase()}};
                }
        {% else %}
            struct {{as_cppType(type.name)}} {
        {% endif %}
            {% if type.extensible %}
                ChainedStruct const * nextInChain = nullptr;
            {% endif %}
            {% for member in type.members %}
                {% set member_declaration = as_annotated_frontendType(member) + render_cpp_default_value(member) %}
                {% if type.chained and loop.first %}
                    //* Align the first member to ChainedStruct to match the C struct layout.
                    alignas(ChainedStruct) {{member_declaration}};
                {% else %}
                    {{member_declaration}};
                {% endif %}
            {% endfor %}
        };

    {% endfor %}

} // namespace webnn_native

#endif  // WEBNN_NATIVE_WEBNN_STRUCTS_H_
