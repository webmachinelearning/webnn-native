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

#include "webnn_native/webnn_platform.h"
#include "webnn_native/WebnnNative.h"

#include <algorithm>
#include <vector>

{% for type in by_category["object"] %}
    {% if type.name.canonical_case() not in ["texture view"] %}
        #include "webnn_native/{{type.name.CamelCase()}}.h"
    {% endif %}
{% endfor %}

namespace webnn_native {

    namespace {

        {% for type in by_category["object"] %}
            {% for method in c_methods(type) %}
                {% set suffix = as_MethodSuffix(type.name, method.name) %}

                {{as_cType(method.return_type.name)}} Native{{suffix}}(
                    {{-as_cType(type.name)}} cSelf
                    {%- for arg in method.arguments -%}
                        , {{as_annotated_cType(arg)}}
                    {%- endfor -%}
                ) {
                    //* Perform conversion between C types and frontend types
                    auto self = reinterpret_cast<{{as_frontendType(type)}}>(cSelf);

                    {% for arg in method.arguments %}
                        {% set varName = as_varName(arg.name) %}
                        {% if arg.type.category in ["enum", "bitmask"] %}
                            auto {{varName}}_ = static_cast<{{as_frontendType(arg.type)}}>({{varName}});
                        {% elif arg.annotation != "value" or arg.type.category == "object" %}
                            auto {{varName}}_ = reinterpret_cast<{{decorate("", as_frontendType(arg.type), arg)}}>({{varName}});
                        {% else %}
                            auto {{varName}}_ = {{as_varName(arg.name)}};
                        {% endif %}
                    {%- endfor-%}

                    {% if method.return_type.name.canonical_case() != "void" %}
                        auto result =
                    {%- endif %}
                    self->{{method.name.CamelCase()}}(
                        {%- for arg in method.arguments -%}
                            {%- if not loop.first %}, {% endif -%}
                            {{as_varName(arg.name)}}_
                        {%- endfor -%}
                    );
                    {% if method.return_type.name.canonical_case() != "void" %}
                        {% if method.return_type.category == "object" %}
                            return reinterpret_cast<{{as_cType(method.return_type.name)}}>(result);
                        {% else %}
                            return result;
                        {% endif %}
                    {% endif %}
                }
            {% endfor %}
        {% endfor %}

        struct ProcEntry {
            WebnnProc proc;
            const char* name;
        };
        static const ProcEntry sProcMap[] = {
            {% for (type, method) in c_methods_sorted_by_name %}
                { reinterpret_cast<WebnnProc>(Native{{as_MethodSuffix(type.name, method.name)}}), "{{as_cMethod(type.name, method.name)}}" },
            {% endfor %}
        };
        static constexpr size_t sProcMapSize = sizeof(sProcMap) / sizeof(sProcMap[0]);
    }

    std::vector<const char*> GetProcMapNamesForTestingInternal() {
        std::vector<const char*> result;
        result.reserve(sProcMapSize);
        for (const ProcEntry& entry : sProcMap) {
            result.push_back(entry.name);
        }
        return result;
    }

    WNNGraphBuilder NativeCreateGraphBuilder(WNNContext context) {
        return reinterpret_cast<WNNGraphBuilder>(new GraphBuilderBase(reinterpret_cast<ContextBase *>(context)));
    }

    WNNNamedInputs NativeCreateNamedInputs() {
        return reinterpret_cast<WNNNamedInputs>(new NamedInputsBase());
    }

    WNNNamedOperands NativeCreateNamedOperands() {
         return reinterpret_cast<WNNNamedOperands>(new NamedOperandsBase());
    }

    WNNNamedOutputs NativeCreateNamedOutputs() {
         return reinterpret_cast<WNNNamedOutputs>(new NamedOutputsBase());
    }

    WNNOperatorArray NativeCreateOperatorArray() {
         return reinterpret_cast<WNNOperatorArray>(new OperatorArrayBase());
    }

    static WebnnProcTable gProcTable = {
        NativeCreateGraphBuilder,
        NativeCreateNamedInputs,
        NativeCreateNamedOperands,
        NativeCreateNamedOutputs,
        NativeCreateOperatorArray,
        {% for type in by_category["object"] %}
            {% for method in c_methods(type) %}
                Native{{as_MethodSuffix(type.name, method.name)}},
            {% endfor %}
        {% endfor %}
    };

    const WebnnProcTable& GetProcsAutogen() {
        return gProcTable;
    }

}
