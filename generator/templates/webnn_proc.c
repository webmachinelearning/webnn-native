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

#include "webnn/webnn_proc.h"

static WebnnProcTable procs;

static WebnnProcTable nullProcs;

void webnnProcSetProcs(const WebnnProcTable* procs_) {
    if (procs_) {
        procs = *procs_;
    } else {
        procs = nullProcs;
    }
}

MLGraphBuilder webnnCreateGraphBuilder(MLContext context) {
    return procs.createGraphBuilder(context);
}

MLNamedInputs webnnCreateNamedInputs() {
    return procs.createNamedInputs();
}

MLNamedOperands webnnCreateNamedOperands() {
    return procs.createNamedOperands();
}

MLNamedOutputs webnnCreateNamedOutputs() {
    return procs.createNamedOutputs();
}

{% for type in by_category["object"] %}
    {% for method in c_methods(type) %}
        {{as_cType(method.return_type.name)}} {{as_cMethod(type.name, method.name)}}(
            {{-as_cType(type.name)}} {{as_varName(type.name)}}
            {%- for arg in method.arguments -%}
                , {{as_annotated_cType(arg)}}
            {%- endfor -%}
        ) {
            {% if method.return_type.name.canonical_case() != "void" %}return {% endif %}
            procs.{{as_varName(type.name, method.name)}}({{as_varName(type.name)}}
                {%- for arg in method.arguments -%}
                    , {{as_varName(arg.name)}}
                {%- endfor -%}
            );
        }
    {% endfor %}

{% endfor %}
