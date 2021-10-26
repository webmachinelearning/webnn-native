//* Copyright 2020 The Dawn Authors
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
//*
//*
//* This template itself is part of the Dawn source and follows Dawn's license
//* but the generated file is used for "WebGPU native". The template comments
//* using //* at the top of the file are removed during generation such that
//* the resulting file starts with the BSD 3-Clause comment.
//*
//*
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

#ifndef WEBNN_H_
#define WEBNN_H_

#if defined(WEBNN_SHARED_LIBRARY)
#    if defined(_WIN32)
#        if defined(WEBNN_IMPLEMENTATION)
#            define WEBNN_EXPORT __declspec(dllexport)
#        else
#            define WEBNN_EXPORT __declspec(dllimport)
#        endif
#    else  // defined(_WIN32)
#        if defined(WEBNN_IMPLEMENTATION)
#            define WEBNN_EXPORT __attribute__((visibility("default")))
#        else
#            define WEBNN_EXPORT
#        endif
#    endif  // defined(_WIN32)
#else       // defined(WEBNN_SHARED_LIBRARY)
#    define WEBNN_EXPORT
#endif  // defined(WEBNN_SHARED_LIBRARY)

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef uint32_t WEBNNFlags;

{% for type in by_category["object"] %}
    typedef struct {{as_cType(type.name)}}Impl* {{as_cType(type.name)}};
{% endfor %}

{% for type in by_category["enum"] + by_category["bitmask"] %}
    typedef enum {{as_cType(type.name)}} {
        {% for value in type.values %}
            {{as_cEnum(type.name, value.name)}} = 0x{{format(value.value, "08X")}},
        {% endfor %}
        {{as_cEnum(type.name, Name("force32"))}} = 0x7FFFFFFF
    } {{as_cType(type.name)}};
    {% if type.category == "bitmask" %}
        typedef WebnnFlags {{as_cType(type.name)}}Flags;
    {% endif %}

{% endfor %}

{% for type in by_category["structure"] %}
    typedef struct {{as_cType(type.name)}} {
        {% for member in type.members %}
            {{as_annotated_cType(member)}};
        {% endfor %}
    } {{as_cType(type.name)}};

{% endfor %}

#ifdef __cplusplus
extern "C" {
#endif

{% for type in by_category["callback"] %}
    typedef void (*{{as_cType(type.name)}})(
        {%- for arg in type.arguments -%}
            {% if not loop.first %}, {% endif %}{{as_annotated_cType(arg)}}
        {%- endfor -%}
    );
{% endfor %}

typedef void (*WebnnProc)(void);

#if !defined(WEBNN_SKIP_PROCS)

typedef MLGraphBuilder (*WebnnProcCreateGraphBuilder)(MLContext context);
typedef MLNamedInputs (*WebnnProcCreateNamedInputs)();
typedef MLNamedOperands (*WebnnProcCreateNamedOperands)();
typedef MLNamedOutputs (*WebnnProcCreateNamedOutputs)();
typedef MLOperatorArray (*WebnnProcCreateOperatorArray)();

{% for type in by_category["object"] if len(c_methods(type)) > 0 %}
    // Procs of {{type.name.CamelCase()}}
    {% for method in c_methods(type) %}
        typedef {{as_cType(method.return_type.name)}} (*{{as_cProc(type.name, method.name)}})(
            {{-as_cType(type.name)}} {{as_varName(type.name)}}
            {%- for arg in method.arguments -%}
                , {{as_annotated_cType(arg)}}
            {%- endfor -%}
        );
    {% endfor %}

{% endfor %}
#endif  // !defined(WEBNN_SKIP_PROCS)

#if !defined(WEBNN_SKIP_DECLARATIONS)

WEBNN_EXPORT MLGraphBuilder webnnCreateGraphBuilder(MLContext context);
WEBNN_EXPORT MLNamedInputs webnnCreateNamedInputs();
WEBNN_EXPORT MLNamedOperands webnnCreateNamedOperands();
WEBNN_EXPORT MLNamedOutputs webnnCreateNamedOutputs();
WEBNN_EXPORT MLOperatorArray webnnCreateOperatorArray();

{% for type in by_category["object"] if len(c_methods(type)) > 0 %}
    // Methods of {{type.name.CamelCase()}}
    {% for method in c_methods(type) %}
        WEBNN_EXPORT {{as_cType(method.return_type.name)}} {{as_cMethod(type.name, method.name)}}(
            {{-as_cType(type.name)}} {{as_varName(type.name)}}
            {%- for arg in method.arguments -%}
                , {{as_annotated_cType(arg)}}
            {%- endfor -%}
        );
    {% endfor %}

{% endfor %}
#endif  // !defined(WEBNN_SKIP_DECLARATIONS)

#ifdef __cplusplus
} // extern "C"
#endif

#endif // WEBNN_H_
