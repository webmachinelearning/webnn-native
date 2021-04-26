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

#ifndef WEBNN_WEBNN_PROC_TABLE_H_
#define WEBNN_WEBNN_PROC_TABLE_H_

#include "webnn/webnn.h"

typedef struct WebnnProcTable {
    WebnnProcCreateGraphBuilder createGraphBuilder;
    WebnnProcCreateNamedInputs createNamedInputs;
    WebnnProcCreateNamedOperands createNamedOperands;
    WebnnProcCreateNamedOutputs createNamedOutputs;

    {% for type in by_category["object"] %}
        {% for method in c_methods(type) %}
            {{as_cProc(type.name, method.name)}} {{as_varName(type.name, method.name)}};
        {% endfor %}

    {% endfor %}
} WebnnProcTable;

#endif  // WEBNN_WEBNN_PROC_TABLE_H_
