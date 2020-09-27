//* Copyright 2017 The Dawn Authors
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

#include "mock_webnn.h"

using namespace testing;

namespace {
    {% for type in by_category["object"] %}
        {% for method in c_methods(type) if len(method.arguments) < 10 %}
            {{as_cType(method.return_type.name)}} Forward{{as_MethodSuffix(type.name, method.name)}}(
                {{-as_cType(type.name)}} self
                {%- for arg in method.arguments -%}
                    , {{as_annotated_cType(arg)}}
                {%- endfor -%}
            ) {
                auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
                return object->procs->{{as_MethodSuffix(type.name, method.name)}}(self
                    {%- for arg in method.arguments -%}
                        , {{as_varName(arg.name)}}
                    {%- endfor -%}
                );
            }
        {% endfor %}

    {% endfor %}
}

ProcTableAsClass::~ProcTableAsClass() {
}

void ProcTableAsClass::CompilationCompute(WebnnCompilation self,
                                WebnnNamedInputs inputs,
                                WebnnComputeCallback callback,
                                void* userdata, WebnnNamedOutputs outputs){
   auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
   object->computeCallback = callback;
   object->userdata = userdata;

   OnCompilationComputeCallback(self, inputs, callback, userdata, outputs);

}

void ProcTableAsClass::ModelCompile(WebnnModel self, WebnnCompileCallback callback,
                          void* userdata,
                          WebnnCompilationOptions const * options){
   auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
   object->compileCallback = callback;
   object->userdata = userdata;

   OnModelCompileCallback(self, callback, userdata, options);

}

bool ProcTableAsClass::NeuralNetworkContextPopErrorScope(WebnnNeuralNetworkContext neuralNetworkContext,
                                               WebnnErrorCallback callback, void * userdata){
  return OnNeuralNetworkContextPopErrorScopeCallback(neuralNetworkContext, callback, userdata);
}

void ProcTableAsClass::NeuralNetworkContextSetUncapturedErrorCallback(
		       WebnnNeuralNetworkContext neuralNetworkContext,
                       WebnnErrorCallback callback, void * userdata){
}


void ProcTableAsClass::GetProcTableAndDevice(WebnnProcTable* table) {
    // *device = GetNewDevice();

    {% for type in by_category["object"] %}
        {% for method in c_methods(type) if len(method.arguments) < 10 %}
            table->{{as_varName(type.name, method.name)}} = reinterpret_cast<{{as_cProc(type.name, method.name)}}>(Forward{{as_MethodSuffix(type.name, method.name)}});
        {% endfor %}
    {% endfor %}
}


{% for type in by_category["object"] %}
    {{as_cType(type.name)}} ProcTableAsClass::GetNew{{type.name.CamelCase()}}() {
        mObjects.emplace_back(new Object);
        mObjects.back()->procs = this;
        return reinterpret_cast<{{as_cType(type.name)}}>(mObjects.back().get());
    }
{% endfor %}

MockProcTable::MockProcTable() = default;

MockProcTable::~MockProcTable() = default;

void MockProcTable::IgnoreAllReleaseCalls() {
    {% for type in by_category["object"] %}
        EXPECT_CALL(*this, {{as_MethodSuffix(type.name, Name("release"))}}(_)).Times(AnyNumber());
    {% endfor %}
}
