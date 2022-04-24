//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once

namespace pydml
{
    struct CompiledModel
    {
        CompiledModel(
            dml::Graph& graph, 
            DML_EXECUTION_FLAGS flags,
            std::vector<dml::Expression>& outputs
            ) : 
            op(graph.Compile(flags, outputs))
        {}

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> op;
    };

    struct TensorData
    {
        TensorData(void * buffer,
                   size_t size) :
            buffer(buffer),
            size(size) {}

        TensorData(dml::TensorDesc* desc) :
            size(desc->totalTensorSizeInBytes),
            desc(*desc->AsPtr<DML_BUFFER_TENSOR_DESC>())
        {
            // Free by user code.
            buffer = malloc(size);
        }

        TensorData() {}

        void* Get() const { return buffer; }

        size_t Size() const { return size; }

        const dml::TensorDesc* Desc() const { return &desc; }

        void* buffer;
        size_t size;
        dml::TensorDesc desc;
    };

    struct Binding
    {
        explicit Binding(dml::Expression& expression, 
                         void * buffer,
                         size_t size)
            :   desc(expression.GetOutputDesc()),
                data(buffer, size)
        {}

        Binding() = default;

        dml::TensorDesc desc;
        TensorData data;
    };
}
