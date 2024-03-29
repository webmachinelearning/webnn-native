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

#ifndef DAWNWIRE_CLIENT_CLIENTBASE_AUTOGEN_H_
#define DAWNWIRE_CLIENT_CLIENTBASE_AUTOGEN_H_

#include "webnn/wire/ChunkedCommandHandler.h"
#include "webnn/wire/WireCmd_autogen.h"
#include "webnn/wire/client/ApiObjects.h"
#include "webnn/wire/client/ObjectAllocator.h"

namespace webnn::wire::client {

    class ClientBase : public ChunkedCommandHandler, public ObjectIdProvider {
      public:
        ClientBase() = default;
        virtual ~ClientBase() = default;

        {% for type in by_category["object"] %}
            const ObjectAllocator<{{type.name.CamelCase()}}>& {{type.name.CamelCase()}}Allocator() const {
                return m{{type.name.CamelCase()}}Allocator;
            }
            ObjectAllocator<{{type.name.CamelCase()}}>& {{type.name.CamelCase()}}Allocator() {
                return m{{type.name.CamelCase()}}Allocator;
            }
        {% endfor %}

        void FreeObject(ObjectType objectType, ObjectBase* obj) {
            switch (objectType) {
                {% for type in by_category["object"] %}
                    case ObjectType::{{type.name.CamelCase()}}:
                        m{{type.name.CamelCase()}}Allocator.Free(static_cast<{{type.name.CamelCase()}}*>(obj));
                        break;
                {% endfor %}
            }
        }

      private:
        // Implementation of the ObjectIdProvider interface
        {% for type in by_category["object"] %}
            ObjectId GetId({{as_cType(type.name)}} object) const final {
                return object == nullptr ? 0 : reinterpret_cast<{{as_wireType(type)}}>(object)->id;
            }
            ObjectId GetOptionalId({{as_cType(type.name)}} object) const final {
                if (object == nullptr) {
                    return 0;
                }
                return GetId(object);
            }
        {% endfor %}

        {% for type in by_category["object"] %}
            ObjectAllocator<{{type.name.CamelCase()}}> m{{type.name.CamelCase()}}Allocator;
        {% endfor %}
    };

}  // namespace webnn::wire::client

#endif  // DAWNWIRE_CLIENT_CLIENTBASE_AUTOGEN_H_
