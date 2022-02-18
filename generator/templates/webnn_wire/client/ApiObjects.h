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

#ifndef WEBNNNWIRE_CLIENT_APIOBJECTS_AUTOGEN_H_
#define WEBNNNWIRE_CLIENT_APIOBJECTS_AUTOGEN_H_

#include "webnn_wire/ObjectType_autogen.h"
#include "webnn_wire/client/ObjectBase.h"

namespace webnn_wire { namespace client {

    template <typename T>
    struct ObjectTypeToTypeEnum {
        static constexpr ObjectType value = static_cast<ObjectType>(-1);
    };

    {% for type in by_category["object"] %}
        {% set Type = type.name.CamelCase() %}
        {% if type.name.CamelCase() in client_special_objects %}
            class {{Type}};
        {% else %}
            struct {{type.name.CamelCase()}} final : ObjectBase {
                using ObjectBase::ObjectBase;
            };
        {% endif %}

        inline {{Type}}* FromAPI(WNN{{Type}} obj) {
            return reinterpret_cast<{{Type}}*>(obj);
        }
        inline WNN{{Type}} ToAPI({{Type}}* obj) {
            return reinterpret_cast<WNN{{Type}}>(obj);
        }

        template <>
        struct ObjectTypeToTypeEnum<{{Type}}> {
            static constexpr ObjectType value = ObjectType::{{Type}};
        };

    {% endfor %}
}}  // namespace webnn_wire::client

#endif  // WEBNNNWIRE_CLIENT_APIOBJECTS_AUTOGEN_H_
