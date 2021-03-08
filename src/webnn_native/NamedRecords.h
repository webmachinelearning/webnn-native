// Copyright 2021 The WebNN-native Authors
//
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

#ifndef WEBNN_NATIVE_NAMED_RECORDS_H_
#define WEBNN_NATIVE_NAMED_RECORDS_H_

#include <map>
#include <string>

#include "common/RefCounted.h"

namespace webnn_native {

    template <typename T>
    class NamedRecords : public RefCounted {
      public:
        NamedRecords() = default;
        virtual ~NamedRecords() = default;

        // WebNN API
        void Set(char const* name, const T* record) {
            mRecords[std::string(name)] = record;
        }

        T* Get(char const* name) const {
            if (mRecords.find(std::string(name)) == mRecords.end()) {
                return nullptr;
            }
            return const_cast<T*>(mRecords.at(std::string(name)));
        }

        // Other methods
        const std::map<std::string, const T*>& GetRecords() const {
            return mRecords;
        }

      private:
        std::map<std::string, const T*> mRecords;
    };
}  // namespace webnn_native

#endif  // WEBNN_NATIVE_NAMED_RECORD_H_
