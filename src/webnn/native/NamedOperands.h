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

#ifndef WEBNN_NATIVE_NAMED_OPERANDS_H_
#define WEBNN_NATIVE_NAMED_OPERANDS_H_

#include <map>
#include <string>

#include "common/RefCounted.h"

namespace webnn::native {

    class NamedOperandsBase : public RefCounted {
      public:
        // WebNN API
        void APISet(char const* name, const OperandBase* operand) {
            mNamedOperands[std::string(name)] = operand;
        }

        const OperandBase* Get(char const* name) const {
            if (mNamedOperands.find(std::string(name)) == mNamedOperands.end()) {
                return nullptr;
            }
            return mNamedOperands.at(std::string(name));
        }

        // Other methods
        const std::map<std::string, const OperandBase*>& GetRecords() const {
            return mNamedOperands;
        }

      private:
        std::map<std::string, const OperandBase*> mNamedOperands;
    };

}  // namespace webnn::native

#endif  // WEBNN_NATIVE_NAMED_OPERANDS_H_
