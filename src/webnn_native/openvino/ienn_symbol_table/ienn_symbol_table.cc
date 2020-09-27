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

#include "ienn_symbol_table.h"

#include <memory>

namespace webnn_native {

// The ie_nn_c_api symbols.
#if defined(__linux__)
LATE_BINDING_SYMBOL_TABLE_DEFINE_BEGIN(IESymbolTable, "libie_nn_c_api.so")
#elif defined(_WIN32) || defined(_WIN64)
LATE_BINDING_SYMBOL_TABLE_DEFINE_BEGIN(IESymbolTable, "ie_nn_c_api.dll")
#endif
#define X(sym) LATE_BINDING_SYMBOL_TABLE_DEFINE_ENTRY(IESymbolTable, sym)
IE_SYMBOLS_LIST
#undef X
LATE_BINDING_SYMBOL_TABLE_DEFINE_END(IESymbolTable)

IESymbolTable* GetIESymbolTable() {
  static std::unique_ptr<IESymbolTable> ienn_symbol_table =
      std::make_unique<IESymbolTable>();
  return ienn_symbol_table.get();
}

}  // namespace webnn_native
