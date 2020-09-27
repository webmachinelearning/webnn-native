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

#ifndef WEBNN_NATIVE_DML_COMPILATION_DML_H_
#define WEBNN_NATIVE_DML_COMPILATION_DML_H_

#include "webnn_native/Compilation.h"
#include "webnn_native/dml/ModelDML.h"

namespace pydml {
    struct CompiledModel;
}

namespace webnn_native { namespace dml {

    class Compilation : public CompilationBase {
      public:
        explicit Compilation(const Ref<Model>& model);
        ~Compilation() override = default;

        IDMLCompiledOperator* GetCompiledOperator() {
            return mCompiledModel->op.Get();
        }

      private:
        void ComputeImpl(NamedInputsBase* inputs,
                         WebnnComputeCallback callback,
                         void* userdata,
                         NamedOutputsBase* outputs = nullptr) override;

        Ref<Model> mModel;
        std::unique_ptr<pydml::CompiledModel> mCompiledModel;
    };

}}  // namespace webnn_native::dml

#endif  // WEBNN_NATIVE_DML_COMPILATION_DML_H_
