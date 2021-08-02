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

#include "ops/Gemm.h"

#include "Operand.h"
#include "Utils.h"

namespace node { namespace op {

    Napi::Value Gemm::Build(const Napi::CallbackInfo& info, ml::GraphBuilder builder) {
        // Operand gemm(Operand a, Operand b, optional GemmOptions options = {});
        WEBNN_NODE_ASSERT(info.Length() == 2 || info.Length() == 3,
                          "The number of arguments is invalid.");

        std::vector<napi_value> args;
        ml::Operand a;
        WEBNN_NODE_ASSERT(GetOperand(info[0], a, args), "The a parameter is invalid.");
        ml::Operand b;
        WEBNN_NODE_ASSERT(GetOperand(info[1], b, args), "The b parameter is invalid.");

        // dictionary GemmOptions {
        //   Operand c;
        //   float alpha = 1.0;
        //   float beta = 1.0;
        //   boolean aTranspose = false;
        //   boolean bTranspose = false;
        // };
        ml::GemmOptions options;
        if (info.Length() == 3 && !info[2].IsUndefined()) {
            WEBNN_NODE_ASSERT(info[2].IsObject(), "The options must be an object.");
            Napi::Object jsOptions = info[2].As<Napi::Object>();
            if (HasOptionMember(jsOptions, "c")) {
                WEBNN_NODE_ASSERT(GetOperand(jsOptions.Get("c"), options.c, args),
                                  "The c parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "alpha")) {
                WEBNN_NODE_ASSERT(GetValue(jsOptions.Get("alpha"), options.alpha),
                                  "The alpha parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "beta")) {
                WEBNN_NODE_ASSERT(GetValue(jsOptions.Get("beta"), options.beta),
                                  "The beta parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "aTranspose")) {
                WEBNN_NODE_ASSERT(GetValue(jsOptions.Get("aTranspose"), options.aTranspose),
                                  "The aTranspose parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "bTranspose")) {
                WEBNN_NODE_ASSERT(GetValue(jsOptions.Get("bTranspose"), options.bTranspose),
                                  "The bTranspose parameter is invalid.");
            }
        }
        Napi::Object object = Operand::constructor.New(args);
        Operand* operand = Napi::ObjectWrap<Operand>::Unwrap(object);
        operand->SetImpl(builder.Gemm(a, b, &options));
        return object;
    }
}}  // namespace node::op
