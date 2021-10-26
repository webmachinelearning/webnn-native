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

#include "ops/Gru.h"

#include "Operand.h"
#include "Utils.h"

namespace node { namespace op {

    Napi::Value Gru::Build(const Napi::CallbackInfo& info, ml::GraphBuilder builder) {
        // Operand gru(Operand input, Operand weight, Operand recurrentWeight, int32_t steps,
        // int32_t hiddenSize, options = {})
        WEBNN_NODE_ASSERT(info.Length() == 5 || info.Length() == 6,
                          "The number of arguments is invalid.");

        std::vector<napi_value> args;
        ml::Operand input;
        WEBNN_NODE_ASSERT(GetOperand(info[0], input, args), "The input parameter is invalid.");
        ml::Operand weight;
        WEBNN_NODE_ASSERT(GetOperand(info[1], weight, args), "The weight parameter is invalid.");
        ml::Operand recurrentWeight;
        WEBNN_NODE_ASSERT(GetOperand(info[2], recurrentWeight, args),
                          "The recurrentWeight parameter is invalid.");
        int32_t steps;
        WEBNN_NODE_ASSERT(GetValue(info[3], steps), "The steps parameter is invalid.");
        int32_t hiddenSize;
        WEBNN_NODE_ASSERT(GetValue(info[4], hiddenSize), "The hiddenSize parameter is invalid.");

        // dictionary GruOptions {
        //     Operand bias;
        //     Operand recurrentBias;
        //     Operand initialHiddenState;
        //     bool resetAfter = true;
        //     bool returnSequence = false;
        //     RecurrentNetworkDirection direction = "forward";
        //     RecurrentNetworkWeightLayout layout = "zrn";
        //     sequence<MLOperator> activations;
        // };
        ml::GruOptions options;
        if (info.Length() == 6 && !info[5].IsUndefined()) {
            WEBNN_NODE_ASSERT(info[5].IsObject(), "The options must be an object.");
            Napi::Object jsOptions = info[5].As<Napi::Object>();
            if (HasOptionMember(jsOptions, "bias")) {
                WEBNN_NODE_ASSERT(GetOperand(jsOptions.Get("bias"), options.bias, args),
                                  "The bias parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "recurrentBias")) {
                WEBNN_NODE_ASSERT(
                    GetOperand(jsOptions.Get("recurrentBias"), options.recurrentBias, args),
                    "The recurrentBias parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "initialHiddenState")) {
                WEBNN_NODE_ASSERT(GetOperand(jsOptions.Get("initialHiddenState"),
                                             options.initialHiddenState, args),
                                  "The initialHiddenState parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "resetAfter")) {
                WEBNN_NODE_ASSERT(GetValue(jsOptions.Get("resetAfter"), options.resetAfter),
                                  "The resetAfter parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "returnSequence")) {
                WEBNN_NODE_ASSERT(GetValue(jsOptions.Get("returnSequence"), options.returnSequence),
                                  "The returnSequence parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "direction")) {
                WEBNN_NODE_ASSERT(
                    GetRecurrentNetworkDirection(jsOptions.Get("direction"), options.direction),
                    "The direction parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "layout")) {
                WEBNN_NODE_ASSERT(
                    GetRecurrentNetworkWeightLayout(jsOptions.Get("layout"), options.layout),
                    "The layout parameter is invalid.");
            }
            if (HasOptionMember(jsOptions, "activations")) {
                WEBNN_NODE_ASSERT(
                    GetOperatorArray(jsOptions.Get("activations"), options.activations, args),
                    "The activations parameter is invalid.");
            }
        }
        ml::OperandArray gruOutputs =
            builder.Gru(input, weight, recurrentWeight, steps, hiddenSize, &options);
        size_t len = gruOutputs.Size();
        Napi::Array objectArray = Napi::Array::New(info.Env(), len);
        for (size_t i = 0; i < len; i++) {
            Napi::Object object = Operand::constructor.New(args);
            Operand* operand = Napi::ObjectWrap<Operand>::Unwrap(object);
            operand->SetImpl(gruOutputs.Get(i));
            objectArray[i] = object;
        }

        return objectArray;
    }
}}  // namespace node::op
