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

#ifndef NODE_OPS_INSTANCENORM_H__
#define NODE_OPS_INSTANCENORM_H__

#include <napi.h>
#include <webnn/webnn_cpp.h>

namespace node { namespace op {

    struct InstanceNorm {
        static Napi::Value Build(const Napi::CallbackInfo& info, ml::GraphBuilder builder);
    };

}}  // namespace node::op

#endif  // ___OPS_INSTANCENORM_H__