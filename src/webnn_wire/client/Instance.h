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

#ifndef WEBNN_WIRE_CLIENT_INSTANCE_H_
#define WEBNN_WIRE_CLIENT_INSTANCE_H_

#include <webnn/webnn.h>

#include "webnn_wire/WireClient.h"
#include "webnn_wire/client/ObjectBase.h"

#include <map>

namespace webnn_wire { namespace client {

    class Instance final : public ObjectBase {
      public:
        using ObjectBase::ObjectBase;

        WNNContext CreateContextWithGpuDevice(WNNGpuDevice const* value);
    };

}}  // namespace webnn_wire::client

#endif  // WEBNN_WIRE_CLIENT_INSTANCE_H_
