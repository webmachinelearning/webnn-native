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

#ifndef WEBNN_NATIVE_IE_ERROR_IE_H_
#define WEBNN_NATIVE_IE_ERROR_IE_H_

#include <ngraph_c_api.h>

#include "webnn_native/Error.h"

namespace webnn_native { namespace ie {

    MaybeError CheckStatusCodeImpl(IEStatusCode code, const char* context);

#define CheckStatusCode(code, context) CheckStatusCodeImpl(code, context)

}}  // namespace webnn_native::ie

#endif  // WEBNN_NATIVE_IE_ERROR_IE_H_
