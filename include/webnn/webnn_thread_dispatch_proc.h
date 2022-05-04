// Copyright 2020 The Dawn Authors
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

#ifndef DAWN_WEBNN_THREAD_DISPATCH_PROC_H_
#define DAWN_WEBNN_THREAD_DISPATCH_PROC_H_

#include "webnn_proc.h"

#ifdef __cplusplus
extern "C" {
#endif

// Call webnnProcSetProcs(&webnnThreadDispatchProcTable) and then use webnnProcSetPerThreadProcs
// to set per-thread procs.
WNN_EXPORT extern WebnnProcTable webnnThreadDispatchProcTable;
WNN_EXPORT void webnnProcSetPerThreadProcs(const WebnnProcTable* procs);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // DAWN_WEBNN_THREAD_DISPATCH_PROC_H_
