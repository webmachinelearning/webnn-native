// Copyright 2018 The Dawn Authors
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

#ifndef WEBNN_NATIVE_EXPORT_H_
#define WEBNN_NATIVE_EXPORT_H_

#if defined(WEBNN_NATIVE_SHARED_LIBRARY)
#    if defined(_WIN32)
#        if defined(WEBNN_NATIVE_IMPLEMENTATION)
#            define WEBNN_NATIVE_EXPORT __declspec(dllexport)
#        else
#            define WEBNN_NATIVE_EXPORT __declspec(dllimport)
#        endif
#    else  // defined(_WIN32)
#        if defined(WEBNN_NATIVE_IMPLEMENTATION)
#            define WEBNN_NATIVE_EXPORT __attribute__((visibility("default")))
#        else
#            define WEBNN_NATIVE_EXPORT
#        endif
#    endif  // defined(_WIN32)
#else       // defined(WEBNN_NATIVE_SHARED_LIBRARY)
#    define WEBNN_NATIVE_EXPORT
#endif  // defined(WEBNN_NATIVE_SHARED_LIBRARY)

#endif  // WEBNN_NATIVE_EXPORT_H_
