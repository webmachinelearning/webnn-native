// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef IE_UTILS_H
#define IE_UTILS_H

#include <inference_engine.hpp>
#include <map>
#include <memory>
#include <vector>

#include "ngraph/node.hpp"

namespace InferenceEngine {

// Put this in the declarations for a class to be uncopyable.
#define DISALLOW_COPY(TypeName) TypeName(const TypeName&) = delete

// Put this in the declarations for a class to be unassignable.
#define DISALLOW_ASSIGN(TypeName) TypeName& operator=(const TypeName&) = delete

// Put this in the declarations for a class to be uncopyable and unassignable.
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  DISALLOW_COPY(TypeName);                 \
  DISALLOW_ASSIGN(TypeName)

short f32tof16(float x);

template <typename T>
void CopyDataToBuffer(T* dst, const float* src, size_t length) {
  if (std::is_same<T, float>::value || std::is_same<T, int32_t>::value) {
    memcpy(static_cast<void*>(dst), static_cast<const void*>(src), length);
  } else if (std::is_same<T, int16_t>::value) {
    size_t size = length / sizeof(float);
    for (size_t i = 0; i < size; ++i) {
      dst[i] = f32tof16(src[i]);
    }
  }
}

}  // namespace InferenceEngine

#endif  // IE_UTILS_H