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

#include "utils.h"

namespace InferenceEngine {

namespace {

float asfloat(uint32_t v) {
  union {
    float f;
    std::uint32_t u;
  } converter = {0};
  converter.u = v;
  return converter.f;
}

}  // namespace

short f32tof16(float x) {
  static float min16 = asfloat((127 - 14) << 23);

  static float max16 = asfloat(((127 + 15) << 23) | 0x007FE000);
  static uint32_t max16f16 = ((15 + 15) << 10) | 0x3FF;

  static constexpr std::uint32_t EXP_MASK_F32 = 0x7F800000U;

  union {
    float f;
    uint32_t u;
  } v = {0};
  v.f = x;

  uint32_t s = (v.u >> 16) & 0x8000;

  v.u &= 0x7FFFFFFF;

  if ((v.u & EXP_MASK_F32) == EXP_MASK_F32) {
    if (v.u & 0x007FFFFF) {
      return static_cast<short>(s | (v.u >> (23 - 10)) | 0x0200);
    } else {
      return static_cast<short>(s | (v.u >> (23 - 10)));
    }
  }

  float halfULP = asfloat(v.u & EXP_MASK_F32) * asfloat((127 - 11) << 23);
  v.f += halfULP;

  if (v.f < min16 * 0.5f) {
    return static_cast<short>(s);
  }

  if (v.f < min16) {
    return static_cast<short>(s | (1 << 10));
  }

  if (v.f >= max16) {
    return static_cast<short>(max16f16 | s);
  }

  v.u -= ((127 - 15) << 23);

  v.u >>= (23 - 10);

  return static_cast<short>(v.u | s);
}

}  // namespace InferenceEngine
