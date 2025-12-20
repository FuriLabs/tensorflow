/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_TSL_PLATFORM_FLOAT8_H_
#define TENSORFLOW_TSL_PLATFORM_FLOAT8_H_

#include <cstdint>
#include <cmath>
#include <limits>
#include <type_traits>

namespace tsl {

namespace detail {

enum class Float8Format : uint8_t {
  kE4M3FN,
  kE4M3FNUZ,
  kE4M3B11FNUZ,
  kE5M2,
  kE5M2FNUZ
};

inline float ClampAndQuantize(float x, Float8Format fmt) {
  if (std::isnan(x)) return std::numeric_limits<float>::quiet_NaN();
  if (std::isinf(x)) return std::copysign(std::numeric_limits<float>::infinity(), x);

  float maxv = 0.0f;
  int mant_bits = 0;
  switch (fmt) {
    case Float8Format::kE4M3FN:
    case Float8Format::kE4M3FNUZ:
    case Float8Format::kE4M3B11FNUZ:
      maxv = 240.0f;
      mant_bits = 3;
      break;
    case Float8Format::kE5M2:
    case Float8Format::kE5M2FNUZ:
      maxv = 57344.0f;
      mant_bits = 2;
      break;
  }

  float y = std::copysign(std::fmin(std::fabs(x), maxv), x);
  if (y == 0.0f) return y;

  int exp2 = 0;
  float m = std::frexp(y, &exp2);
  const float scale = std::ldexp(1.0f, mant_bits);
  m = std::round(m * scale) / scale;
  y = std::ldexp(m, exp2);

  if ((fmt == Float8Format::kE4M3FNUZ || fmt == Float8Format::kE4M3B11FNUZ || fmt == Float8Format::kE5M2FNUZ) &&
      y == 0.0f) {
    return 0.0f;
  }

  return y;
}

template <Float8Format kFmt>
struct Float8 {
  using Storage = uint8_t;

  Storage bits;

  constexpr Float8() : bits(0) {}
  explicit Float8(float x) { *this = FromFloat(x); }
  explicit Float8(double x) { *this = FromFloat(static_cast<float>(x)); }
  explicit Float8(int x) { *this = FromFloat(static_cast<float>(x)); }

  static Float8 FromFloat(float x) {
    Float8 out;

    float q = ClampAndQuantize(x, kFmt);
    if (std::isnan(q)) {
      out.bits = 0xFF;
      return out;
    }
    if (std::isinf(q)) {
      out.bits = (q < 0) ? 0xFE : 0xFD;
      return out;
    }

    float s = 0.0f;
    switch (kFmt) {
      case Float8Format::kE4M3FN:
      case Float8Format::kE4M3FNUZ:
      case Float8Format::kE4M3B11FNUZ:
        s = 16.0f;
        break;
      case Float8Format::kE5M2:
      case Float8Format::kE5M2FNUZ:
        s = 128.0f;
        break;
    }

    int v = static_cast<int>(std::lrint(q * s));
    if (v > 127) v = 127;
    if (v < -127) v = -127;

    out.bits = static_cast<Storage>(v + 128);
    return out;
  }

  float ToFloat() const {
    if (bits == 0xFF) return std::numeric_limits<float>::quiet_NaN();
    if (bits == 0xFD) return std::numeric_limits<float>::infinity();
    if (bits == 0xFE) return -std::numeric_limits<float>::infinity();

    int v = static_cast<int>(bits) - 128;  // roughly [-128..127]
    float s = 0.0f;
    switch (kFmt) {
      case Float8Format::kE4M3FN:
      case Float8Format::kE4M3FNUZ:
      case Float8Format::kE4M3B11FNUZ:
        s = 16.0f;
        break;
      case Float8Format::kE5M2:
      case Float8Format::kE5M2FNUZ:
        s = 128.0f;
        break;
    }
    return static_cast<float>(v) / s;
  }

  operator float() const { return ToFloat(); }

  friend bool operator==(Float8 a, Float8 b) { return a.bits == b.bits; }
  friend bool operator!=(Float8 a, Float8 b) { return a.bits != b.bits; }

  friend bool operator<(Float8 a, Float8 b) { return float(a) < float(b); }
  friend bool operator<=(Float8 a, Float8 b) { return float(a) <= float(b); }
  friend bool operator>(Float8 a, Float8 b) { return float(a) > float(b); }
  friend bool operator>=(Float8 a, Float8 b) { return float(a) >= float(b); }
};

static_assert(sizeof(Float8<Float8Format::kE4M3FN>) == 1, "Float8 must be 1 byte");

}  // namespace detail

using float8_e4m3fn = detail::Float8<detail::Float8Format::kE4M3FN>;
using float8_e4m3fnuz = detail::Float8<detail::Float8Format::kE4M3FNUZ>;
using float8_e4m3b11fnuz = detail::Float8<detail::Float8Format::kE4M3B11FNUZ>;
// Deprecated: old name for backward-compatibility only.
using float8_e4m3b11 = float8_e4m3b11fnuz;
using float8_e5m2 = detail::Float8<detail::Float8Format::kE5M2>;
using float8_e5m2fnuz = detail::Float8<detail::Float8Format::kE5M2FNUZ>;

}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_FLOAT8_H_
