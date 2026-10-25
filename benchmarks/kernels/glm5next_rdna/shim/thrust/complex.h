// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// isa_check stand-in for rocThrust's thrust/complex.h, used when the ROCm
// path has no thrust/. torch's complex headers convert to and from it under
// __HIPCC__; the host code that does so is parsed, never emitted.
#pragma once
namespace thrust {
template <class T>
struct complex {
  T re{}, im{};
  complex() = default;
  __host__ __device__ complex(T r, T i = T()) : re(r), im(i) {}
  template <class U>
  __host__ __device__ complex(const complex<U>& o) : re(o.re), im(o.im) {}
  __host__ __device__ T real() const { return re; }
  __host__ __device__ T imag() const { return im; }
};
#define ISA_CHECK_THRUST_UNARY(name)                         \
  template <class T>                                         \
  __host__ __device__ complex<T> name(const complex<T>& z) { \
    return z;                                                \
  }
ISA_CHECK_THRUST_UNARY(exp)
ISA_CHECK_THRUST_UNARY(log)
ISA_CHECK_THRUST_UNARY(log10)
ISA_CHECK_THRUST_UNARY(sqrt)
ISA_CHECK_THRUST_UNARY(sin)
ISA_CHECK_THRUST_UNARY(cos)
ISA_CHECK_THRUST_UNARY(tan)
ISA_CHECK_THRUST_UNARY(asin)
ISA_CHECK_THRUST_UNARY(acos)
ISA_CHECK_THRUST_UNARY(atan)
ISA_CHECK_THRUST_UNARY(sinh)
ISA_CHECK_THRUST_UNARY(cosh)
ISA_CHECK_THRUST_UNARY(tanh)
ISA_CHECK_THRUST_UNARY(asinh)
ISA_CHECK_THRUST_UNARY(acosh)
ISA_CHECK_THRUST_UNARY(atanh)
ISA_CHECK_THRUST_UNARY(proj)
#undef ISA_CHECK_THRUST_UNARY
template <class T>
__host__ __device__ T abs(const complex<T>& z) {
  return z.re;
}
template <class T>
__host__ __device__ complex<T> polar(const T& r, const T& t = T()) {
  return complex<T>(r, t);
}
template <class T>
__host__ __device__ complex<T> pow(const complex<T>& a, const complex<T>&) {
  return a;
}
template <class T>
__host__ __device__ complex<T> pow(const complex<T>& a, const T&) {
  return a;
}
template <class T>
__host__ __device__ complex<T> pow(const T& a, const complex<T>&) {
  return complex<T>(a);
}
template <class T, class U>
__host__ __device__ complex<decltype(T() + U())> pow(const complex<T>& a,
                                                     const complex<U>&) {
  return a;
}
template <class T, class U>
__host__ __device__ complex<decltype(T() + U())> pow(const complex<T>& a,
                                                     const U&) {
  return a;
}
template <class T, class U>
__host__ __device__ complex<decltype(T() + U())> pow(const T& a,
                                                     const complex<U>&) {
  return complex<decltype(T() + U())>(a);
}
}  // namespace thrust
