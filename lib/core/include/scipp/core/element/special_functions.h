// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
#pragma once

#include <boost/math/special_functions/bessel.hpp>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

#include "scipp/common/overloaded.h"
#include "scipp/core/element/arg_list.h"
#include "scipp/core/transform_common.h"

namespace scipp::core::element {

namespace detail {
namespace bessel_policies = boost::math::policies;
// Return inf or NaN like scipy.special instead of throwing from inside a
// threaded transform. Ignoring evaluation errors means that orders far beyond
// any practical use (around 1e8) may return inaccurate values instead of
// failing. Do not promote to long double, so results are the same on all
// platforms.
using bessel_policy = bessel_policies::policy<
    bessel_policies::domain_error<bessel_policies::ignore_error>,
    bessel_policies::pole_error<bessel_policies::ignore_error>,
    bessel_policies::overflow_error<bessel_policies::ignore_error>,
    bessel_policies::evaluation_error<bessel_policies::ignore_error>,
    bessel_policies::promote_double<false>>;

/// Evaluate a spherical Bessel function of non-negative x with Boost, which
/// requires n >= 0 and x >= 0 (NaN fails Boost's assertion as well).
/// Negative x is handled by reflection, the sign flips if
/// `odd_under_reflection` is true. Integer x is evaluated as double.
template <class N, class X, class Func>
auto reflected_spherical_bessel(const N n, const X x_in,
                                const bool odd_under_reflection,
                                const Func &boost_func) {
  using T = std::conditional_t<std::is_integral_v<X>, double, X>;
  const auto x = static_cast<T>(x_in);
  if (n < 0 || std::cmp_greater(n, std::numeric_limits<unsigned>::max()) ||
      std::isnan(x))
    return std::numeric_limits<T>::quiet_NaN();
  const T value = boost_func(static_cast<unsigned>(n), std::abs(x));
  return x < 0 && odd_under_reflection ? -value : value;
}
} // namespace detail

constexpr auto spherical_bessel = overloaded{
    arg_list<std::tuple<int64_t, double>, std::tuple<int32_t, double>,
             std::tuple<int64_t, float>, std::tuple<int32_t, float>,
             std::tuple<int64_t, int64_t>, std::tuple<int32_t, int64_t>,
             std::tuple<int64_t, int32_t>, std::tuple<int32_t, int32_t>>,
    dimensionless_unit_check_return, transform_flags::expect_no_variance_arg<0>,
    transform_flags::expect_no_variance_arg<1>};

/// Spherical Bessel function of the first kind, j_n(x).
constexpr auto spherical_jn = overloaded{
    spherical_bessel, [](const auto n, const auto x) {
      // j_n(-x) = (-1)^n j_n(x)
      return detail::reflected_spherical_bessel(
          n, x, n % 2 != 0, [](const unsigned order, const auto abs_x) {
            return boost::math::sph_bessel(order, abs_x,
                                           detail::bessel_policy{});
          });
    }};

/// Spherical Bessel function of the second kind, y_n(x).
constexpr auto spherical_yn = overloaded{
    spherical_bessel, [](const auto n, const auto x) {
      // y_n(-x) = (-1)^(n+1) y_n(x)
      return detail::reflected_spherical_bessel(
          n, x, n % 2 == 0, [](const unsigned order, const auto abs_x) {
            return boost::math::sph_neumann(order, abs_x,
                                            detail::bessel_policy{});
          });
    }};

} // namespace scipp::core::element
