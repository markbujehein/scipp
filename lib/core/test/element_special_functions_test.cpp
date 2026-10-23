// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include "scipp/core/element/special_functions.h"
#include "scipp/units/unit.h"
#include "test_macros.h"

#include <cmath>
#include <limits>
#include <type_traits>

using namespace scipp;
using namespace scipp::core;

namespace {
void expect_rel_near(const double actual, const double expected,
                     const double rtol = 1e-13) {
  EXPECT_NEAR(actual, expected, rtol * std::abs(expected));
}
} // namespace

TEST(ElementSphericalBesselTest, jn_matches_closed_form_for_low_orders) {
  for (const double x : {0.1, 1.0, 2.5, 10.0}) {
    expect_rel_near(element::spherical_jn(int64_t{0}, x), std::sin(x) / x);
    expect_rel_near(element::spherical_jn(int64_t{1}, x),
                    std::sin(x) / (x * x) - std::cos(x) / x);
  }
}

TEST(ElementSphericalBesselTest, yn_matches_closed_form_for_low_orders) {
  for (const double x : {0.1, 1.0, 2.5, 10.0}) {
    expect_rel_near(element::spherical_yn(int64_t{0}, x), -std::cos(x) / x);
    expect_rel_near(element::spherical_yn(int64_t{1}, x),
                    -std::cos(x) / (x * x) - std::sin(x) / x);
  }
}

TEST(ElementSphericalBesselTest, high_order_matches_scipy) {
  // Reference values from scipy.special 1.18.0.
  expect_rel_near(element::spherical_jn(int64_t{100}, 1.0),
                  7.444727741661447e-190, 1e-12);
  expect_rel_near(element::spherical_jn(int64_t{50}, 1e3),
                  -0.0007793195563639975, 1e-12);
  expect_rel_near(element::spherical_yn(int64_t{100}, 1.0),
                  -6.68307946325868e+186, 1e-12);
}

TEST(ElementSphericalBesselTest, jn_at_zero) {
  EXPECT_EQ(element::spherical_jn(int64_t{0}, 0.0), 1.0);
  EXPECT_EQ(element::spherical_jn(int64_t{3}, 0.0), 0.0);
}

TEST(ElementSphericalBesselTest, yn_at_zero_is_minus_infinity) {
  EXPECT_EQ(element::spherical_yn(int64_t{0}, 0.0),
            -std::numeric_limits<double>::infinity());
  EXPECT_EQ(element::spherical_yn(int64_t{3}, 0.0),
            -std::numeric_limits<double>::infinity());
}

TEST(ElementSphericalBesselTest, yn_overflow_is_minus_infinity) {
  EXPECT_EQ(element::spherical_yn(int64_t{200}, 0.1),
            -std::numeric_limits<double>::infinity());
}

TEST(ElementSphericalBesselTest, jn_negative_x_has_parity_of_order) {
  // j_n(-x) = (-1)^n j_n(x)
  for (const int64_t n : {0, 1, 2, 3}) {
    const double sign = n % 2 == 0 ? 1.0 : -1.0;
    EXPECT_EQ(element::spherical_jn(n, -1.7),
              sign * element::spherical_jn(n, 1.7));
  }
}

TEST(ElementSphericalBesselTest, yn_negative_x_has_parity_of_order_plus_one) {
  // y_n(-x) = (-1)^(n+1) y_n(x)
  for (const int64_t n : {0, 1, 2, 3}) {
    const double sign = n % 2 == 0 ? -1.0 : 1.0;
    EXPECT_EQ(element::spherical_yn(n, -1.7),
              sign * element::spherical_yn(n, 1.7));
  }
}

TEST(ElementSphericalBesselTest, negative_order_gives_nan) {
  EXPECT_TRUE(std::isnan(element::spherical_jn(int64_t{-1}, 1.0)));
  EXPECT_TRUE(std::isnan(element::spherical_yn(int32_t{-1}, 1.0)));
}

TEST(ElementSphericalBesselTest, nan_gives_nan) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(std::isnan(element::spherical_jn(int64_t{0}, nan)));
  EXPECT_TRUE(std::isnan(element::spherical_yn(int64_t{0}, nan)));
}

TEST(ElementSphericalBesselTest, float_input_gives_float) {
  static_assert(
      std::is_same_v<decltype(element::spherical_jn(int32_t{2}, 0.5f)), float>);
  static_assert(
      std::is_same_v<decltype(element::spherical_yn(int32_t{2}, 0.5f)), float>);
  EXPECT_NEAR(element::spherical_jn(int32_t{2}, 0.5f), 0.0163711067f,
              1e-6f * 0.0163711067f);
}

TEST(ElementSphericalBesselTest, integer_input_gives_double) {
  static_assert(
      std::is_same_v<decltype(element::spherical_jn(int64_t{1}, int64_t{2})),
                     double>);
  static_assert(
      std::is_same_v<decltype(element::spherical_yn(int32_t{1}, int32_t{2})),
                     double>);
  EXPECT_EQ(element::spherical_jn(int64_t{1}, int64_t{2}),
            element::spherical_jn(int64_t{1}, 2.0));
  EXPECT_EQ(element::spherical_yn(int32_t{1}, int32_t{-2}),
            element::spherical_yn(int32_t{1}, -2.0));
}

TEST(ElementSphericalBesselTest, infinite_x_gives_zero) {
  const double inf = std::numeric_limits<double>::infinity();
  for (const int64_t n : {0, 1, 2}) {
    EXPECT_EQ(element::spherical_jn(n, inf), 0.0);
    EXPECT_EQ(element::spherical_jn(n, -inf), 0.0);
    EXPECT_EQ(element::spherical_yn(n, inf), 0.0);
    EXPECT_EQ(element::spherical_yn(n, -inf), 0.0);
  }
}

namespace {
template <class Op> void expect_dimensionless_only(const Op &op) {
  EXPECT_EQ(op(sc_units::one, sc_units::one), sc_units::one);
  EXPECT_THROW_DISCARD(op(sc_units::one, sc_units::m), except::UnitError);
  EXPECT_THROW_DISCARD(op(sc_units::m, sc_units::one), except::UnitError);
}
} // namespace

TEST(ElementSphericalBesselTest, unit) {
  expect_dimensionless_only(element::spherical_jn);
  expect_dimensionless_only(element::spherical_yn);
}
