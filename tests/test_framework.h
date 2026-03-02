#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace testfw {

inline void fail(const std::string& msg) { throw std::runtime_error(msg); }

#define ASSERT_TRUE(x)                                                                             \
  do {                                                                                             \
    if (!(x))                                                                                      \
      testfw::fail(std::string("ASSERT_TRUE failed: ") + #x);                                    \
  } while (0)

#define ASSERT_LT(a, b)                                                                            \
  do {                                                                                             \
    if (!((a) < (b)))                                                                              \
      testfw::fail("ASSERT_LT failed");                                                           \
  } while (0)

struct DiffStats {
  double l2_rel = 0.0;
  double max_abs = 0.0;
};

inline bool has_nan_inf(const std::vector<float>& v) {
  for (float x : v)
    if (!std::isfinite(x))
      return true;
  return false;
}

inline DiffStats compare_rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size())
    fail("compare_rel_l2: size mismatch");
  long double num = 0;
  long double den = 0;
  double mx = 0;
  for (size_t i = 0; i < a.size(); i++) {
    const double da = static_cast<double>(a[i]);
    const double db = static_cast<double>(b[i]);
    const double d = da - db;
    num += static_cast<long double>(d * d);
    den += static_cast<long double>(da * da);
    mx = std::max(mx, std::abs(d));
  }

  DiffStats s;
  s.max_abs = mx;
  s.l2_rel = (den > 0) ? std::sqrt(static_cast<double>(num) / static_cast<double>(den))
                       : std::sqrt(static_cast<double>(num));
  return s;
}

} // namespace testfw
