#include "tests/cpu_model.h"
#include "tests/test_framework.h"

#include <cmath>
#include <iostream>
#include <vector>

using namespace dax;

static double sum_host(const std::vector<float>& v) {
  long double s = 0;
  for (float x : v)
    s += static_cast<long double>(x);
  return static_cast<double>(s);
}

int main() {
  try {
    GridSpec g{32, 32, 16, 16, 1e-3f};
    ModelParams mp{};
    RuntimeSpec rt{};

    BuffersHost b{};
    alloc_fields(g, b);
    const size_t N = total_size(g);

    for (int s = 0; s < 50; ++s)
      step_sim_cpu(s, g, mp, rt, b);

    const auto& rho = rho_current_vec(b);
    const auto& Rre = Rre_current_vec(b);

    ASSERT_TRUE(!testfw::has_nan_inf(rho));
    ASSERT_TRUE(!testfw::has_nan_inf(Rre));
    ASSERT_TRUE(!testfw::has_nan_inf(b.I));

    auto max_abs = [](const std::vector<float>& v) {
      double m = 0;
      for (float x : v)
        m = std::max(m, std::abs(static_cast<double>(x)));
      return m;
    };

    ASSERT_LT(max_abs(rho), 1e-6);
    ASSERT_LT(max_abs(Rre), 1e-6);

    std::vector<float> rho_seed(N, 0.0f);
    const int cx = g.Nx / 2, cy = g.Ny / 2, cz = g.Nz / 2, cu = g.Nu / 2;
    rho_seed[idx(cx, cy, cz, cu, g.Nx, g.Ny, g.Nz, g.Nu)] = 1.0f;
    b.rho0 = rho_seed;
    b.rho1 = rho_seed;
    b.ping = 0;

    for (int s = 0; s < 30; ++s)
      step_sim_cpu(s, g, mp, rt, b);

    const auto& rho2 = rho_current_vec(b);
    const double center = rho2[idx(cx, cy, cz, cu, g.Nx, g.Ny, g.Nz, g.Nu)];
    ASSERT_LT(center, 1.0);

    const double s0 = sum_host(rho_seed);
    const double s1 = sum_host(rho2);
    ASSERT_TRUE(std::isfinite(s1));
    ASSERT_LT(std::abs(s1), 1e9);
    ASSERT_LT(std::abs(s1 - s0), 1e-2);

    std::cout << "✓ Tier 1 numeric tests PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Tier 1 numeric tests FAILED: " << e.what() << "\n";
    return 1;
  }
}
