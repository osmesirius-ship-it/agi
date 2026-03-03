#include "tests/cpu_model.h"
#include "tests/test_framework.h"

#include <cmath>
#include <iostream>
#include <vector>

using namespace dax;

static double energy_R(const std::vector<float>& Rre, const std::vector<float>& Rim) {
  long double e = 0;
  for (size_t i = 0; i < Rre.size(); ++i) {
    const long double re = Rre[i], im = Rim[i];
    e += re * re + im * im;
  }
  return static_cast<double>(e);
}

int main() {
  try {
    GridSpec g{32, 32, 16, 16, 1e-3f};
    ModelParams mp{};
    mp.RFN_enabled = false;
    RuntimeSpec rt{};

    BuffersHost b{};
    alloc_fields(g, b);
    const size_t N = total_size(g);

    std::vector<float> Rre(N, 0.0f), Rim(N, 0.0f);
    const int cx = g.Nx / 2, cy = g.Ny / 2, cz = g.Nz / 2, cu = g.Nu / 2;
    Rre[idx(cx, cy, cz, cu, g.Nx, g.Ny, g.Nz, g.Nu)] = 1.0f;

    b.R0re = Rre; b.R1re = Rre;
    b.R0im = Rim; b.R1im = Rim;

    double e0 = 0, e_mid = 0, e_end = 0;
    for (int s = 0; s < 400; ++s) {
      step_sim_cpu(s, g, mp, rt, b);
      if (s == 0) e0 = energy_R(Rre_current_vec(b), Rim_current_vec(b));
      if (s == 200) e_mid = energy_R(Rre_current_vec(b), Rim_current_vec(b));
    }
    e_end = energy_R(Rre_current_vec(b), Rim_current_vec(b));

    ASSERT_TRUE(std::isfinite(e0));
    ASSERT_TRUE(std::isfinite(e_mid));
    ASSERT_TRUE(std::isfinite(e_end));
    ASSERT_LT(e_end, 1e6);
    ASSERT_LT(e_end, e_mid);
    ASSERT_LT(e_mid, e0);

    std::cout << "✓ Tier 2 physics tests PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Tier 2 physics tests FAILED: " << e.what() << "\n";
    return 1;
  }
}
