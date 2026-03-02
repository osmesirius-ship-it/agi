#include "tests/cpu_model.h"
#include "tests/test_framework.h"

#include <iostream>
#include <vector>

using namespace dax;

struct Pulled {
  std::vector<float> rho, Rre, Rim, I, delta;
};

static Pulled pull_state(const BuffersHost& b) {
  Pulled p;
  p.rho = rho_current_vec(b);
  p.Rre = Rre_current_vec(b);
  p.Rim = Rim_current_vec(b);
  p.I = b.I;
  p.delta = b.delta;
  return p;
}

static void seed_ic(const GridSpec& g, BuffersHost& b) {
  const size_t N = total_size(g);
  std::vector<float> rho(N, 0.0f), Rre(N, 0.0f), Rim(N, 0.0f);

  const int cx = g.Nx / 2, cy = g.Ny / 2, cz = g.Nz / 2, cu = g.Nu / 2;
  rho[idx(cx, cy, cz, cu, g.Nx, g.Ny, g.Nz, g.Nu)] = 0.25f;
  Rre[idx(cx, cy, cz, cu, g.Nx, g.Ny, g.Nz, g.Nu)] = 1.0f;

  b.rho0 = rho; b.rho1 = rho;
  b.R0re = Rre; b.R1re = Rre;
  b.R0im = Rim; b.R1im = Rim;
  std::fill(b.V0re.begin(), b.V0re.end(), 0.0f);
  std::fill(b.V1re.begin(), b.V1re.end(), 0.0f);
  std::fill(b.V0im.begin(), b.V0im.end(), 0.0f);
  std::fill(b.V1im.begin(), b.V1im.end(), 0.0f);
  std::fill(b.I.begin(), b.I.end(), 0.0f);
  std::fill(b.delta.begin(), b.delta.end(), 0.0f);
  b.ping = 0;
}

int main() {
  try {
    GridSpec g{32, 32, 16, 16, 1e-4f};
    ModelParams mp{};

    RuntimeSpec rt_exp{};
    rt_exp.diffusion = DiffusionMethod::EXPLICIT_ADAPTIVE;
    RuntimeSpec rt_adi{};
    rt_adi.diffusion = DiffusionMethod::ADI_REFERENCE;

    BuffersHost bE{}, bA{};
    alloc_fields(g, bE);
    alloc_fields(g, bA);

    seed_ic(g, bE);
    seed_ic(g, bA);

    constexpr int steps = 200;
    for (int s = 0; s < steps; ++s) {
      step_sim_cpu(s, g, mp, rt_exp, bE);
      step_sim_cpu(s, g, mp, rt_adi, bA);
    }

    const auto E = pull_state(bE);
    const auto A = pull_state(bA);

    const auto drho = testfw::compare_rel_l2(E.rho, A.rho);
    const auto dRre = testfw::compare_rel_l2(E.Rre, A.Rre);
    const auto dRim = testfw::compare_rel_l2(E.Rim, A.Rim);
    const auto dI = testfw::compare_rel_l2(E.I, A.I);
    const auto ddel = testfw::compare_rel_l2(E.delta, A.delta);

    std::cout << "rho   l2=" << drho.l2_rel << " max=" << drho.max_abs << "\n";
    std::cout << "Rre   l2=" << dRre.l2_rel << " max=" << dRre.max_abs << "\n";
    std::cout << "Rim   l2=" << dRim.l2_rel << " max=" << dRim.max_abs << "\n";
    std::cout << "I     l2=" << dI.l2_rel << " max=" << dI.max_abs << "\n";
    std::cout << "delta l2=" << ddel.l2_rel << " max=" << ddel.max_abs << "\n";

    ASSERT_LT(drho.l2_rel, 1e-4);
    ASSERT_LT(dRre.l2_rel, 2e-4);
    ASSERT_LT(dRim.l2_rel, 2e-4);
    ASSERT_LT(dI.l2_rel, 1e-5);
    ASSERT_LT(ddel.l2_rel, 5e-4);

    ASSERT_LT(drho.max_abs, 1e-3);
    ASSERT_LT(dRre.max_abs, 1e-3);
    ASSERT_LT(dRim.max_abs, 1e-3);
    ASSERT_LT(dI.max_abs, 1e-3);
    ASSERT_LT(ddel.max_abs, 1e-3);

    std::cout << "✓ Tier 3 cross-validation PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Tier 3 cross-validation FAILED: " << e.what() << "\n";
    return 1;
  }
}
