#include "tests/test_framework.h"

#include "core/fields.h"
#include "host/stepper.h"
#include "params.h"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

using namespace dax;

static inline void ck(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}

static bool detect_current_is_one(const GridSpec& g,
                                  ModelParams& mp,
                                  RuntimeSpec& rt,
                                  BuffersDevice& b,
                                  LUTsDevice& l) {
  const float a = 111.0f;
  const float c = 222.0f;
  ck(cudaMemcpy(b.rho0, &a, sizeof(float), cudaMemcpyHostToDevice));
  ck(cudaMemcpy(b.rho1, &c, sizeof(float), cudaMemcpyHostToDevice));

  step_sim(0, g, mp, rt, b, l);
  ck(cudaGetLastError());
  ck(cudaDeviceSynchronize());

  float r0 = 0;
  float r1 = 0;
  ck(cudaMemcpy(&r0, b.rho0, sizeof(float), cudaMemcpyDeviceToHost));
  ck(cudaMemcpy(&r1, b.rho1, sizeof(float), cudaMemcpyDeviceToHost));

  const bool rho0_changed = std::abs(r0 - 111.0f) > 1e-6f;
  const bool rho1_changed = std::abs(r1 - 222.0f) > 1e-6f;

  if (rho0_changed == rho1_changed)
    return false;
  return rho1_changed;
}

static inline const float* curr_rho_ptr(const BuffersDevice& b, bool current_is_one) {
  return current_is_one ? b.rho1 : b.rho0;
}

static inline const float* curr_Rre_ptr(const BuffersDevice& b, bool current_is_one) {
  return current_is_one ? b.R1re : b.R0re;
}

static inline const float* curr_Rim_ptr(const BuffersDevice& b, bool current_is_one) {
  return current_is_one ? b.R1im : b.R0im;
}

struct Pulled {
  std::vector<float> rho, Rre, Rim, I, delta;
};

static Pulled pull_state(const GridSpec& g, const BuffersDevice& b, bool current_is_one) {
  const size_t N = static_cast<size_t>(g.Nx) * g.Ny * g.Nz * g.Nu;
  Pulled p;
  p.rho.resize(N);
  p.Rre.resize(N);
  p.Rim.resize(N);
  p.I.resize(N);
  p.delta.resize(N);
  ck(cudaMemcpy(p.rho.data(), curr_rho_ptr(b, current_is_one), N * sizeof(float), cudaMemcpyDeviceToHost));
  ck(cudaMemcpy(p.Rre.data(), curr_Rre_ptr(b, current_is_one), N * sizeof(float), cudaMemcpyDeviceToHost));
  ck(cudaMemcpy(p.Rim.data(), curr_Rim_ptr(b, current_is_one), N * sizeof(float), cudaMemcpyDeviceToHost));
  ck(cudaMemcpy(p.I.data(), b.I, N * sizeof(float), cudaMemcpyDeviceToHost));
  ck(cudaMemcpy(p.delta.data(), b.delta, N * sizeof(float), cudaMemcpyDeviceToHost));
  return p;
}

static void seed_ic(const GridSpec& g, BuffersDevice& b) {
  const size_t N = static_cast<size_t>(g.Nx) * g.Ny * g.Nz * g.Nu;
  std::vector<float> rho(N, 0.0f), Rre(N, 0.0f), Rim(N, 0.0f), Vre(N, 0.0f), Vim(N, 0.0f);

  const int cx = g.Nx / 2;
  const int cy = g.Ny / 2;
  const int cz = g.Nz / 2;
  const int cu = g.Nu / 2;
  rho[idx(cx, cy, cz, cu, g.Nx, g.Ny, g.Nz, g.Nu)] = 0.25f;
  Rre[idx(cx, cy, cz, cu, g.Nx, g.Ny, g.Nz, g.Nu)] = 1.0f;

  ck(cudaMemcpy(b.rho0, rho.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  ck(cudaMemcpy(b.rho1, rho.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  ck(cudaMemcpy(b.R0re, Rre.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  ck(cudaMemcpy(b.R1re, Rre.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  ck(cudaMemcpy(b.R0im, Rim.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  ck(cudaMemcpy(b.R1im, Rim.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  ck(cudaMemcpy(b.V0re, Vre.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  ck(cudaMemcpy(b.V1re, Vre.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  ck(cudaMemcpy(b.V0im, Vim.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  ck(cudaMemcpy(b.V1im, Vim.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  ck(cudaMemset(b.I, 0, N * sizeof(float)));
  ck(cudaMemset(b.delta, 0, N * sizeof(float)));
}

int main() {
  try {
    GridSpec g{};
    g.Nx = 32;
    g.Ny = 32;
    g.Nz = 16;
    g.Nu = 16;
    g.dt = 1e-4f;

    ModelParams mp{};
    RuntimeSpec rt_exp{};
    rt_exp.diffusion = DiffusionMethod::EXPLICIT_ADAPTIVE;
    RuntimeSpec rt_adi{};
    rt_adi.diffusion = DiffusionMethod::ADI_REFERENCE;

    BuffersDevice bE{}, bA{};
    LUTsDevice lE{}, lA{};
    alloc_fields(g, bE);
    alloc_luts(g, lE);
    init_default_luts(g, lE);
    alloc_fields(g, bA);
    alloc_luts(g, lA);
    init_default_luts(g, lA);

    seed_ic(g, bE);
    seed_ic(g, bA);

    const bool current_is_one = detect_current_is_one(g, mp, rt_exp, bE, lE);

    seed_ic(g, bE);
    seed_ic(g, bA);

    constexpr int steps = 200;
    for (int s = 0; s < steps; s++) {
      step_sim(s, g, mp, rt_exp, bE, lE);
      ck(cudaGetLastError());

      step_sim(s, g, mp, rt_adi, bA, lA);
      ck(cudaGetLastError());
    }
    ck(cudaDeviceSynchronize());

    auto E = pull_state(g, bE, current_is_one);
    auto A = pull_state(g, bA, current_is_one);

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

    free_luts(lE);
    free_fields(bE);
    free_luts(lA);
    free_fields(bA);

    std::cout << "✓ Tier 3 cross-validation PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Tier 3 cross-validation FAILED: " << e.what() << "\n";
    return 1;
  }
}
