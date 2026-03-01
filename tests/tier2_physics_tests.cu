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

static inline const float* curr_Rre_ptr(const BuffersDevice& b, bool current_is_one) {
  return current_is_one ? b.R1re : b.R0re;
}

static inline const float* curr_Rim_ptr(const BuffersDevice& b, bool current_is_one) {
  return current_is_one ? b.R1im : b.R0im;
}

static std::vector<float> pull(const float* dptr, size_t n) {
  std::vector<float> h(n);
  ck(cudaMemcpy(h.data(), dptr, n * sizeof(float), cudaMemcpyDeviceToHost));
  return h;
}

static double energy_R(const std::vector<float>& Rre, const std::vector<float>& Rim) {
  long double e = 0;
  for (size_t i = 0; i < Rre.size(); i++) {
    const long double re = Rre[i];
    const long double im = Rim[i];
    e += re * re + im * im;
  }
  return static_cast<double>(e);
}

int main() {
  try {
    GridSpec g{};
    g.Nx = 32;
    g.Ny = 32;
    g.Nz = 16;
    g.Nu = 16;
    g.dt = 1e-3f;

    ModelParams mp{};
    mp.RFN_enabled = false;

    RuntimeSpec rt{};
    rt.diffusion = DiffusionMethod::EXPLICIT_ADAPTIVE;

    BuffersDevice b{};
    LUTsDevice l{};
    alloc_fields(g, b);
    alloc_luts(g, l);
    init_default_luts(g, l);

    const size_t N = static_cast<size_t>(g.Nx) * g.Ny * g.Nz * g.Nu;

    std::vector<float> Rre(N, 0.0f), Rim(N, 0.0f);
    const int cx = g.Nx / 2;
    const int cy = g.Ny / 2;
    const int cz = g.Nz / 2;
    const int cu = g.Nu / 2;
    Rre[idx(cx, cy, cz, cu, g.Nx, g.Ny, g.Nz, g.Nu)] = 1.0f;

    ck(cudaMemcpy(b.R0re, Rre.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(b.R1re, Rre.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(b.R0im, Rim.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(b.R1im, Rim.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    ck(cudaMemset(b.V0re, 0, N * sizeof(float)));
    ck(cudaMemset(b.V1re, 0, N * sizeof(float)));
    ck(cudaMemset(b.V0im, 0, N * sizeof(float)));
    ck(cudaMemset(b.V1im, 0, N * sizeof(float)));
    ck(cudaMemset(b.rho0, 0, N * sizeof(float)));
    ck(cudaMemset(b.rho1, 0, N * sizeof(float)));
    ck(cudaMemset(b.I, 0, N * sizeof(float)));
    ck(cudaMemset(b.delta, 0, N * sizeof(float)));

    const bool current_is_one = detect_current_is_one(g, mp, rt, b, l);

    ck(cudaMemcpy(b.R0re, Rre.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(b.R1re, Rre.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(b.R0im, Rim.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(b.R1im, Rim.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    ck(cudaMemset(b.V0re, 0, N * sizeof(float)));
    ck(cudaMemset(b.V1re, 0, N * sizeof(float)));
    ck(cudaMemset(b.V0im, 0, N * sizeof(float)));
    ck(cudaMemset(b.V1im, 0, N * sizeof(float)));
    ck(cudaMemset(b.rho0, 0, N * sizeof(float)));
    ck(cudaMemset(b.rho1, 0, N * sizeof(float)));
    ck(cudaMemset(b.I, 0, N * sizeof(float)));
    ck(cudaMemset(b.delta, 0, N * sizeof(float)));

    double e0 = 0;
    double e_mid = 0;
    double e_end = 0;

    for (int s = 0; s < 400; s++) {
      step_sim(s, g, mp, rt, b, l);
      ck(cudaGetLastError());

      if (s == 0) {
        auto a = pull(curr_Rre_ptr(b, current_is_one), N);
        auto b2 = pull(curr_Rim_ptr(b, current_is_one), N);
        e0 = energy_R(a, b2);
      }
      if (s == 200) {
        auto a = pull(curr_Rre_ptr(b, current_is_one), N);
        auto b2 = pull(curr_Rim_ptr(b, current_is_one), N);
        e_mid = energy_R(a, b2);
      }
    }
    ck(cudaDeviceSynchronize());

    {
      auto a = pull(curr_Rre_ptr(b, current_is_one), N);
      auto b2 = pull(curr_Rim_ptr(b, current_is_one), N);
      e_end = energy_R(a, b2);
    }

    ASSERT_TRUE(std::isfinite(e0));
    ASSERT_TRUE(std::isfinite(e_mid));
    ASSERT_TRUE(std::isfinite(e_end));
    ASSERT_LT(e_end, 1e6);
    ASSERT_LT(e_end, e0 * 1.05);

    free_luts(l);
    free_fields(b);

    std::cout << "✓ Tier 2 physics tests PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Tier 2 physics tests FAILED: " << e.what() << "\n";
    return 1;
  }
}
