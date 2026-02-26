#include "tests/test_framework.h"

#include "core/fields.h"
#include "host/stepper.h"
#include "params.h"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

using namespace dax;

static inline void ck(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}

static std::vector<float> pull(const float* dptr, size_t n) {
  std::vector<float> h(n);
  ck(cudaMemcpy(h.data(), dptr, n * sizeof(float), cudaMemcpyDeviceToHost));
  return h;
}

static double sum_host(const std::vector<float>& v) {
  long double s = 0;
  for (float x : v)
    s += static_cast<long double>(x);
  return static_cast<double>(s);
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
    RuntimeSpec rt{};
    rt.diffusion = DiffusionMethod::EXPLICIT_ADAPTIVE;

    BuffersDevice b{};
    LUTsDevice l{};
    alloc_fields(g, b);
    alloc_luts(g, l);
    init_default_luts(g, l);

    const size_t N = static_cast<size_t>(g.Nx) * g.Ny * g.Nz * g.Nu;

    ck(cudaMemset(b.rho0, 0, N * sizeof(float)));
    ck(cudaMemset(b.rho1, 0, N * sizeof(float)));
    ck(cudaMemset(b.R0re, 0, N * sizeof(float)));
    ck(cudaMemset(b.R0im, 0, N * sizeof(float)));
    ck(cudaMemset(b.V0re, 0, N * sizeof(float)));
    ck(cudaMemset(b.V0im, 0, N * sizeof(float)));
    ck(cudaMemset(b.I, 0, N * sizeof(float)));
    ck(cudaMemset(b.delta, 0, N * sizeof(float)));

    for (int s = 0; s < 50; s++)
      step_sim(s, g, mp, rt, b, l);
    ck(cudaDeviceSynchronize());

    auto rho = pull(b.rho0, N);
    auto Rre = pull(b.R0re, N);
    auto I = pull(b.I, N);

    ASSERT_TRUE(!testfw::has_nan_inf(rho));
    ASSERT_TRUE(!testfw::has_nan_inf(Rre));
    ASSERT_TRUE(!testfw::has_nan_inf(I));

    auto max_abs = [](const std::vector<float>& v) {
      double m = 0;
      for (float x : v)
        m = std::max(m, std::abs(static_cast<double>(x)));
      return m;
    };

    ASSERT_LT(max_abs(rho), 1e-6);
    ASSERT_LT(max_abs(Rre), 1e-6);

    std::vector<float> rho_seed(N, 0.0f);
    const int cx = g.Nx / 2;
    const int cy = g.Ny / 2;
    const int cz = g.Nz / 2;
    const int cu = g.Nu / 2;
    rho_seed[idx(cx, cy, cz, cu, g.Nx, g.Ny, g.Nz, g.Nu)] = 1.0f;
    ck(cudaMemcpy(b.rho0, rho_seed.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    for (int s = 0; s < 30; s++)
      step_sim(s, g, mp, rt, b, l);
    ck(cudaDeviceSynchronize());

    auto rho2 = pull(b.rho0, N);
    const double center = rho2[idx(cx, cy, cz, cu, g.Nx, g.Ny, g.Nz, g.Nu)];
    ASSERT_LT(center, 1.0);

    const double s0 = sum_host(rho_seed);
    const double s1 = sum_host(rho2);
    ASSERT_TRUE(std::isfinite(s1));
    ASSERT_LT(std::abs(s1), 1e9);
    ASSERT_LT(std::abs(s1 - s0), 1e6);

    free_luts(l);
    free_fields(b);

    std::cout << "✓ Tier 1 numeric tests PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Tier 1 numeric tests FAILED: " << e.what() << "\n";
    return 1;
  }
}
