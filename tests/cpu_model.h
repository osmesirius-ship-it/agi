#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace dax {

enum class DiffusionMethod { EXPLICIT_ADAPTIVE, ADI_REFERENCE };

struct GridSpec {
  int Nx = 1;
  int Ny = 1;
  int Nz = 1;
  int Nu = 1;
  float dt = 1e-3f;
};

struct ModelParams {
  bool RFN_enabled = true;
  float D = 0.1f;
  float omega0 = 2.0f;
  float zeta = 0.2f;
  float coupling = 0.05f;
};

struct RuntimeSpec {
  DiffusionMethod diffusion = DiffusionMethod::EXPLICIT_ADAPTIVE;
};

inline int idx(int x, int y, int z, int u, int Nx, int Ny, int Nz, int Nu) {
  return (((z * Ny + y) * Nx + x) * Nu + u);
}

struct BuffersHost {
  std::vector<float> rho0, rho1;
  std::vector<float> R0re, R0im, R1re, R1im;
  std::vector<float> V0re, V0im, V1re, V1im;
  std::vector<float> I, delta, C;
  int ping = 0;
};

inline size_t total_size(const GridSpec& g) {
  return static_cast<size_t>(g.Nx) * g.Ny * g.Nz * g.Nu;
}

inline void alloc_fields(const GridSpec& g, BuffersHost& b) {
  const size_t N = total_size(g);
  b.rho0.assign(N, 0.0f); b.rho1.assign(N, 0.0f);
  b.R0re.assign(N, 0.0f); b.R0im.assign(N, 0.0f); b.R1re.assign(N, 0.0f); b.R1im.assign(N, 0.0f);
  b.V0re.assign(N, 0.0f); b.V0im.assign(N, 0.0f); b.V1re.assign(N, 0.0f); b.V1im.assign(N, 0.0f);
  b.I.assign(N, 0.0f); b.delta.assign(N, 0.0f); b.C.assign(N, 0.0f);
  b.ping = 0;
}

inline float* rho_curr(BuffersHost& b) { return b.ping ? b.rho1.data() : b.rho0.data(); }
inline float* rho_next(BuffersHost& b) { return b.ping ? b.rho0.data() : b.rho1.data(); }
inline float* Rre_curr(BuffersHost& b) { return b.ping ? b.R1re.data() : b.R0re.data(); }
inline float* Rim_curr(BuffersHost& b) { return b.ping ? b.R1im.data() : b.R0im.data(); }
inline float* Rre_next(BuffersHost& b) { return b.ping ? b.R0re.data() : b.R1re.data(); }
inline float* Rim_next(BuffersHost& b) { return b.ping ? b.R0im.data() : b.R1im.data(); }
inline float* Vre_curr(BuffersHost& b) { return b.ping ? b.V1re.data() : b.V0re.data(); }
inline float* Vim_curr(BuffersHost& b) { return b.ping ? b.V1im.data() : b.V0im.data(); }
inline float* Vre_next(BuffersHost& b) { return b.ping ? b.V0re.data() : b.V1re.data(); }
inline float* Vim_next(BuffersHost& b) { return b.ping ? b.V0im.data() : b.V1im.data(); }

inline const std::vector<float>& rho_current_vec(const BuffersHost& b) { return b.ping ? b.rho1 : b.rho0; }
inline const std::vector<float>& Rre_current_vec(const BuffersHost& b) { return b.ping ? b.R1re : b.R0re; }
inline const std::vector<float>& Rim_current_vec(const BuffersHost& b) { return b.ping ? b.R1im : b.R0im; }

inline int clampi(int v, int lo, int hi) {
  return std::max(lo, std::min(v, hi));
}

inline float laplacian_at(const GridSpec& g, const float* src, int x, int y, int z, int u) {
  const int i = idx(x, y, z, u, g.Nx, g.Ny, g.Nz, g.Nu);
  const int xm = idx(clampi(x - 1, 0, g.Nx - 1), y, z, u, g.Nx, g.Ny, g.Nz, g.Nu);
  const int xp = idx(clampi(x + 1, 0, g.Nx - 1), y, z, u, g.Nx, g.Ny, g.Nz, g.Nu);
  const int ym = idx(x, clampi(y - 1, 0, g.Ny - 1), z, u, g.Nx, g.Ny, g.Nz, g.Nu);
  const int yp = idx(x, clampi(y + 1, 0, g.Ny - 1), z, u, g.Nx, g.Ny, g.Nz, g.Nu);
  const int zm = idx(x, y, clampi(z - 1, 0, g.Nz - 1), u, g.Nx, g.Ny, g.Nz, g.Nu);
  const int zp = idx(x, y, clampi(z + 1, 0, g.Nz - 1), u, g.Nx, g.Ny, g.Nz, g.Nu);
  return src[xp] + src[xm] + src[yp] + src[ym] + src[zp] + src[zm] - 6.0f * src[i];
}

inline void diffuse_rho_explicit(const GridSpec& g, const ModelParams& mp, BuffersHost& b, float dt) {
  float* rc = rho_curr(b);
  float* rn = rho_next(b);
  for (int z = 0; z < g.Nz; ++z) {
    for (int y = 0; y < g.Ny; ++y) {
      for (int x = 0; x < g.Nx; ++x) {
        for (int u = 0; u < g.Nu; ++u) {
          const int i = idx(x, y, z, u, g.Nx, g.Ny, g.Nz, g.Nu);
          const float lap = laplacian_at(g, rc, x, y, z, u);
          rn[i] = rc[i] + dt * mp.D * lap;
        }
      }
    }
  }
}

inline void diffuse_rho_adi_reference(const GridSpec& g, const ModelParams& mp, BuffersHost& b, float dt) {
  float* rc = rho_curr(b);
  float* rn = rho_next(b);
  const size_t N = total_size(g);
  std::vector<float> stage1(N, 0.0f);

  for (int z = 0; z < g.Nz; ++z) {
    for (int y = 0; y < g.Ny; ++y) {
      for (int x = 0; x < g.Nx; ++x) {
        for (int u = 0; u < g.Nu; ++u) {
          const int i = idx(x, y, z, u, g.Nx, g.Ny, g.Nz, g.Nu);
          stage1[i] = rc[i] + dt * mp.D * laplacian_at(g, rc, x, y, z, u);
        }
      }
    }
  }

  for (int z = 0; z < g.Nz; ++z) {
    for (int y = 0; y < g.Ny; ++y) {
      for (int x = 0; x < g.Nx; ++x) {
        for (int u = 0; u < g.Nu; ++u) {
          const int i = idx(x, y, z, u, g.Nx, g.Ny, g.Nz, g.Nu);
          const float lap0 = laplacian_at(g, rc, x, y, z, u);
          const float lap1 = laplacian_at(g, stage1.data(), x, y, z, u);
          rn[i] = rc[i] + 0.5f * dt * mp.D * (lap0 + lap1);
        }
      }
    }
  }
}

inline void step_sim_cpu(int step, const GridSpec& g, const ModelParams& mp, const RuntimeSpec& rt, BuffersHost& b) {
  (void)step;
  const size_t N = total_size(g);
  float* rc = rho_curr(b);
  float* rn = rho_next(b);
  float* rre_c = Rre_curr(b);
  float* rim_c = Rim_curr(b);
  float* vre_c = Vre_curr(b);
  float* vim_c = Vim_curr(b);
  float* rre_n = Rre_next(b);
  float* rim_n = Rim_next(b);
  float* vre_n = Vre_next(b);
  float* vim_n = Vim_next(b);

  for (size_t i = 0; i < N; ++i) {
    b.I[i] = std::sqrt(rre_c[i] * rre_c[i] + rim_c[i] * rim_c[i]);
    b.C[i] = rc[i] + b.I[i];
  }

  if (rt.diffusion == DiffusionMethod::ADI_REFERENCE) {
    diffuse_rho_adi_reference(g, mp, b, g.dt);
  } else {
    diffuse_rho_explicit(g, mp, b, g.dt);
  }

  for (size_t i = 0; i < N; ++i) {
    rn[i] += g.dt * mp.coupling * (rre_c[i] - rc[i]);
    b.delta[i] = rn[i] - rc[i];

    const float force = mp.RFN_enabled ? rn[i] : 0.0f;
    const float damp = 2.0f * mp.zeta * mp.omega0;
    const float spring = mp.omega0 * mp.omega0;

    vre_n[i] = vre_c[i] + g.dt * (-damp * vre_c[i] - spring * rre_c[i] + force);
    vim_n[i] = vim_c[i] + g.dt * (-damp * vim_c[i] - spring * rim_c[i]);
    rre_n[i] = rre_c[i] + g.dt * vre_n[i];
    rim_n[i] = rim_c[i] + g.dt * vim_n[i];
  }

  b.ping ^= 1;
}

} // namespace dax
