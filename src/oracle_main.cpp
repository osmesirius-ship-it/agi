#include "tests/cpu_model.h"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

using namespace dax;

static int get_int(const char* s, int def) { return s ? std::atoi(s) : def; }
static float get_float(const char* s, float def) { return s ? std::atof(s) : def; }

static DiffusionMethod parse_diffusion(const std::string& mode) {
  if (mode == "adi") {
    return DiffusionMethod::ADI_REFERENCE;
  }
  return DiffusionMethod::EXPLICIT_ADAPTIVE;
}

static std::string json_escape(const std::string& in) {
  std::string out;
  out.reserve(in.size());
  for (char c : in) {
    if (c == '\\') {
      out += "\\\\";
    } else if (c == '"') {
      out += "\\\"";
    } else if (c == '\n') {
      out += "\\n";
    } else {
      out += c;
    }
  }
  return out;
}

static std::string build_response(const std::string& prompt, double rho_sum, double r_energy) {
  if (prompt.empty()) {
    return "Run complete. Ask about energy, mass, or diffusion for a quick summary.";
  }

  std::ostringstream out;
  out << "You asked: " << prompt << ". ";
  out << "Current rho_sum=" << rho_sum << " and R_energy=" << r_energy << ". ";

  if (prompt.find("diffusion") != std::string::npos) {
    out << "Use --diffusion explicit|adi to compare trajectories.";
  } else if (prompt.find("energy") != std::string::npos) {
    out << "Lower energy over time indicates expected damping behavior in Tier 2.";
  } else if (prompt.find("mass") != std::string::npos || prompt.find("rho") != std::string::npos) {
    out << "rho_sum should remain near the seeded total for stable timesteps.";
  } else {
    out << "Try prompts containing diffusion, energy, or mass for targeted diagnostics.";
  }

  return out.str();
}

int main(int argc, char** argv) {
  int steps = 200;
  float dt = 1e-4f;
  bool json = false;
  std::string diffusion = "explicit";
  std::string prompt;

  int positional = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--json") {
      json = true;
    } else if (arg == "--diffusion" && i + 1 < argc) {
      diffusion = argv[++i];
    } else if (arg == "--prompt" && i + 1 < argc) {
      prompt = argv[++i];
    } else if (positional == 0) {
      steps = get_int(argv[i], steps);
      ++positional;
    } else if (positional == 1) {
      dt = get_float(argv[i], dt);
      ++positional;
    }
  }

  GridSpec g{32, 32, 16, 16, dt};
  ModelParams mp{};
  RuntimeSpec rt{};
  rt.diffusion = parse_diffusion(diffusion);

  BuffersHost b{};
  alloc_fields(g, b);

  const int cx = g.Nx / 2, cy = g.Ny / 2, cz = g.Nz / 2, cu = g.Nu / 2;
  b.rho0[idx(cx, cy, cz, cu, g.Nx, g.Ny, g.Nz, g.Nu)] = 0.25f;
  b.rho1 = b.rho0;
  b.R0re[idx(cx, cy, cz, cu, g.Nx, g.Ny, g.Nz, g.Nu)] = 1.0f;
  b.R1re = b.R0re;

  for (int s = 0; s < steps; ++s) {
    step_sim_cpu(s, g, mp, rt, b);
  }

  const auto& rho = rho_current_vec(b);
  const auto& Rre = Rre_current_vec(b);
  const auto& Rim = Rim_current_vec(b);

  double rho_sum = 0.0;
  double R_energy = 0.0;
  for (size_t i = 0; i < rho.size(); ++i) {
    rho_sum += rho[i];
    R_energy += double(Rre[i]) * double(Rre[i]) + double(Rim[i]) * double(Rim[i]);
  }

  const std::string response = build_response(prompt, rho_sum, R_energy);

  if (json) {
    std::cout << "{\"mode\":\"" << diffusion << "\",\"steps\":" << steps << ",\"dt\":" << dt
              << ",\"rho_sum\":" << rho_sum << ",\"R_energy\":" << R_energy
              << ",\"response\":\"" << json_escape(response) << "\"}\n";
  } else {
    std::cout << "oracle_run\n";
    std::cout << "mode=" << diffusion << "\n";
    std::cout << "steps=" << steps << " dt=" << dt << "\n";
    std::cout << "rho_sum=" << rho_sum << "\n";
    std::cout << "R_energy=" << R_energy << "\n";
    if (!prompt.empty()) {
      std::cout << "assistant=" << response << "\n";
    }
  }

  return 0;
}
