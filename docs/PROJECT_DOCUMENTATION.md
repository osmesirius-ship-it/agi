# Project Documentation

## 1. Overview

This repository provides a **CPU-first simulation reference and validation harness** for a 4D flattened field model. It is designed to make numerical debugging deterministic and easy to run in any C++17 environment without requiring CUDA.

The primary deliverable is a 3-tier executable test suite:

- **Tier 1 (`tier1_numeric_tests`)**: numeric sanity and invariants.
- **Tier 2 (`tier2_physics_tests`)**: damped oscillator plausibility.
- **Tier 3 (`tier3_cross_validation`)**: explicit vs reference method cross-check.

## 2. Repository Layout

```text
.
├── CMakeLists.txt                 # Root build config (C++17 + tests)
├── src/
│   └── oracle_main.cpp            # Minimal runtime executable
├── ui/
│   └── chat.html                  # Browser chat interface
├── tools/
│   └── oracle_chat_server.py      # Local HTTP bridge to oracle binary
├── README.md                      # Quick-start and high-level usage
├── docs/
│   └── PROJECT_DOCUMENTATION.md   # This document
└── tests/
    ├── CMakeLists.txt             # Test executables + CTest registration
    ├── test_framework.h           # Assertions and vector diff utilities
    ├── cpu_model.h                # CPU reference model + stepper
    ├── tier1_numeric_tests.cpp    # Tier 1 executable
    ├── tier2_physics_tests.cpp    # Tier 2 executable
    └── tier3_cross_validation.cpp # Tier 3 executable
```

## 3. Build and Execution

### Configure & build

```bash
cmake -S . -B build
cmake --build build -j
```

### Run all tests via CTest

```bash
ctest --test-dir build --output-on-failure
```

### Run tests individually

```bash
./build/tests/tier1_numeric_tests
./build/tests/tier2_physics_tests
./build/tests/tier3_cross_validation
```

### Run the runtime executable

```bash
./build/oracle 200 1e-4
./build/oracle --diffusion adi --json

python3 tools/oracle_chat_server.py
# then open http://127.0.0.1:8080
```

## 4. Model Data Structures

The reference model lives in `tests/cpu_model.h`.

### Core config types

- `GridSpec`
  - dimensions: `Nx`, `Ny`, `Nz`, `Nu`
  - timestep: `dt`
- `ModelParams`
  - `RFN_enabled`, `D`, `omega0`, `zeta`, `coupling`
- `RuntimeSpec`
  - diffusion selector: `DiffusionMethod`

### Field storage

`BuffersHost` stores flattened vectors for:

- density ping-pong: `rho0`, `rho1`
- oscillator real/imag ping-pong: `R*`, `V*`
- derived fields: `I`, `delta`, `C`
- active buffer selector: `ping`

### Indexing convention

Fields are flattened in `[z][y][x][u]` order:

```cpp
idx(x,y,z,u) = (((z*Ny + y)*Nx + x)*Nu + u)
```

where `u` is contiguous (fastest-varying).

## 5. Simulation Pipeline (`step_sim_cpu`)

Each step follows this order:

1. Update intensity-like term `I` from complex state `R`.
2. Build coupling helper `C`.
3. Apply diffusion update on `rho`.
4. Apply coupling term into next `rho`.
5. Update `delta = rho_next - rho_curr`.
6. Update oscillator velocity/position (`V`, `R`) with damping and spring terms.
7. Toggle ping-pong state.

This fixed order allows deterministic comparisons and easier debugging.

## 6. Test Suite Semantics

## Tier 1: Numeric Correctness

Checks include:

- Zero-state remains finite and near-zero.
- Diffusion spreads a seeded center bump.
- Aggregate mass is finite and bounded.

## Tier 2: Physical Plausibility

Checks include:

- Oscillator energy remains finite.
- Oscillator energy decays monotonically over measured windows.

## Tier 3: Cross-Validation

Runs two simulations from identical initial conditions:

- `DiffusionMethod::EXPLICIT_ADAPTIVE`
- `DiffusionMethod::ADI_REFERENCE`

Then compares `rho`, `Rre`, `Rim`, `I`, and `delta` using relative-L2 and max-abs metrics.

## 7. Utility Test Framework

`tests/test_framework.h` provides:

- `ASSERT_TRUE`, `ASSERT_LT`
- `has_nan_inf(vector<float>)`
- `compare_rel_l2(a, b)` returning:
  - relative L2 error
  - max absolute error

This keeps test executables simple and consistent.

## 8. Extending the Project

### Add a new check to an existing tier

1. Add helper logic to the specific tier source file.
2. Use `ASSERT_*` macros from `test_framework.h`.
3. Keep thresholds explicit and printed where useful.

### Add a new tier executable

1. Create `tests/tierN_*.cpp`.
2. Register target and `add_test(...)` in `tests/CMakeLists.txt`.
3. Update `README.md` and this documentation.

### Tune parameters

Per-test parameters (grid size, timestep, damping, etc.) are intentionally local in each tier file for easy experimentation.

## 9. Troubleshooting

### Build fails

- Ensure CMake >= 3.21.
- Ensure a C++17-capable compiler is available.

### Test instability

- Reduce `dt` in the affected tier.
- Start from smaller grids to isolate indexing errors.
- Add temporary metric prints in the tier executable.

### Unexpected cross-validation divergence

- Verify both runs use identical seeding and step counts.
- Check method-specific branches in `step_sim_cpu`.
- Compare individual metrics (`rho`, `Rre`, `Rim`, `I`, `delta`) to localize drift.

## 10. Design Goal

This project intentionally prioritizes **clarity, determinism, and portability** over speed so correctness can be established before backend optimization work.
