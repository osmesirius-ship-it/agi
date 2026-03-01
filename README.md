# agi
agi field

## CUDA tiered validation tests

This repository includes three CUDA test executables under `tests/`:

- `tier1_numeric_tests`: zero-state stability, diffusion spread, and mass sanity checks.
- `tier2_physics_tests`: damped oscillator boundedness/decay envelope checks.
- `tier3_cross_validation`: explicit-vs-ADI field-level oracle comparison with frozen tolerances.

### Build and run

```bash
cmake -S . -B build
cmake --build build -j
./build/tier1_numeric_tests
./build/tier2_physics_tests
./build/tier3_cross_validation
```

### Ping-pong buffer selection

By default, tests read the `*0` field buffers as the current state. If your stepper keeps the
current state in `*1` after stepping, configure with:

```bash
cmake -S . -B build -DDAX_TEST_CURRENT_IS_1=ON
```

### CUDA launch hardening

Each test checks `cudaGetLastError()` immediately after every `step_sim(...)` call to catch launch
configuration/runtime errors at the step where they occur.
