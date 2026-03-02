# agi
agi field

## CPU tiered validation tests (no CUDA required)

This repository now ships a CPU-first validation path so tests run on CI/laptops without a GPU toolchain.

- `tier1_numeric_tests`: zero-state stability, diffusion spread, and mass sanity checks.
- `tier2_physics_tests`: damped oscillator boundedness/decay envelope checks.
- `tier3_cross_validation`: explicit-vs-reference diffusion comparison with frozen tolerances.

### CPU reference model

The tests use a readable host-side stepper (`step_sim_cpu`) with the same high-level order:

1. `I` update
2. `B/C` construction
3. diffusion
4. coupling
5. `delta` update
6. oscillator update (`R/V`)

Buffers use ping-pong host vectors in the same flattened layout `[z][y][x][u]` with `u` contiguous.

### Build and run

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```


## Full documentation

See `docs/PROJECT_DOCUMENTATION.md` for architecture, data model, test semantics, and extension guidelines.
