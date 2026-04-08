# Lava feasibility note

Date: 2026-04-07 21:09 UTC

## Availability check
- Local Python environment: `import lava`, `lava.magma`, `lava.proc`, and `lava.lib.dl` all failed with `ModuleNotFoundError`.
- Local packages: no installed `lava*` packages were present in `pip list`.
- PyPI visibility: `pip index versions lava-nc` and `lava-dl` returned no matching distributions from this environment. `pip index versions lava` showed a package named `lava`, but that is not enough to confirm it is the Intel Lava neuromorphic stack we want.
- Docs: `https://lava-nc.org/` is reachable and describes Lava as an open-source framework for neuromorphic computing.
- Repo context: project goals already mention mapping the attention mechanism into Meta-TF or Lava.

## Fit for the current prototype
Current prototype structure is a good conceptual fit for Lava-style decomposition:
- Q/K/V projections map naturally to separate processes or process groups.
- Thresholding already creates sparse event activity (`kept_fraction = 0.140625`, about `7.1x` fewer active weights than dense), which aligns with event-driven execution.
- The prototype is small and deterministic, so it is a reasonable target for a first process-graph port.

## Main blockers
- No usable Lava runtime is installed locally, so we cannot validate APIs or execute a Lava graph yet.
- Softmax attention is the biggest mismatch. Exact softmax, global normalization, and dense score matrices are awkward for spike-native execution.
- The current surrogate is thresholded dense attention, not yet a true temporal spike formulation with membrane dynamics, spike timing, or rate coding.
- Direct Akida deployment is not guaranteed through Lava, so a Lava prototype may help algorithm exploration more than final hardware execution.

## Recommended mapping path
1. Port only the sparse surrogate first, not exact softmax.
2. Replace softmax with a spike-friendly approximation, for example thresholded competition, top-k gating, or local normalization.
3. Express Q/K/V projection, score accumulation, threshold gate, and value accumulation as separate Lava-style components.
4. Compare against current metrics: quality loss (`rel_mse = 0.000540`) and sparsity/energy proxy (`0.140625`).
5. Keep Lava as an exploration layer unless we also confirm a clean handoff into the Akida toolchain.

## Bottom line
Feasible as a research path, not yet executable in this workspace. Lava looks useful for prototyping a spike-native attention approximation, but the immediate blocker is missing local Lava packages, and the technical blocker is replacing exact softmax with something event-driven and hardware-friendly.
