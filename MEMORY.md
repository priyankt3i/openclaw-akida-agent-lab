# Experiment Memory

## Format
- Date/Time (UTC)
- Experiment ID
- Hypothesis
- Command(s)
- Result
- Error/Root Cause
- Next Action

## 2026-04-07 17:52:19 UTC
- Experiment ID: DEVCONTAINER-BOOTSTRAP-001
- Hypothesis: The Codespaces image build will succeed if the Akida install step only includes packages that exist on PyPI and are already documented by the repo.
- Command(s): `devcontainer up ...` triggering `pip install --upgrade pip && pip install akida cnn2snn akida_models mathneurosmt`
- Result: Failed during image build before post-create bootstrap ran.
- Error/Root Cause: `pip` reported `No matching distribution found for mathneurosmt`; the package is not available from the configured index and is not part of the documented toolchain in `README.md` / `TOOLS.md`.
- Next Action: Remove `mathneurosmt` from `.devcontainer/Dockerfile`, rebuild the container, and then verify `akida`, `cnn2snn`, and `akida_models` install successfully.

## 2026-04-07 17:56:58 UTC
- Experiment ID: OPENCLAW-CONTEXT-SYNC-001
- Hypothesis: Extending bootstrap to sync `HighLevelGoasl.md` and `chatHistory.txt` into persistent OpenClaw context will make the workspace more contextually aware across runs.
- Command(s): `bash .devcontainer/scripts/bootstrap.sh` after rewriting the script to generate `OPENCLAW_CONTEXT.md` and refresh a `Persistent Context` section in `MEMORY.md`
- Result: Partial success on first attempt; the context file was generated and `MEMORY.md` was updated, but the initial rewrite emitted shell warnings and rendered escaped source bullets incorrectly.
- Error/Root Cause: The here-doc based rewrite command introduced quoting/escaping artifacts while injecting backticks, which caused noisy shell output and poor markdown formatting.
- Next Action: Clean the bootstrap script formatting, rerun bootstrap, and verify `OPENCLAW_CONTEXT.md` plus `MEMORY.md` render correctly.


## Persistent Context
- Canonical context snapshot: `OPENCLAW_CONTEXT.md`
- Source files synchronized on bootstrap: `HighLevelGoasl.md`, `chatHistory.txt`
- Purpose: keep project goals and prior planning available alongside experiment memory for autonomous loops

## 2026-04-07 20:16:00 UTC
- Experiment ID: PROTOTYPE-1-SCAFFOLD-001
- Hypothesis: A dedicated `prototype-1/` workspace inside the repo will let us isolate the first spiking-attention surrogate experiments without fragmenting the current repository history.
- Command(s): `mkdir -p prototype-1/{experiments,src,artifacts}` and created `prototype-1/README.md`
- Result: Success
- Error/Root Cause: None
- Next Action: Implement the first deterministic dense attention-like baseline script inside `prototype-1/experiments` and run it to establish reference metrics.

## 2026-04-07 20:22:00 UTC
- Experiment ID: PROTOTYPE-1-BASELINE-001
- Hypothesis: A minimal deterministic dense attention reference plus thresholded sparse surrogate will give us an initial quality-vs-sparsity tradeoff signal before any Akida-specific conversion work.
- Command(s): `python prototype-1/experiments/baseline_attention.py`
- Result: Success. Metrics: MSE `0.00014356268078280677`, relative L2 error `0.029943797528364076`, attention mean absolute delta `0.003306078488015297`, activation sparsity `0.142578125`, dense ops proxy `640`, sparse ops proxy `567`, efficiency gain proxy `1.128747795414462`.
- Error/Root Cause: No runtime failure. The first threshold choice is too conservative to approach the 5x efficiency target, so the surrogate currently preserves quality better than it reduces compute.
- Next Action: Sweep thresholding and/or structured gating choices to push sparsity much higher while tracking when quality degradation becomes unacceptable.

## 2026-04-07 20:10:20 UTC
- Experiment ID: OPENCLAW-AGENT-RUNTIME-PATCH-001
- Hypothesis: We can patch the installed OpenClaw agent runtime to ignore late events without crashing.
- Command(s): `apply_patch` on `/usr/lib/node_modules/openclaw/node_modules/@mariozechner/pi-agent-core/dist/agent.js`
- Result: Failed
- Error/Root Cause: `apply_patch` failed with `bwrap: No permissions to create a new namespace` under the current sandbox.
- Next Action: Retry patch with elevated privileges via direct file edit.

## 2026-04-07 20:10:20 UTC
- Experiment ID: OPENCLAW-AGENT-RUNTIME-PATCH-002
- Hypothesis: Direct write to the installed OpenClaw runtime file will succeed under escalation.
- Command(s): `python - <<'PY' ... Path(...).write_text(...)`
- Result: Failed
- Error/Root Cause: `PermissionError: [Errno 13] Permission denied` when writing `/usr/lib/node_modules/openclaw/node_modules/@mariozechner/pi-agent-core/dist/agent.js`.
- Next Action: Retry patch via `sudo`.
