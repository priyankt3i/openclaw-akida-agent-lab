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
