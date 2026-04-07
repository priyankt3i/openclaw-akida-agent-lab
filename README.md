# openclaw-akida-agent-lab

Codespaces workspace for building a neuromorphic "spiking transformer block" prototype on Akida tooling with an autonomous OpenClaw loop.

## Project Goal
- Convert an attention-like transformer component from a small dense model into an SNN-compatible form.
- Target measurable improvement: around 5x energy reduction with less than 1% quality drop vs FP32 baseline.
- Keep all experiments reproducible and logged.

## 1. Rebuild Into the Devcontainer
After opening the repository in GitHub Codespaces, run:
- `Ctrl+Shift+P` (or `Cmd+Shift+P`)
- `Codespaces: Rebuild Container`

This repo depends on the custom `.devcontainer` image to install Node.js, Akida Python packages, and bootstrap files.

## 2. Verify Tools
Run:

```bash
node --version
npm --version
python -c "import akida; print(akida.__version__)"
```

Optional checks:

```bash
openclaw --version
codex --version
```

## 3. Authenticate Codex (OAuth Device Flow)
If you are using Codex OAuth instead of an API key:

```bash
codex login --device-auth
codex login status
```

Open the device-auth link on your local browser and complete authorization.

## 4. Initialize/OpenClaw
The post-create bootstrap script will automatically:
- Validate Node/npm availability
- Attempt OpenClaw install if missing
- Create and seed `AGENTS.md`, `TOOLS.md`, and `MEMORY.md`

Manual rerun:

```bash
bash .devcontainer/scripts/bootstrap.sh
```

If OpenClaw exists in your environment, start your loop with your preferred command.

## 5. Working Conventions
- Record each experiment outcome in `MEMORY.md`.
- Keep agent mission/constraints in `AGENTS.md`.
- Track available commands and tooling in `TOOLS.md`.

## Troubleshooting
- `npm: command not found` or `node: command not found`: rebuild container.
- `openclaw: command not found`: run `bash .devcontainer/scripts/bootstrap.sh` and inspect output.
- Codex auth issues: rerun `codex login --device-auth` and confirm with `codex login status`.
