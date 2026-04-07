# Tooling Notes
- Python: akida, cnn2snn, akida_models
- Runtime: OpenClaw (agent loop)
- Auth: Codex OAuth (device flow) preferred over raw API key

# Standard Checks
- node --version
- npm --version
- python -c "import akida; print(akida.__version__)"
- codex login --device-auth
- codex login status
