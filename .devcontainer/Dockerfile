FROM mcr.microsoft.com/devcontainers/python:3.11

# Install system prerequisites and Node.js 22 (needed for npm-based tooling).
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates gnupg && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install OpenClaw globally when available on npm.
# If package resolution fails, post-create bootstrap will surface next steps.
RUN npm install -g openclaw@latest || true

# Install Akida simulation stack.
RUN pip install --upgrade pip && \
    pip install akida cnn2snn akida_models

# Create the OpenClaw workspace directory.
RUN mkdir -p /workspaces/openclaw-akida-agent-lab/agent_workspace
