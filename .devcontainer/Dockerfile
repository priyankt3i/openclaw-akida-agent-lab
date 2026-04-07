FROM mcr.microsoft.com/devcontainers/python:3.11

# Install Node.js 22 (Required for OpenClaw)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs

# Install OpenClaw globally
RUN npm install -g openclaw@latest

# Install Akida Simulation Stack
RUN pip install --upgrade pip && \
    pip install akida cnn2snn akida_models mathneurosmt

# Create the OpenClaw Workspace directory
RUN mkdir -p /workspaces/akida-agent-lab/agent_workspace

# Install the Codex CLI (The bridge for ChatGPT OAuth)
RUN curl -fsSL https://codexlang.com/install.sh | bash