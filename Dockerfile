# ─────────────────────────────────────────────────────────────────────────────
# HASS-MCPProxy
# Extends the official mcp-proxy image with:
#   • Python + uv  (for uvx and github-python servers)
#   • Node.js + npm/npx  (for npx and github-node servers)
#   • git  (for cloning GitHub repos)
#   • PyYAML  (for the Python entrypoint that parses servers.yaml)
# ─────────────────────────────────────────────────────────────────────────────
FROM ghcr.io/sparfenyuk/mcp-proxy:latest

# Install system deps: Python, pip, git, Node.js, npm
# The base image is Alpine Linux
RUN apk add --no-cache \
        python3 \
        py3-pip \
        git \
        nodejs \
        npm \
        curl \
        bash

# Install uv (fast Python package manager / runner)
RUN python3 -m ensurepip && \
    pip install --no-cache-dir uv pyyaml && \
    uv --version

# Ensure uvx is on PATH (uv installs it alongside uv)
ENV PATH="/root/.local/bin:/usr/local/bin:$PATH" \
    UV_PYTHON_PREFERENCE=only-system \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

# Copy application source
COPY app/ /app/app/

# Directories created at runtime (mounted or generated)
RUN mkdir -p /app/config /app/repos /app/generated

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Our Python script reads config/servers.yaml, builds GitHub-sourced servers,
# generates the mcp-proxy config JSON, and exec's mcp-proxy.
ENTRYPOINT ["python3", "/app/app/entrypoint.py"]
