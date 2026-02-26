# ─────────────────────────────────────────────────────────────────────────────
# HASS-MCPProxy Add-on Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
ARG BUILD_FROM=ghcr.io/hassio-addons/base:16.3.2
FROM ${BUILD_FROM}

# Install system dependencies
RUN apk add --no-cache \
        python3 \
        py3-pip \
        py3-virtualenv \
        git \
        nodejs \
        npm \
        curl \
        bash \
        ca-certificates

# Create a virtualenv so we don't pollute the system Python
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install uv, mcp-proxy, and Python web server deps inside the venv
COPY app/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir uv mcp-proxy && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# UV_SYSTEM_PYTHON makes uv use the venv Python for uvx runs
ENV UV_PYTHON_PREFERENCE=only-system \
    UV_SYSTEM_PYTHON=1

# Copy add-on application files
COPY app/ /app/
COPY run.sh /run.sh
RUN chmod +x /run.sh

# Persistent data dir (mapped by HA supervisor to /data)
RUN mkdir -p /data

WORKDIR /app

CMD ["/run.sh"]
