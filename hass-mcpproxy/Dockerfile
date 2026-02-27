# ─────────────────────────────────────────────────────────────────────────────
# HASS-MCPProxy Add-on Dockerfile – Ubuntu 24.04 LTS (Noble)
# Ubuntu uses glibc so Playwright's bundled Chromium works natively.
# ubuntu:24.04 is a public multi-arch Docker Hub image (amd64, arm64).
# Hardcoded FROM prevents HA Supervisor from overriding with Alpine base.
# ─────────────────────────────────────────────────────────────────────────────
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        git \
        curl \
        bash \
        ca-certificates \
        gnupg \
        # Playwright / Chromium glibc runtime deps \
        libglib2.0-0 \
        libnss3 \
        libnspr4 \
        libatk1.0-0 \
        libatk-bridge2.0-0 \
        libcups2 \
        libdrm2 \
        libdbus-1-3 \
        libxcb1 \
        libxkbcommon0 \
        libx11-6 \
        libxcomposite1 \
        libxdamage1 \
        libxext6 \
        libxfixes3 \
        libxrandr2 \
        libgbm1 \
        libpango-1.0-0 \
        libcairo2 \
        libasound2 \
        libatspi2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Node.js 22 (via NodeSource) ───────────────────────────────────────────────
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Playwright configuration ──────────────────────────────────────────────────
# Store downloaded browsers in /data (persisted HA volume) so they survive
# container restarts. On first use Playwright downloads Chromium (~170 MB) once.
# No workarounds needed – Ubuntu glibc is fully compatible with Playwright.
ENV PLAYWRIGHT_BROWSERS_PATH=/data/playwright-browsers \
    PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true \
    CHROMIUM_FLAGS="--no-sandbox --disable-dev-shm-usage --disable-gpu"

# ── Go (official binary tarball, arch-aware) ──────────────────────────────────
ARG GO_VERSION=1.23.5
RUN ARCH=$(uname -m) && \
    case "$ARCH" in \
        x86_64)  GOARCH=amd64 ;; \
        aarch64) GOARCH=arm64 ;; \
        *)       GOARCH=amd64 ;; \
    esac && \
    curl -fsSL "https://golang.org/dl/go${GO_VERSION}.linux-${GOARCH}.tar.gz" \
        | tar -C /usr/local -xz

ENV PATH="/usr/local/go/bin:$PATH" \
    GOPATH=/go \
    GOMODCACHE=/go/pkg/mod \
    GOCACHE=/go/cache \
    CGO_ENABLED=0
RUN mkdir -p /go/pkg/mod /go/cache

# ── Python virtualenv + packages ─────────────────────────────────────────────
# Create a virtualenv so we don't pollute the system Python
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install uv, mcp-proxy, and Python web server deps
COPY app/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir uv mcp-proxy && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# UV_SYSTEM_PYTHON makes uv use the venv Python for uvx runs
ENV UV_PYTHON_PREFERENCE=only-system \
    UV_SYSTEM_PYTHON=1

# ── Add-on files ──────────────────────────────────────────────────────────────
# Copy add-on application files
COPY app/ /app/
COPY run.sh /run.sh
RUN chmod +x /run.sh

# Persistent data dir (mapped by HA supervisor to /data)
RUN mkdir -p /data /data/playwright-browsers

WORKDIR /app

CMD ["/run.sh"]
