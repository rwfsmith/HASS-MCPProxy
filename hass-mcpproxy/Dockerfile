# ─────────────────────────────────────────────────────────────────────────────
# HASS-MCPProxy Add-on Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
ARG BUILD_FROM=ghcr.io/hassio-addons/base:16.3.2
FROM ${BUILD_FROM}

# Install system dependencies (no 'go' here – installed from official binary below)
RUN apk add --no-cache \
        python3 \
        py3-pip \
        py3-virtualenv \
        git \
        nodejs \
        npm \
        curl \
        bash \
        ca-certificates \
        # Chromium + runtime libraries required by Playwright \
        chromium \
        chromium-chromedriver \
        nss \
        freetype \
        freetype-dev \
        harfbuzz \
        ttf-freefont \
        font-noto-emoji \
        libstdc++ \
        libgcc \
        udev \
        at-spi2-core \
        mesa-gl \
        xvfb

# Tell Playwright to use the system Chromium instead of downloading its own.
# Playwright's bundled Chromium is glibc-compiled; Alpine uses musl – incompatible.
# Point everything at the Alpine-native /usr/bin/chromium-browser instead.
ENV PLAYWRIGHT_BROWSERS_PATH=/usr/bin \
    PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium-browser \
    PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 \
    PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium-browser \
    PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true \
    # Required flags for Chromium inside an unprivileged container \
    CHROMIUM_FLAGS="--no-sandbox --disable-dev-shm-usage --disable-gpu --headless"

# Install Go from the official binary tarball (arch-aware)
ARG GO_VERSION=1.23.5
RUN ARCH=$(uname -m) && \
    case "$ARCH" in \
        x86_64)  GOARCH=amd64   ;; \
        aarch64) GOARCH=arm64   ;; \
        armv7l)  GOARCH=armv6l  ;; \
        armv6l)  GOARCH=armv6l  ;; \
        i686)    GOARCH=386     ;; \
        *)       GOARCH=amd64   ;; \
    esac && \
    curl -fsSL "https://golang.org/dl/go${GO_VERSION}.linux-${GOARCH}.tar.gz" \
        | tar -C /usr/local -xz

ENV PATH="/usr/local/go/bin:$PATH" \
    GOPATH=/go \
    GOMODCACHE=/go/pkg/mod \
    GOCACHE=/go/cache \
    CGO_ENABLED=0
RUN mkdir -p /go/pkg/mod /go/cache

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
