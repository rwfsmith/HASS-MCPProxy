# HASS-MCPProxy

A self-hosted MCP (Model Context Protocol) proxy for the [Home Assistant MCP Server integration](https://www.home-assistant.io/integrations/mcp_server/).

Built on top of [sparfenyuk/mcp-proxy](https://github.com/sparfenyuk/mcp-proxy) and adds:

- **Web UI** – manage servers directly from the Home Assistant sidebar (HA OS/Supervised add-on).
- **YAML-based configuration** – declare MCP servers in a config file.
- **GitHub server support** – specify a GitHub repo and the container clones, builds, and runs it automatically.
- **Secret injection** – reference environment variables (`${MY_VAR}`) in the config.
- **Named-server routing** – every configured server gets its own SSE endpoint.

---

## Installation

### Option A – Home Assistant Add-on (recommended for HassOS / Supervised)

1. In Home Assistant go to **Settings → Add-ons → Add-on Store → ⋮ → Repositories**.
2. Add this repository URL:
   ```
   https://github.com/rwfsmith/HASS-MCPProxy
   ```
3. Find **HASS MCP Proxy** in the store and click **Install**.
4. Start the add-on. A **MCP Proxy** entry will appear in your sidebar.
5. Open the UI to add and configure MCP servers.
6. Add each server to Home Assistant via **Settings → Devices & Services → MCP Server** using:
   ```
   http://homeassistant.local:8080/servers/<name>/sse
   ```

### Option B – Docker Compose (HA Container / Core / standalone)

See [Docker Compose Setup](#docker-compose-setup) below.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                   HASS-MCPProxy Container / Add-on                 │
│                                                                    │
│  ┌─────────────────────────────────┐                               │
│  │  Management API + Web UI :8099  │ ◄── HA Sidebar (Ingress)      │
│  │  (FastAPI)                      │                               │
│  │  • Add / edit / delete servers  │                               │
│  │  • Toggle enable/disable        │                               │
│  │  • Apply & Restart button       │                               │
│  └───────────────┬─────────────────┘                               │
│                  │ manages (subprocess)                             │
│  ┌───────────────▼─────────────────┐                               │
│  │  mcp-proxy :8080               │ ◄── HA MCP Server integration  │
│  │  /servers/<name>/sse            │     (one entry per server)     │
│  └───────────────┬─────────────────┘                               │
│                  │ stdio spawn                                      │
│       ┌──────────┼──────────┐                                      │
│   uvx/PyPI   npx/npm   GitHub clone                                │
│   servers    servers    Python/Node                                 │
└────────────────────────────────────────────────────────────────────┘
```

---

## Web UI

The add-on includes a full management UI accessible from the Home Assistant sidebar:

| Feature | Description |
|---|---|
| Server list | See all configured servers with enable/disable toggle |
| Add Server | Form with type-specific fields (PyPI, npm, GitHub, custom) |
| Edit / Delete | Modify or remove existing servers |
| Environment variables | Add key/value pairs with `${SECRET}` substitution support |
| Proxy settings | Port, log level, CORS, and environment pass-through |
| Apply & Restart | Reload mcp-proxy to pick up configuration changes |
| Status bar | Live proxy status, server counts, SSE base URL |

---

## Docker Compose Setup

### 1. Clone this repo

```bash
git clone https://github.com/rwfsmith/HASS-MCPProxy
cd HASS-MCPProxy
```

### 2. Create your `.env` file

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 3. Edit `config/servers.yaml`

Enable and configure the MCP servers you want. See [Configuration](#configuration) below.

### 4. Build and run

```bash
docker compose up --build -d
```

The proxy starts on port **8080** by default.

### 5. Add to Home Assistant

In Home Assistant go to **Settings → Devices & Services → Add Integration → MCP Server**.

For each named server, use the SSE URL:

```
http://<your-host>:8080/servers/<name>/sse
```

For example, with the `fetch` server enabled:

```
http://192.168.1.100:8080/servers/fetch/sse
```

---

## Configuration

All configuration lives in **`config/servers.yaml`**.

### Proxy settings

```yaml
proxy:
  port: 8080          # Port exposed by the container
  host: "0.0.0.0"    # Listen address (keep 0.0.0.0 for Docker)
  log_level: "INFO"   # DEBUG | INFO | WARNING | ERROR
  allow_origins:
    - "*"             # CORS origins; restrict for security
  pass_environment: false  # Forward all host env vars to child processes
```

### Server types

#### `uvx` – PyPI package (Python)

Runs any Python MCP server available on PyPI using `uvx` (no local install needed).

```yaml
- name: fetch
  enabled: true
  type: uvx
  package: mcp-server-fetch
  args: []
  env: {}
```

#### `npx` – npm package (Node.js)

Runs any Node.js MCP server available on npm using `npx`.

```yaml
- name: github
  enabled: true
  type: npx
  package: "@modelcontextprotocol/server-github"
  args: []
  env:
    GITHUB_PERSONAL_ACCESS_TOKEN: "${GITHUB_PAT}"
```

#### `github-python` – Python server from GitHub

Clones a GitHub repository, runs an install command, then starts the server.

```yaml
- name: my-python-server
  enabled: true
  type: github-python
  repo: "https://github.com/example/my-mcp-server"
  branch: "main"
  install: "uv pip install -e ."   # Runs once inside cloned dir
  run: "python -m my_mcp_server"   # Command to start the server
  args: []
  env:
    API_KEY: "${MY_API_KEY}"
```

#### `github-node` – Node.js server from GitHub

```yaml
- name: my-node-server
  enabled: true
  type: github-node
  repo: "https://github.com/example/my-node-mcp-server"
  branch: "main"
  install: "npm install && npm run build"
  run: "node dist/index.js"
  args: []
  env: {}
```

#### `command` – Arbitrary command

```yaml
- name: custom
  enabled: true
  type: command
  command: "/usr/local/bin/my-server"
  args: ["--port", "0"]
  env: {}
```

### Environment variable substitution

Any value in `env:` blocks may contain `${VAR_NAME}` placeholders.
Values are resolved from the container's environment at startup (populated from `.env`).

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `PROXY_PORT` | Host port mapped to the container's port 8080 | `8080` |
| `GITHUB_PAT` | GitHub Personal Access Token (for the GitHub MCP server) | – |

Add any additional variables your servers need to `.env`.

---

## Endpoints

Once running, the following endpoints are available:

| Path | Description |
|---|---|
| `GET /status` | Global proxy status (JSON) |
| `GET /servers/<name>/sse` | SSE endpoint for Home Assistant |
| `GET /servers/<name>/` | StreamableHTTP endpoint |

---

## Updating GitHub-Sourced Servers

Cloned repos are stored in the `repos` Docker volume. To pull the latest code:

```bash
docker compose restart
```

The entrypoint will `git pull` each repo on every startup.

To force a full re-clone, remove the volume:

```bash
docker compose down -v
docker compose up --build -d
```

---

## Rebuilding the Image

If you change `config/servers.yaml`, restart the container (no rebuild needed):

```bash
docker compose restart
```

If you change `Dockerfile` or `app/entrypoint.py`:

```bash
docker compose up --build -d
```

---

## Troubleshooting

### Checking logs

```bash
docker compose logs -f hass-mcpproxy
```

### Enable debug logging

Set `log_level: DEBUG` in `config/servers.yaml` and restart.

### Home Assistant can't connect

- Confirm the container is running: `docker compose ps`
- Confirm the port is reachable from HA: `curl http://<host>:8080/status`
- Check that the `name` in `servers.yaml` matches the URL path exactly.

---

## License

MIT
