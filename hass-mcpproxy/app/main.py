"""
HASS-MCPProxy – Management API + mcp-proxy supervisor
Serves on port 8099 (HA Ingress) and manages mcp-proxy (port 8080).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import signal
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
CONFIG_FILE = DATA_DIR / "servers.yaml"
GENERATED_CONFIG = DATA_DIR / "mcp_servers.json"
PLAYWRIGHT_CONFIG = DATA_DIR / "playwright-mcp-config.json"
STATIC_DIR = Path(__file__).parent / "ui"
MCP_PROXY_PORT = int(os.environ.get("MCP_PROXY_PORT", "8080"))
MCP_PROXY_HOST = os.environ.get("MCP_PROXY_HOST", "0.0.0.0")
REPOS_DIR = DATA_DIR / "repos"
# ── Read HA Supervisor add-on options (/data/options.json) ───────────────────
_OPTIONS_FILE = Path("/data/options.json")

def _load_ha_options() -> dict:
    if _OPTIONS_FILE.exists():
        try:
            import json as _json
            return _json.loads(_OPTIONS_FILE.read_text())
        except Exception:
            pass
    return {}

_ha_opts = _load_ha_options()

# ── Fetch HA home location from Supervisor API ────────────────────────────────
# Injects HA_LATITUDE, HA_LONGITUDE, HA_LOCATION_NAME, HA_TIMEZONE,
# HA_ELEVATION, HA_COUNTRY into os.environ so server configs can reference
# them with ${HA_LATITUDE} etc. Requires SUPERVISOR_TOKEN (auto-injected by HA).
def _inject_ha_location() -> None:
    token = os.environ.get("SUPERVISOR_TOKEN")
    if not token:
        return
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://supervisor/core/api/config",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        mapping = {
            "HA_LATITUDE":      str(data.get("latitude", "")),
            "HA_LONGITUDE":     str(data.get("longitude", "")),
            "HA_ELEVATION":     str(data.get("elevation", "")),
            "HA_LOCATION_NAME": str(data.get("location_name", "")),
            "HA_TIMEZONE":      str(data.get("time_zone", "")),
            "HA_COUNTRY":       str(data.get("country", "")),
        }
        for k, v in mapping.items():
            if v:
                os.environ.setdefault(k, v)
    except Exception as exc:
        # Non-fatal – log after logger is configured
        os.environ["_HA_LOCATION_ERR"] = str(exc)

_inject_ha_location()

LOG_LEVEL = _ha_opts.get("log_level", os.environ.get("LOG_LEVEL", "info")).upper()
ALLOW_ALL_ORIGINS = bool(_ha_opts.get("allow_all_origins", os.environ.get("ALLOW_ALL_ORIGINS", "true") == "true"))
PASS_ENVIRONMENT = bool(_ha_opts.get("pass_environment", os.environ.get("PASS_ENVIRONMENT", "false") == "true"))
UI_PORT = int(os.environ.get("UI_PORT", "8099"))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("hass-mcpproxy")

_loc_err = os.environ.pop("_HA_LOCATION_ERR", None)
if _loc_err:
    log.warning("Could not fetch HA location from Supervisor API: %s", _loc_err)
elif os.environ.get("HA_LATITUDE"):
    log.info("HA location: %s (%s, %s)", os.environ.get("HA_LOCATION_NAME"),
             os.environ.get("HA_LATITUDE"), os.environ.get("HA_LONGITUDE"))

# ── Default config ────────────────────────────────────────────────────────────
DEFAULT_CONFIG: dict[str, Any] = {
    "proxy": {
        "port": MCP_PROXY_PORT,
        "host": MCP_PROXY_HOST,
        "log_level": LOG_LEVEL,
        "allow_all_origins": ALLOW_ALL_ORIGINS,
        "pass_environment": PASS_ENVIRONMENT,
    },
    "servers": [],
}

# ── Environment variable substitution ─────────────────────────────────────────
_ENV_RE = re.compile(r"\$\{([^}]+)\}")


def resolve_env(value: str) -> str:
    def _sub(m: re.Match) -> str:
        return os.environ.get(m.group(1), "")
    return _ENV_RE.sub(_sub, str(value))


def resolve_env_dict(d: dict) -> dict:
    return {k: resolve_env(v) for k, v in (d or {}).items()}


# ── Config I/O ────────────────────────────────────────────────────────────────

def load_config() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CONFIG_FILE.exists():
        with CONFIG_FILE.open() as fh:
            return yaml.safe_load(fh) or DEFAULT_CONFIG
    save_config(DEFAULT_CONFIG)
    return DEFAULT_CONFIG


def save_config(cfg: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_FILE.open("w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)


# ── mcp-proxy subprocess ──────────────────────────────────────────────────────

class ProxyManager:
    def __init__(self) -> None:
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def start(self) -> None:
        async with self._lock:
            if self.running:
                return
            await self._launch()

    async def restart(self) -> None:
        async with self._lock:
            await self._stop()
            await self._launch()

    async def stop(self) -> None:
        async with self._lock:
            await self._stop()

    async def _launch(self) -> None:
        cfg = load_config()
        proxy_cfg = cfg.get("proxy", {})
        servers = cfg.get("servers", [])
        enabled = [s for s in servers if s.get("enabled", True)]

        if not enabled:
            log.warning("No enabled servers – mcp-proxy will not start.")
            return

        # Write Playwright config (--no-sandbox required when running as root in Docker)
        PLAYWRIGHT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        PLAYWRIGHT_CONFIG.write_text(json.dumps({
            "browser": {
                "launchOptions": {
                    "args": ["--no-sandbox", "--disable-dev-shm-usage"]
                }
            }
        }, indent=2))

        # Build generated config for --named-server-config
        mcp_servers: dict[str, dict] = {}
        for srv in enabled:
            name = srv.get("name", "")
            if not name:
                continue
            entry = _build_server_entry(srv)
            if entry:
                entry["transportType"] = "stdio"
                mcp_servers[name] = entry

        GENERATED_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        GENERATED_CONFIG.write_text(json.dumps({"mcpServers": mcp_servers}, indent=2))

        port = proxy_cfg.get("port", MCP_PROXY_PORT)
        host = proxy_cfg.get("host", MCP_PROXY_HOST)
        level = proxy_cfg.get("log_level", "INFO")
        allow_all = proxy_cfg.get("allow_all_origins", ALLOW_ALL_ORIGINS)
        pass_env = proxy_cfg.get("pass_environment", PASS_ENVIRONMENT)

        cmd = [
            "mcp-proxy",
            f"--port={port}",
            f"--host={host}",
            f"--log-level={level}",
            "--named-server-config", str(GENERATED_CONFIG),
        ]
        if pass_env:
            cmd.append("--pass-environment")
        if allow_all:
            cmd.extend(["--allow-origin", "*"])

        log.info("Launching mcp-proxy: %s", " ".join(shlex.quote(c) for c in cmd))
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        asyncio.create_task(self._log_output())
        log.info("mcp-proxy started (pid=%d)", self._process.pid)

    async def _stop(self) -> None:
        if self._process and self._process.returncode is None:
            log.info("Stopping mcp-proxy (pid=%d)…", self._process.pid)
            try:
                self._process.send_signal(signal.SIGTERM)
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
        self._process = None

    async def _log_output(self) -> None:
        if self._process and self._process.stdout:
            async for line in self._process.stdout:
                log.info("[mcp-proxy] %s", line.decode().rstrip())


proxy_manager = ProxyManager()


# ── Server entry builder ──────────────────────────────────────────────────────

def _build_server_entry(srv: dict) -> dict | None:
    stype = srv.get("type", "").lower()
    args = [str(a) for a in (srv.get("args") or [])]
    env = resolve_env_dict(srv.get("env") or {})

    if stype == "uvx":
        pkg = srv.get("package", "")
        if not pkg:
            return None
        return {"command": "uvx", "args": [pkg, *args], "env": env}

    if stype == "npx":
        pkg = srv.get("package", "")
        if not pkg:
            return None
        # For @playwright/mcp: strip any --launch-options args (not supported),
        # and auto-inject --config pointing to our pre-written config file that
        # provides --no-sandbox (required when running as root in Docker).
        if "playwright/mcp" in pkg:
            filtered_args: list[str] = []
            skip_next = False
            for a in args:
                if skip_next:
                    skip_next = False
                    continue
                if a == "--launch-options":
                    skip_next = True
                    continue
                if a.startswith("--launch-options="):
                    continue
                filtered_args.append(a)
            args = filtered_args
            if "--config" not in args:
                args = ["--config", str(PLAYWRIGHT_CONFIG)] + args
        return {"command": "npx", "args": ["-y", pkg, *args], "env": env}

    if stype in ("github-python", "github-node", "github-go"):
        name = srv.get("name", "")
        repo = srv.get("repo", "")
        branch = srv.get("branch", "main")
        if stype == "github-python":
            default_install = "uv pip install -e ."
        elif stype == "github-node":
            default_install = "npm install"
        else:
            default_install = "go build ."
        install_cmd = srv.get("install", default_install)
        run_cmd = srv.get("run", "")
        if not repo or not run_cmd:
            return None
        dest = REPOS_DIR / name
        # Clone is done synchronously at startup via _prepare_github (see below)
        parts = shlex.split(run_cmd) + args
        # Resolve relative paths (./binary, bin/server) to absolute so mcp-proxy
        # can find the binary regardless of its own working directory.
        cmd_part = parts[0]
        if not cmd_part.startswith("/"):
            cmd_part = str((dest / cmd_part).resolve())
        return {"command": cmd_part, "args": parts[1:], "env": env}

    if stype == "command":
        cmd = srv.get("command", "")
        if not cmd:
            return None
        return {"command": cmd, "args": args, "env": env}

    return None


def _prepare_github_repos(servers: list[dict]) -> None:
    """Synchronously clone/update GitHub repos before starting mcp-proxy."""
    import subprocess
    for srv in servers:
        if not srv.get("enabled", True):
            continue
        stype = srv.get("type", "").lower()
        if stype not in ("github-python", "github-node", "github-go"):
            continue
        name = srv.get("name", "")
        repo = srv.get("repo", "")
        branch = srv.get("branch", "main")
        if stype == "github-python":
            default_install = "uv pip install -e ."
        elif stype == "github-node":
            default_install = "npm install"
        else:
            default_install = "go build ."
        install_cmd = srv.get("install", default_install)
        dest = REPOS_DIR / name
        REPOS_DIR.mkdir(parents=True, exist_ok=True)
        def _run(cmd: str, **kwargs) -> None:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, **kwargs
            )
            if result.stdout.strip():
                for line in result.stdout.strip().splitlines():
                    log.info("[build:%s] %s", name, line)
            if result.stderr.strip():
                for line in result.stderr.strip().splitlines():
                    log.warning("[build:%s] %s", name, line)
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd)

        try:
            if dest.exists():
                log.info("Updating repo '%s'…", name)
                _run(f"git fetch origin && git checkout {branch} && git pull", cwd=dest)
            else:
                log.info("Cloning '%s' (%s)…", repo, branch)
                _run(
                    f"git clone --depth=1 --branch {shlex.quote(branch)} "
                    f"{shlex.quote(repo)} {shlex.quote(str(dest))}"
                )
            log.info("Running install for '%s': %s", name, install_cmd)
            _run(install_cmd, cwd=dest)
            log.info("Build complete for '%s'. Contents of %s:", name, dest)
            for f in sorted(dest.rglob("*")):
                if f.is_file() and not any(p.startswith(".") for p in f.parts):
                    log.info("[build:%s]   %s", name, f.relative_to(dest))
        except subprocess.CalledProcessError as exc:
            log.error("Failed to prepare GitHub server '%s' (exit %d) – check build logs above",
                      name, exc.returncode)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    servers = cfg.get("servers", [])
    _prepare_github_repos(servers)
    await proxy_manager.start()
    yield
    await proxy_manager.stop()


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="HASS-MCPProxy", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class ServerBase(BaseModel):
    name: str = Field(..., description="Server name (used in URL path)")
    description: str = Field("", description="Human-readable description")
    enabled: bool = Field(True)
    type: str = Field(..., description="uvx | npx | github-python | github-node | command")
    # uvx / npx
    package: str | None = Field(None)
    # github-*
    repo: str | None = Field(None)
    branch: str | None = Field("main")
    install: str | None = Field(None)
    run: str | None = Field(None)
    # command
    command: str | None = Field(None)
    # common
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class ProxySettings(BaseModel):
    port: int = Field(MCP_PROXY_PORT)
    host: str = Field(MCP_PROXY_HOST)
    log_level: str = Field("INFO")
    allow_all_origins: bool = Field(True)
    pass_environment: bool = Field(False)


# ── API routes ────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    cfg = load_config()
    servers = cfg.get("servers", [])
    return {
        "proxy_running": proxy_manager.running,
        "proxy_pid": proxy_manager._process.pid if proxy_manager.running else None,
        "server_count": len(servers),
        "enabled_count": sum(1 for s in servers if s.get("enabled", True)),
        "mcp_proxy_url": f"http://<your-ha-host>:{cfg.get('proxy', {}).get('port', MCP_PROXY_PORT)}",
    }


@app.get("/api/servers")
async def list_servers():
    cfg = load_config()
    return cfg.get("servers", [])


@app.post("/api/servers", status_code=201)
async def add_server(server: ServerBase):
    cfg = load_config()
    servers: list[dict] = cfg.setdefault("servers", [])
    if any(s["name"] == server.name for s in servers):
        raise HTTPException(status_code=409, detail=f"Server '{server.name}' already exists.")
    servers.append(server.model_dump(exclude_none=False))
    save_config(cfg)
    return {"ok": True, "name": server.name}


@app.put("/api/servers/{name}")
async def update_server(name: str, server: ServerBase):
    cfg = load_config()
    servers: list[dict] = cfg.get("servers", [])
    for i, s in enumerate(servers):
        if s["name"] == name:
            servers[i] = {**server.model_dump(exclude_none=False), "name": name}
            save_config(cfg)
            return {"ok": True}
    raise HTTPException(status_code=404, detail=f"Server '{name}' not found.")


@app.delete("/api/servers/{name}")
async def delete_server(name: str):
    cfg = load_config()
    servers: list[dict] = cfg.get("servers", [])
    original = len(servers)
    cfg["servers"] = [s for s in servers if s["name"] != name]
    if len(cfg["servers"]) == original:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found.")
    save_config(cfg)
    return {"ok": True}


@app.patch("/api/servers/{name}/toggle")
async def toggle_server(name: str):
    cfg = load_config()
    for s in cfg.get("servers", []):
        if s["name"] == name:
            s["enabled"] = not s.get("enabled", True)
            save_config(cfg)
            return {"ok": True, "enabled": s["enabled"]}
    raise HTTPException(status_code=404, detail=f"Server '{name}' not found.")


@app.get("/api/proxy-settings")
async def get_proxy_settings():
    cfg = load_config()
    return cfg.get("proxy", DEFAULT_CONFIG["proxy"])


@app.put("/api/proxy-settings")
async def update_proxy_settings(settings: ProxySettings):
    cfg = load_config()
    cfg["proxy"] = settings.model_dump()
    save_config(cfg)
    return {"ok": True}


@app.post("/api/reload")
async def reload_proxy():
    """Apply configuration changes by restarting mcp-proxy."""
    cfg = load_config()
    _prepare_github_repos(cfg.get("servers", []))
    await proxy_manager.restart()
    return {"ok": True, "running": proxy_manager.running}


# ── Static UI ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
async def serve_ui(request: Request):
    index = STATIC_DIR / "index.html"
    html = index.read_text()
    # HA Supervisor sets X-Ingress-Path to the ingress prefix, e.g.
    # /api/hassio_ingress/TOKEN  – inject it so JS API calls are correct.
    ingress_path = request.headers.get("X-Ingress-Path", "").rstrip("/")
    html = html.replace(
        "__INGRESS_PATH__",
        ingress_path,
    )
    return HTMLResponse(html)


static_subdir = STATIC_DIR / "static"
if static_subdir.exists():
    app.mount("/static", StaticFiles(directory=str(static_subdir)), name="static")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=UI_PORT,
        log_level=LOG_LEVEL.lower(),
        access_log=False,
    )
