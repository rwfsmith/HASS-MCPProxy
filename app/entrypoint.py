#!/usr/bin/env python3
"""
HASS-MCPProxy entrypoint.

Reads config/servers.yaml, optionally clones and builds GitHub-sourced MCP
servers, writes a mcp-proxy named-server-config JSON file, then exec's
mcp-proxy so it becomes PID 1's direct child (clean signal handling).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path("/app")
CONFIG_FILE = ROOT_DIR / "config" / "servers.yaml"
REPOS_DIR = ROOT_DIR / "repos"
GENERATED_CONFIG = ROOT_DIR / "generated" / "mcp_servers.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("hass-mcpproxy")


# ── Helpers ──────────────────────────────────────────────────────────────────

_ENV_RE = re.compile(r"\$\{([^}]+)\}")


def resolve_env(value: str) -> str:
    """Replace ${VAR} placeholders with environment variable values."""
    def _sub(m: re.Match) -> str:
        var = m.group(1)
        val = os.environ.get(var, "")
        if not val:
            log.warning("Environment variable '%s' is not set (or empty).", var)
        return val

    return _ENV_RE.sub(_sub, value)


def resolve_env_dict(d: dict[str, str]) -> dict[str, str]:
    return {k: resolve_env(v) for k, v in (d or {}).items()}


def run(cmd: str, cwd: Path | None = None, env: dict | None = None) -> None:
    """Run a shell command, raising on failure."""
    log.info("$ %s", cmd)
    subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        env={**os.environ, **(env or {})},
        check=True,
    )


# ── GitHub repo management ───────────────────────────────────────────────────

def clone_or_update(repo_url: str, branch: str, dest: Path) -> None:
    """Clone *repo_url* into *dest*, or pull if it already exists."""
    if dest.exists():
        log.info("Updating existing clone at %s …", dest)
        run(f"git fetch origin && git checkout {branch} && git pull", cwd=dest)
    else:
        log.info("Cloning %s (branch: %s) → %s …", repo_url, branch, dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        run(f"git clone --depth=1 --branch {shlex.quote(branch)} {shlex.quote(repo_url)} {dest}")


# ── Server builders ──────────────────────────────────────────────────────────

def build_uvx(srv: dict) -> dict:
    """Build mcp-proxy entry for a PyPI package via uvx."""
    pkg = srv["package"]
    args: list[str] = srv.get("args") or []
    return {
        "command": "uvx",
        "args": [pkg, *[str(a) for a in args]],
        "env": resolve_env_dict(srv.get("env") or {}),
    }


def build_npx(srv: dict) -> dict:
    """Build mcp-proxy entry for an npm package via npx."""
    pkg = srv["package"]
    args: list[str] = srv.get("args") or []
    return {
        "command": "npx",
        "args": ["-y", pkg, *[str(a) for a in args]],
        "env": resolve_env_dict(srv.get("env") or {}),
    }


def build_github_python(srv: dict) -> dict:
    """Clone a Python MCP server from GitHub and install it."""
    name = srv["name"]
    repo = srv["repo"]
    branch = srv.get("branch", "main")
    install_cmd = srv.get("install", "uv pip install -e .")
    run_cmd = srv.get("run")  # e.g. "python -m mymodule" or "uv run mymodule"
    args: list[str] = srv.get("args") or []

    if not run_cmd:
        raise ValueError(f"Server '{name}' (github-python) requires a 'run' field.")

    dest = REPOS_DIR / name
    clone_or_update(repo, branch, dest)

    log.info("Installing Python dependencies for '%s' …", name)
    run(install_cmd, cwd=dest)

    # Build the command string that mcp-proxy will use.
    # run_cmd is split by shlex so e.g. "python -m foo" → ["python", "-m", "foo"]
    parts = shlex.split(run_cmd) + [str(a) for a in args]
    return {
        "command": parts[0],
        "args": parts[1:],
        "env": resolve_env_dict(srv.get("env") or {}),
    }


def build_github_node(srv: dict) -> dict:
    """Clone a Node.js MCP server from GitHub and install it."""
    name = srv["name"]
    repo = srv["repo"]
    branch = srv.get("branch", "main")
    install_cmd = srv.get("install", "npm install")
    run_cmd = srv.get("run")
    args: list[str] = srv.get("args") or []

    if not run_cmd:
        raise ValueError(f"Server '{name}' (github-node) requires a 'run' field.")

    dest = REPOS_DIR / name
    clone_or_update(repo, branch, dest)

    log.info("Installing Node dependencies for '%s' …", name)
    run(install_cmd, cwd=dest)

    parts = shlex.split(run_cmd) + [str(a) for a in args]
    return {
        "command": parts[0],
        "args": parts[1:],
        "env": resolve_env_dict(srv.get("env") or {}),
    }


def build_command(srv: dict) -> dict:
    """Arbitrary command entry."""
    cmd = srv.get("command")
    if not cmd:
        raise ValueError(f"Server '{srv['name']}' (command) requires a 'command' field.")
    args: list[str] = srv.get("args") or []
    return {
        "command": cmd,
        "args": [str(a) for a in args],
        "env": resolve_env_dict(srv.get("env") or {}),
    }


BUILDERS = {
    "uvx": build_uvx,
    "npx": build_npx,
    "github-python": build_github_python,
    "github-node": build_github_node,
    "command": build_command,
}


# ── Config reader ────────────────────────────────────────────────────────────

def load_config() -> dict[str, Any]:
    if not CONFIG_FILE.exists():
        log.error("Configuration file not found: %s", CONFIG_FILE)
        sys.exit(1)
    with CONFIG_FILE.open() as fh:
        return yaml.safe_load(fh)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    config = load_config()

    proxy_cfg: dict = config.get("proxy", {})
    port: int = int(proxy_cfg.get("port", 8080))
    host: str = proxy_cfg.get("host", "0.0.0.0")
    log_level: str = proxy_cfg.get("log_level", "INFO").upper()
    allow_origins: list[str] = proxy_cfg.get("allow_origins", [])
    pass_environment: bool = bool(proxy_cfg.get("pass_environment", False))

    servers_cfg: list[dict] = config.get("servers", [])
    if not servers_cfg:
        log.error("No servers defined in config/servers.yaml – nothing to do.")
        sys.exit(1)

    mcp_servers: dict[str, dict] = {}

    for srv in servers_cfg:
        name: str = srv.get("name", "")
        if not name:
            log.warning("Skipping server entry without a name.")
            continue

        enabled: bool = bool(srv.get("enabled", True))
        if not enabled:
            log.info("Server '%s' is disabled – skipping.", name)
            continue

        srv_type: str = srv.get("type", "").lower()
        builder = BUILDERS.get(srv_type)
        if builder is None:
            log.error(
                "Unknown server type '%s' for server '%s'. "
                "Valid types: %s",
                srv_type,
                name,
                ", ".join(BUILDERS),
            )
            sys.exit(1)

        log.info("Preparing server '%s' (type: %s) …", name, srv_type)
        try:
            entry = builder(srv)
        except Exception as exc:
            log.error("Failed to prepare server '%s': %s", name, exc)
            sys.exit(1)

        entry["transportType"] = "stdio"
        mcp_servers[name] = entry

    if not mcp_servers:
        log.error("No enabled servers remained after processing.")
        sys.exit(1)

    # Write generated config
    GENERATED_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    generated = {"mcpServers": mcp_servers}
    GENERATED_CONFIG.write_text(json.dumps(generated, indent=2))
    log.info("Generated mcp-proxy config → %s", GENERATED_CONFIG)
    log.info("Active servers: %s", ", ".join(mcp_servers))

    # Build mcp-proxy command
    cmd: list[str] = [
        "mcp-proxy",
        f"--port={port}",
        f"--host={host}",
        f"--log-level={log_level}",
        "--named-server-config", str(GENERATED_CONFIG),
    ]

    if pass_environment:
        cmd.append("--pass-environment")

    for origin in allow_origins:
        cmd.extend(["--allow-origin", origin])

    log.info("Starting mcp-proxy: %s", " ".join(shlex.quote(c) for c in cmd))
    log.info(
        "Home Assistant SSE endpoints:\n%s",
        "\n".join(
            f"  http://{host}:{port}/servers/{n}/sse"
            for n in mcp_servers
        ),
    )

    # Replace the current process with mcp-proxy (clean PID handling)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
