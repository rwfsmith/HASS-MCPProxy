#!/usr/bin/env bash
set -e

# Read options from HA Supervisor (bashio is available in hassio-addons/base)
export LOG_LEVEL=$(bashio::config 'log_level' 'info')
export ALLOW_ALL_ORIGINS=$(bashio::config 'allow_all_origins' 'true')
export PASS_ENVIRONMENT=$(bashio::config 'pass_environment' 'false')

bashio::log.info "Starting HASS-MCPProxy management API..."
exec python3 /app/main.py
