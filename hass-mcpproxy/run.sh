#!/usr/bin/with-contenv bashio
# HA Supervisor writes add-on options to /data/options.json.
# main.py reads that file directly, so no bashio calls needed here.
bashio::log.info "Starting HASS-MCPProxy management API..."
exec python3 /app/main.py
