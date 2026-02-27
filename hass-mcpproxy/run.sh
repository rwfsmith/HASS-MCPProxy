#!/bin/bash
# HA Supervisor writes add-on options to /data/options.json.
# main.py reads that file directly; no bashio dependency needed.
exec python3 /app/main.py
