#!/bin/sh
set -eu

APP_ROOT=/root/diansai2026
APP_PROCESS='[/]root/diansai2026/.venv/bin/python3 viz/recognition_app.py --fullscreen'

if /usr/bin/pgrep -f "$APP_PROCESS" >/dev/null; then
    exit 0
fi

cd "$APP_ROOT"
exec /root/.local/bin/uv run --no-sync viz/recognition_app.py --fullscreen
