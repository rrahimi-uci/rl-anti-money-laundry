#!/usr/bin/env bash
set -euo pipefail

echo "Stopping RL Training Dashboard..."
# Kill backend
lsof -ti:8200 | xargs kill -9 2>/dev/null || true
# Kill frontend
lsof -ti:4003 | xargs kill -9 2>/dev/null || true
echo "Done."
