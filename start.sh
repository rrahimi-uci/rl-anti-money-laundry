#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "=== RL Training Dashboard ==="

# ── Backend ──────────────────────────────────────────────────────────────
echo "[1/3] Checking Python venv..."
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt

echo "[2/3] Starting backend on :8200..."
uvicorn server:app --host 0.0.0.0 --port 8200 --reload &
BACKEND_PID=$!

# ── Frontend ─────────────────────────────────────────────────────────────
echo "[3/3] Starting frontend on :4003..."
cd ui
if [ ! -d "node_modules" ]; then
  npm install
fi
npm run dev &
FRONTEND_PID=$!

cd "$DIR"

echo ""
echo "  Backend:  http://localhost:8200"
echo "  Frontend: http://localhost:4003"
echo ""
echo "Press Ctrl-C to stop both."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM
wait
