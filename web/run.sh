#!/usr/bin/env bash
# run.sh — Start the Heston Calibration Engine web server
# Usage: bash web/run.sh [port]
#
# Run from the project root:
#   bash web/run.sh
#   bash web/run.sh 8080

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
PORT="${1:-8000}"

cd "$ROOT"

# Check binary exists
if [[ ! -f "build/heston_demo" ]]; then
  echo "ERROR: build/heston_demo not found. Build first:"
  echo "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release"
  echo "  cmake --build build -j"
  exit 1
fi

# Install deps if needed
if ! python3 -c "import fastapi" 2>/dev/null; then
  echo "Installing Python dependencies..."
  pip install -r web/requirements.txt -q
fi

echo "============================================="
echo "  Heston Calibration Engine — Web UI"
echo "============================================="
echo "  http://localhost:${PORT}"
echo "  Binary: build/heston_demo"
echo "  Press Ctrl-C to stop"
echo ""

python3 -m uvicorn web.app:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --reload \
  --reload-dir web
