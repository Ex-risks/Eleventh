#!/bin/bash
# ELEVENTH launcher — double-click this to start.

cd "$(dirname "$0")"
APP_DIR="$(pwd)"

export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"

# Install uv if missing.
if ! command -v uv &>/dev/null; then
  echo "[setup] installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Sync Python environment (fast no-op after first run).
uv sync --quiet

# Kill any previous engine on these ports.
lsof -ti :7878 | xargs kill -9 2>/dev/null || true
lsof -ti :7979 | xargs kill -9 2>/dev/null || true
sleep 0.3

# Start the engine detached from this terminal so closing the window doesn't kill it.
nohup uv run python engine/server.py > /tmp/eleventh-engine.log 2>&1 &
ENGINE_PID=$!
echo "[ELEVENTH] engine started (pid $ENGINE_PID)"

# Wait up to 10s for the server to be ready.
for i in $(seq 1 25); do
  if curl -sf http://localhost:7878 >/dev/null 2>&1; then
    echo "[ELEVENTH] server ready"
    break
  fi
  sleep 0.4
done

# Open the browser.
URL="http://localhost:7878"
if open -a "Google Chrome" "$URL" 2>/dev/null; then
  :
elif open -a "Safari" "$URL" 2>/dev/null; then
  :
else
  open "$URL"
fi

echo "[ELEVENTH] running. Engine log: /tmp/eleventh-engine.log"
echo "To stop: kill $ENGINE_PID"
# This terminal can now be closed — engine keeps running.
