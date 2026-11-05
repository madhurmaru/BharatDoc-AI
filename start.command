#!/bin/zsh
set -eu
cd "$(dirname "$0")"
export PATH="/opt/homebrew/opt/python@3.11/libexec/bin:/opt/homebrew/bin:/usr/bin:/bin:$PATH"
export HF_HOME="$PWD/.cache/huggingface"
export XDG_CACHE_HOME="$PWD/.cache"
export TORCH_HOME="$PWD/.cache/torch"
export TMPDIR="$PWD/.cache/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NEXT_TELEMETRY_DISABLED=1
mkdir -p "$TMPDIR" "$TORCH_HOME"
if [[ ! -x backend/bin/vision-ocr ]]; then
  echo "OCR helper missing. Run ./backend/setup-vision.sh first."
  exit 1
fi
if [[ ! -d frontend/node_modules ]]; then
  echo "Frontend packages missing. Run npm install in the frontend folder first."
  exit 1
fi
if [[ ! -f frontend/.next/BUILD_ID ]]; then
  (cd frontend && npm run build)
fi
python3 -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 &
backend_pid=$!
trap 'kill "$backend_pid" 2>/dev/null || true' EXIT INT TERM
(cd frontend && npm run start) &
frontend_pid=$!
trap 'kill "$backend_pid" "$frontend_pid" 2>/dev/null || true' EXIT INT TERM
sleep 3
open http://127.0.0.1:3000
echo "BharatDoc AI is running locally. Keep this window open. Press Ctrl+C to stop."
wait "$backend_pid" "$frontend_pid"
