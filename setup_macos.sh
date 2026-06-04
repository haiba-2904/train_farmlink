#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN="python3.12"
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="python3.11"
else
  echo "Khong tim thay python3.11 hoac python3.12." >&2
  echo "Tren macOS, nen cai bang Homebrew truoc: brew install python@3.12" >&2
  exit 1
fi

echo "Su dung interpreter: $PYTHON_BIN"

if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-macos.txt

echo
echo "Moi truong da san sang."
echo "Kich hoat lai bang: source .venv/bin/activate"
echo "Tien xu ly:        python src/preprocess.py"
echo "Chia dataset:      python src/splitter.py"
echo "Train model:       python src/train.py"
echo "Du doan anh:       python src/evaluate.py --image /duong/dan/anh.jpg"
