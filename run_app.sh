#!/usr/bin/env bash
# ===================================================================
#  Smart Waste Sorter - launcher for macOS and Linux
#  1st run: creates a virtual environment and installs the libraries
#  (needs internet, about 1-3 minutes). Later runs start immediately.
# ===================================================================
set -e
cd "$(dirname "$0")"
PY=python3
command -v $PY >/dev/null 2>&1 || PY=python
if ! command -v $PY >/dev/null 2>&1; then
  echo "Python 3.10 or newer is required: https://www.python.org/downloads/"; exit 1
fi
if [ ! -x ".venv/bin/python" ]; then
  echo "Creating virtual environment..."
  $PY -m venv .venv
fi
if [ ! -f ".venv/installed.ok" ]; then
  echo "Installing the required libraries (first run only)..."
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
  touch .venv/installed.ok
fi
echo
echo "Starting Smart Waste Sorter... your browser will open at http://localhost:8501"
echo "(If it does not, copy the 'Local URL' shown below into your browser.) Press Ctrl+C to stop."
( sleep 5; (command -v open >/dev/null && open http://localhost:8501) || (command -v xdg-open >/dev/null && xdg-open http://localhost:8501) || true ) &
.venv/bin/python -m streamlit run app/streamlit_app.py
