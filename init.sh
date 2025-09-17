#!/bin/bash

# Check if "venv" folder exists; create it if not
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✅ Virtual environment created."
fi

# Activate the virtual environment
# source venv/bin/activate
source .venv/Scripts/activate  # For Windows compatibility
echo "🔹 Virtual environment activated."

# Upgrade pip
python -m pip install --upgrade pip

# Install Poetry inside the venv
pip install poetry

# Install project dependencies from pyproject.toml
poetry install

echo "✅ Setup complete. To activate venv in the future, run: source .venv/Scripts/activate"

