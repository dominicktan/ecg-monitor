#!/bin/bash
set -e

# Download MIT-BIH data if not already present
if [ ! -f "mit-bih-arrhythmia-database-1.0.0/100.dat" ]; then
    echo "Downloading MIT-BIH Arrhythmia Database..."
    uv run --no-dev python scripts/download_data.py
fi

# Run the provided command
exec "$@"
