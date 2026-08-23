#!/usr/bin/env bash
set -euo pipefail

# Test runner for agentic-ai-lab
TARGET="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

if [[ "$TARGET" == "unit" || "$TARGET" == "all" ]]; then
    echo "Running unit test suite..."
    pytest tests/unit -v
fi

if [[ "$TARGET" == "parity" || "$TARGET" == "all" ]]; then
    echo "Checking parity between v2-ebook and agentic-ai-lab..."
    python3 scripts/check_book_parity.py --ebook ../v2-ebook --lab .
fi

echo "All test targets completed successfully."
