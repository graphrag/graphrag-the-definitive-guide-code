#!/usr/bin/env bash
set -euo pipefail

# Loads the Chapter 8 consultant graph the agent benchmark scores against.
cd "$(dirname "$0")/../ch08"
bash run_import.sh
