#!/usr/bin/env bash
# run_setup.sh -- one-shot bootstrap for chapter 9.
#
# Assumes chapter 8's data has already been imported. This script:
#   1. Installs chapter 9 requirements (adds neo4j-agent-memory >=0.2.0).
#   2. Runs the OMG graph adoption (adds :Entity / type / id properties
#      to existing OMG nodes via client.schema.adopt_existing_graph).
#   3. The library creates its own constraints and indexes on first
#      MemoryClient connection (handled inside app_agent_memory.py).
set -euo pipefail

uv pip install -r requirements.txt

uv run python adopt_omg_graph.py

echo "Adoption complete. Launch the app with:"
echo "    streamlit run app_agent_memory.py"
