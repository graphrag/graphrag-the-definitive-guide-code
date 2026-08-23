#!/usr/bin/env bash
set -euo pipefail

# Agent benchmark. Needs the ch08 graph loaded and `uvx` on PATH.
PYTHONUNBUFFERED=1 uv run --env-file graphrag-book.env \
  --with openai --with neo4j --with python-dotenv --with pandas --with pydantic --with mcp \
  agent_eval.py

# Construction benchmark. OpenAI only, no Neo4j data required.
PYTHONUNBUFFERED=1 uv run --env-file graphrag-book.env \
  --with openai --with neo4j --with python-dotenv --with pandas --with pydantic \
  kg_construction_eval.py
