import json, os
import openai
from neo4j import GraphDatabase
from neo4j.time import Date, DateTime, Duration

# ── Neo4j connection ─────────────────────────────────────────────────────────
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))


# ── Helpers ──────────────────────────────────────────────────────────────────
def get_openai_embedding(text: str, model: str = "text-embedding-3-small") -> list:
    """Return an OpenAI embedding vector for *text*."""
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def _neo4j_default(obj):
    """JSON serializer for Neo4j temporal types."""
    if isinstance(obj, (Date, DateTime)):
        return obj.iso_format()
    if isinstance(obj, Duration):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def records_to_json(records) -> str:
    """Convert Neo4j records to a JSON string, handling dates and dropping embeddings."""
    rows = []
    for record in records:
        row = record.data()
        for v in row.values():
            if isinstance(v, dict):
                v.pop("embedding", None)
        rows.append(row)
    return json.dumps(rows, default=_neo4j_default)
