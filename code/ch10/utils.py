import os
from openai import AsyncOpenAI

from neo4j import GraphDatabase

TUPLE_DELIMITER = ";"
RECORD_DELIMITER = "|"
COMPLETION_DELIMITER = "\n\n"

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

oai_client = AsyncOpenAI()


def parse_extraction_output(
    output_str: str,
    record_delimiter: str = RECORD_DELIMITER,
    tuple_delimiter: str = TUPLE_DELIMITER,
) -> tuple[list[dict], list[dict]]:
    # Strip the actual completion delimiter value, not the template placeholder
    output_str = output_str.strip().strip(COMPLETION_DELIMITER)

    raw_records = [r.strip() for r in output_str.split(record_delimiter)]
    entities, relationships = [], []

    for rec in raw_records:
        if not rec:
            continue
        # Strip outer parentheses
        if rec.startswith("(") and rec.endswith(")"):
            rec = rec[1:-1]
        rec = rec.strip()

        tokens = [t.strip().strip("\"'") for t in rec.split(tuple_delimiter)]
        if len(tokens) < 2:
            continue

        rec_type = tokens[0].lower()

        if rec_type == "entity" and len(tokens) >= 4:
            # Join overflow tokens back into description (handles semicolons in text)
            description = tuple_delimiter.join(tokens[3:]).strip().strip("\"'")
            entities.append(
                {
                    "entity_name": tokens[1],
                    "entity_type": tokens[2],
                    "entity_description": description,
                }
            )
        elif rec_type == "relationship" and len(tokens) >= 5:
            # Strength is always last; everything between index 3 and -1 is description
            try:
                strength = int(float(tokens[-1]))
            except (ValueError, TypeError):
                strength = 5
            description = tuple_delimiter.join(tokens[3:-1]).strip().strip("\"'")
            relationships.append(
                {
                    "source_entity": tokens[1],
                    "target_entity": tokens[2],
                    "relationship_description": description,
                    "relationship_strength": strength,
                }
            )

    return entities, relationships


async def get_embedding(
    text: str, model: str = "text-embedding-3-small"
) -> list[float]:
    resp = await oai_client.embeddings.create(input=[text], model=model)
    return resp.data[0].embedding
