import asyncio
from utils import driver, get_embedding, oai_client


# tag::summarize_prompt[]
SUMMARIZE_PROMPT = """
You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or more entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we have the full context.
Limit the final description length to {max_length} words.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""
# end::summarize_prompt[]


FETCH_NODES_CYPHER = """
MATCH (n:`__Entity__`)
WHERE n.summary IS NULL
MATCH (n)<-[r:MENTIONS]-()
RETURN n.id AS entity, collect(r.description) AS descriptions
"""

STORE_NODE_SUMMARY_CYPHER = """
UNWIND $rows AS row
MATCH (e:`__Entity__` {id: row.name})
SET e.summary = row.summary, e.initial_summary_date = datetime()
WITH e, row
CALL db.create.setNodeVectorProperty(e, 'embedding', row.embedding)
WITH e
MATCH (e)<-[r:MENTIONS]-()
SET r.processed = True
"""


# tag::summarize_nodes[]
async def summarize_node(
    entity: str, descriptions: list[str], max_length: int = 200
) -> str:
    prompt = SUMMARIZE_PROMPT.format(
        max_length=max_length,
        entity_name=entity,
        description_list="\n- ".join(descriptions),
    )
    resp = await oai_client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an expert data extraction assistant. Follow formatting strictly.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


async def process_entity(
    entity: str,
    descriptions: list[str],
    sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        if len(descriptions) > 1:
            summary = await summarize_node(entity, descriptions)
        else:
            summary = descriptions[0]
        embedding = await get_embedding(summary)
        print(f"✓ {entity}")
        return {"name": entity, "summary": summary, "embedding": embedding}


# end::summarize_nodes[]


FETCH_RELS_CYPHER = """
MATCH (e1:__Entity__)-[r:RELATIONSHIP]-(e2:__Entity__)
WHERE elementId(e1) < elementId(e2) AND NOT (e1)-[:RELATIONSHIP_SUMMARY]->(e2)
WITH e1, e2, collect(r) AS rels
RETURN e1.id AS entity1, e2.id AS entity2, [r in rels | r.description] AS descriptions
"""

STORE_REL_SUMMARY_CYPHER = """
UNWIND $rows AS row
MATCH (e1:__Entity__ {id: row.entity1})
MATCH (e2:__Entity__ {id: row.entity2})
MERGE (e1)-[rs:RELATIONSHIP_SUMMARY]->(e2)
SET rs.summary = row.summary
WITH rs, row
CALL db.create.setRelationshipVectorProperty(rs, 'embedding', row.embedding)
"""


# tag::summarize_rels[]
async def summarize_rel(
    entity1: str,
    entity2: str,
    descriptions: list[str],
    max_length: int = 200,
) -> str:
    prompt = SUMMARIZE_PROMPT.format(
        max_length=max_length,
        entity_name=f"{entity1} <-> {entity2}",
        description_list="\n- ".join(descriptions),
    )
    resp = await oai_client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an expert data extraction assistant. Follow formatting strictly.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


async def process_relationship(
    entity1: str,
    entity2: str,
    descriptions: list[str],
    sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        if len(descriptions) > 1:
            summary = await summarize_rel(entity1, entity2, descriptions)
        else:
            summary = descriptions[0]
        embedding = await get_embedding(summary)
        print(f"✓ {entity1} <-> {entity2}")
        return {
            "entity1": entity1,
            "entity2": entity2,
            "summary": summary,
            "embedding": embedding,
        }


# end::summarize_rels[]


async def main():
    sem = asyncio.Semaphore(4)

    # --- Summarize + embed entities ---
    records, _, _ = driver.execute_query(FETCH_NODES_CYPHER)
    entities = [r.data() for r in records]
    print(f"Found {len(entities)} entities to process")

    if entities:
        tasks = [process_entity(e["entity"], e["descriptions"], sem) for e in entities]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        rows = [r for r in results if isinstance(r, dict)]
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            print(f"⚠ {len(errors)} entities failed:")
            for e in errors:
                print(f"  {e}")

        if rows:
            driver.execute_query(STORE_NODE_SUMMARY_CYPHER, rows=rows)
            print(f"Stored {len(rows)} node summaries + embeddings")

    # --- Summarize relationships ---
    records, _, _ = driver.execute_query(FETCH_RELS_CYPHER)
    rels = [r.data() for r in records]
    print(f"Found {len(rels)} relationships to summarize")

    if rels:
        tasks = [
            process_relationship(r["entity1"], r["entity2"], r["descriptions"], sem)
            for r in rels
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        rows = [r for r in results if isinstance(r, dict)]
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            print(f"⚠ {len(errors)} relationships failed:")
            for e in errors:
                print(f"  {e}")

        if rows:
            driver.execute_query(STORE_REL_SUMMARY_CYPHER, rows=rows)
            print(f"Stored {len(rows)} relationship summaries")

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
