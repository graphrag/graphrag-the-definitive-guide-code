import asyncio
from utils import driver, get_embedding, oai_client


# tag::incremental_update_prompt[]
INCREMENTAL_UPDATE_PROMPT = """
You are a helpful assistant responsible for updating an existing summary with newly discovered information.
Given one or more entities, an existing summary, and a list of new descriptions — all related to the same entity or group of entities.
Please produce an updated summary that integrates the new descriptions into the existing one.
Make sure to include information collected from all new descriptions without dropping facts already present in the existing summary.
If new descriptions contradict the existing summary, resolve the contradictions by preferring the newer information.
If the existing summary is empty, generate a fresh summary from the new descriptions alone.
Do not repeat information that is already adequately captured in the existing summary.
Make sure it is written in third person, and include the entity names so we have the full context.

#######
-Data-
Entities: {entity_name}
Existing Summary: {existing_summary}
New Description List: {description_list}
#######
Output:
"""
# end::incremental_update_prompt[]


FETCH_NODES_CYPHER = """
MATCH (n:`__Entity__`)
WHERE n.summary IS NOT NULL
  AND count{(n)<-[:MENTIONS {processed: false}]-()} > 1
MATCH (n)<-[r:MENTIONS {processed: false}]-()
RETURN n.id AS entity, n.summary AS existing_summary, collect(r.description) AS descriptions
"""

STORE_NODE_SUMMARY_CYPHER = """
UNWIND $rows AS row
MATCH (e:`__Entity__` {id: row.name})
SET e.summary = row.summary, e.last_update_date = datetime()
WITH e, row
CALL db.create.setNodeVectorProperty(e, 'embedding', row.embedding)
WITH e
MATCH (e)<-[r:MENTIONS {processed: false}]-()
SET r.processed = true
"""


# tag::incremental_update_nodes[]
async def incremental_summarize_node(
    entity: str,
    existing_summary: str,
    descriptions: list[str],
) -> str:
    prompt = INCREMENTAL_UPDATE_PROMPT.format(
        entity_name=entity,
        existing_summary=existing_summary,
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
    existing_summary: str,
    descriptions: list[str],
    sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        summary = await incremental_summarize_node(
            entity, existing_summary, descriptions
        )
        embedding = await get_embedding(summary)
        print(f"✓ {entity}")
        return {"name": entity, "summary": summary, "embedding": embedding}


# end::incremental_update_nodes[]


FETCH_RELS_CYPHER = """
MATCH (e1:__Entity__)-[rs:RELATIONSHIP_SUMMARY]->(e2:__Entity__)
WHERE count{(e1)-[r:RELATIONSHIP {processed: false}]-(e2)} > 1
MATCH (e1)-[r:RELATIONSHIP {processed: false}]-(e2)
RETURN e1.id AS entity1, e2.id AS entity2, rs.summary AS existing_summary, collect(r.description) AS descriptions
"""

STORE_REL_SUMMARY_CYPHER = """
UNWIND $rows AS row
MATCH (e1:__Entity__ {id: row.entity1})-[rs:RELATIONSHIP_SUMMARY]->(e2:__Entity__ {id: row.entity2})
SET rs.summary = row.summary, rs.last_update_date = datetime()
WITH rs, row
CALL db.create.setRelationshipVectorProperty(rs, 'embedding', row.embedding)
WITH row
MATCH (e1:__Entity__ {id: row.entity1})-[r:RELATIONSHIP {processed: false}]-(e2:__Entity__ {id: row.entity2})
SET r.processed = true
"""


# tag::incremental_update_rels[]
async def incremental_summarize_rel(
    entity1: str,
    entity2: str,
    existing_summary: str,
    descriptions: list[str],
) -> str:
    prompt = INCREMENTAL_UPDATE_PROMPT.format(
        entity_name=f"{entity1} <-> {entity2}",
        existing_summary=existing_summary,
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
    existing_summary: str,
    descriptions: list[str],
    sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        summary = await incremental_summarize_rel(
            entity1, entity2, existing_summary, descriptions
        )
        embedding = await get_embedding(summary)
        print(f"✓ {entity1} <-> {entity2}")
        return {
            "entity1": entity1,
            "entity2": entity2,
            "summary": summary,
            "embedding": embedding,
        }


# end::incremental_update_rels[]


async def main():
    sem = asyncio.Semaphore(4)

    # --- Incremental update nodes ---
    records, _, _ = driver.execute_query(FETCH_NODES_CYPHER)
    entities = [r.data() for r in records]
    print(f"Found {len(entities)} entities to incrementally update")

    if entities:
        tasks = [
            process_entity(e["entity"], e["existing_summary"], e["descriptions"], sem)
            for e in entities
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        rows = [r for r in results if isinstance(r, dict)]
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            print(f"⚠ {len(errors)} entities failed:")
            for e in errors:
                print(f"  {e}")

        if rows:
            driver.execute_query(STORE_NODE_SUMMARY_CYPHER, rows=rows)
            print(f"Updated {len(rows)} node summaries")

    # --- Incremental update relationships ---
    records, _, _ = driver.execute_query(FETCH_RELS_CYPHER)
    rels = [r.data() for r in records]
    print(f"Found {len(rels)} relationships to incrementally update")

    if rels:
        tasks = [
            process_relationship(
                r["entity1"],
                r["entity2"],
                r["existing_summary"],
                r["descriptions"],
                sem,
            )
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
            print(f"Updated {len(rows)} relationship summaries")

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
