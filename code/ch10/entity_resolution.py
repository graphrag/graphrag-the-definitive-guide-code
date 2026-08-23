import json
import asyncio
from openai import AsyncOpenAI
from utils import driver, oai_client

# tag::entity_resolution_prompt[]
ENTITY_RESOLUTION_PROMPT = """You are an expert entity resolution system. Given a list of entity names of type "{label}", identify groups of names that refer to the EXACT same real-world entity, just written differently.

ONLY merge when names are clearly alternate forms of the identical entity, such as:
- Abbreviations: "EUROPEAN-UNION" and "EU"
- Spelling variations: "ACME-CORPORATION" and "ACME-CORP"
- Acronyms: "INTERNATIONAL-BUSINESS-MACHINES" and "IBM"

DO NOT merge:
- Entities that are subsets or parts of each other (e.g. "US-DEPARTMENT-OF-DEFENCE" and "PENTAGON" are related but distinct)
- Entities that overlap geographically or conceptually but are not the same thing (e.g. "MIDDLE-EAST" and "WEST-ASIA")
- Parent orgs and subsidiaries (e.g. "ALPHABET" and "GOOGLE")
- An organization/institution and the place it is named after or located in (e.g. "UNIVERSITY-OF-WARWICK" and "WARWICK")
- Entities where the equivalence is debatable or context-dependent

It is FAR better to miss a true duplicate than to incorrectly merge two distinct entities. When in doubt, do NOT merge.

Return a JSON list of lists. Each inner list should contain all names that refer to the same entity, with the canonical (most complete/formal) name as the FIRST element. Only include groups where there are actual duplicates (2+ names). If no duplicates exist, return an empty list.

Example output:
[
  ["MICROSOFT-CORP", "MICROSOFT", "MSFT"],
  ["EUROPEAN-UNION", "EU"]
]

Entity names:
{entities}

Output ONLY valid JSON, no explanation."""
# end::entity_resolution_prompt[]


# tag::resolution[]
async def resolve_entities_for_label(
    sem: asyncio.Semaphore, label: str, entities: list[str]
) -> list[list[str]]:
    async with sem:
        if len(entities) < 2:
            return []

    prompt = ENTITY_RESOLUTION_PROMPT.format(
        label=label,
        entities=json.dumps(entities),
    )

    response = await oai_client.responses.create(
        model="gpt-5.4-mini",
        input=[{"role": "user", "content": prompt}],
    )
    text = response.output_text.strip()
    # Strip markdown fences if present
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    result = json.loads(text)
    return result


# end::resolution[]


MAX_CONCURRENCY = 4


async def main():
    records, _, _ = driver.execute_query(
        """
        MATCH (n:`__Entity__`)
        RETURN [x in labels(n) WHERE x <> '__Entity__'][0] AS label,
               collect(n.id) AS entities
        """
    )

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [
        resolve_entities_for_label(sem, record["label"], record["entities"])
        for record in records
    ]
    results = await asyncio.gather(*tasks)

    all_merges = []
    for record, merges in zip(records, results):
        label = record["label"]
        if merges:
            for group in merges:
                print(f"  {label}: {group[0]} <- {group[1:]}")
            all_merges.extend(merges)

    print(f"\nTotal merge groups: {len(all_merges)}")
    print(json.dumps(all_merges, indent=2))

    # Merge nodes in Neo4j
    if all_merges:
        print("\nMerging nodes in Neo4j...")
        driver.execute_query(
            # tag::cypher[]
            """
            UNWIND $groups AS group
            MATCH (e:__Entity__) WHERE e.id IN group
            WITH group, collect(e) AS nodes
            // Reorder so the canonical node (first in group) is first in the list
            WITH [n IN nodes WHERE n.id = group[0]] + [n IN nodes WHERE n.id <> group[0]] AS ordered
            CALL apoc.refactor.mergeNodes(ordered, {properties: "discard", mergeRels: false})
            YIELD node
            RETURN count(node) AS merged
            """,
            # end::cypher[]
            groups=all_merges,
        )
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
