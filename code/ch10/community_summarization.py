import asyncio
import json
from openai import AsyncOpenAI
from utils import driver, get_embedding, oai_client


# tag::community_report_prompt[]
COMMUNITY_REPORT_PROMPT = """
You are an AI assistant that helps a human analyst to perform general
information discovery. Information discovery is the process of identifying
and assessing relevant information associated with certain entities (e.g.,
organizations and individuals) within a network.

# Goal
Write a comprehensive report of a community, given a list of entities that
belong to the community as well as their relationships and optional
associated claims. The report will be used to inform decision-makers about
information associated with the community and their potential impact. The
content of this report includes an overview of the community's key
entities, their legal compliance, technical capabilities, reputation, and
noteworthy claims.

# Report Structure

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should
  be short but specific. When possible, include representative named
  entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how
  its entities are related to each other, and significant information
  associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the
  severity of IMPACT posed by entities within the community. IMPACT is the
  scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT
  severity rating.

Return output as a well-formed JSON-formatted string with the following format:
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>
    }}

# Grounding Rules

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references \
[Data: <dataset name> (record ids); <dataset name> (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list \
the top 5 most relevant record ids and add "+more" to indicate that there \
are more.

For example:
"Person X is the owner of Company Y and subject to many allegations of \
wrongdoing [Data: Reports (1), Entities (5, 7); Relationships (23); \
Claims (7, 2, 34, 64, 46, +more)]."

where 1, 5, 7, 23, 2, 34, 46, and 64 represent the id (not the index) of the
relevant data record.

Do not include information where the supporting evidence for it is not provided.

Limit the total report length to {max_report_length} words.

# Example Input
-----------
Text:

Entities

human_readable_id,title,description
5,VERDANT OASIS PLAZA,Verdant Oasis Plaza is the location of the Unity March
6,HARMONY ASSEMBLY,Harmony Assembly is holding a march at Verdant Oasis Plaza

Relationships

human_readable_id,source,target,description
37,VERDANT OASIS PLAZA,UNITY MARCH,Location of the Unity March
38,VERDANT OASIS PLAZA,HARMONY ASSEMBLY,Holding a march at Verdant Oasis Plaza
39,VERDANT OASIS PLAZA,UNITY MARCH,Taking place at Verdant Oasis Plaza
40,VERDANT OASIS PLAZA,TRIBUNE SPOTLIGHT,Reporting on the march
41,VERDANT OASIS PLAZA,BAILEY ASADI,Speaking about the march
43,HARMONY ASSEMBLY,UNITY MARCH,Harmony Assembly is organizing the Unity March

Output:
{{
    "title": "Verdant Oasis Plaza and Unity March",
    "summary": "The community revolves around the Verdant Oasis Plaza, which \
is the location of the Unity March. The plaza has relationships with the \
Harmony Assembly, Unity March, and Tribune Spotlight, all of which are \
associated with the march event.",
    "rating": 5.0,
    "rating_explanation": "The impact severity rating is moderate due to \
the potential for unrest or conflict during the Unity March."
}}

# Real Data

Use the following text for your answer. Do not make anything up in your answer.

Text:
{input_text}

Output:
"""
# end::community_report_prompt[]

STORE_COMMUNITY_SUMMARY_CYPHER = """
UNWIND $rows AS row
MATCH (c:__Community__ {id: row.community_id})
SET c.summary = row.summary,
    c.title = row.title,
    c.rating = row.rating,
    c.rating_explanation = row.rating_explanation
WITH c, row
CALL db.create.setNodeVectorProperty(c, 'embedding', row.embedding)
"""


# tag::summarization[]

FETCH_COMMUNITIES_CYPHER = """
MATCH (c:__Community__)
WITH max(c.level) AS max_level
MATCH (c:__Community__)<-[:IN_COMMUNITY*]-(e:__Entity__)
WHERE c.level = max_level
WITH c, collect(e) AS community_nodes
WHERE size(community_nodes) > 3
CALL (community_nodes) {
  MATCH (n1)-[r:RELATIONSHIP]->(n2)
  WHERE n1 IN community_nodes AND n2 IN community_nodes
  RETURN collect(r.description) AS rel_descriptions
}
RETURN c.id AS community_id,
       [node in community_nodes | node.summary] AS node_summaries,
       rel_descriptions
"""


def build_input_text(
    node_summaries: list[str], rel_descriptions: list[str]
) -> str:
    entities_section = "Entities\n\n"
    for i, summary in enumerate(node_summaries):
        if summary:
            entities_section += f"{i}: {summary}\n"

    rels_section = "\nRelationships\n\n"
    for i, desc in enumerate(rel_descriptions):
        if desc:
            rels_section += f"{i}: {desc}\n"

    return entities_section + rels_section


async def summarize_community(
    node_summaries: list[str],
    rel_descriptions: list[str],
    max_report_length: int = 1500,
) -> dict:
    input_text = build_input_text(node_summaries, rel_descriptions)
    prompt = COMMUNITY_REPORT_PROMPT.format(
        input_text=input_text,
        max_report_length=max_report_length,
    )
    resp = await oai_client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI assistant that helps a human analyst "
                    "perform general information discovery. Respond only "
                    "with valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]
    return json.loads(raw)


async def process_community(
    community_id: str,
    node_summaries: list[str],
    rel_descriptions: list[str],
    sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        report = await summarize_community(node_summaries, rel_descriptions)
        summary_text = report.get("summary", "")
        embedding = await get_embedding(summary_text)
        title = report.get('title', 'Untitled')
        print(f"✓ Community {community_id}: {title}")
        return {
            "community_id": community_id,
            "title": report.get("title", ""),
            "summary": summary_text,
            "rating": report.get("rating", 0.0),
            "rating_explanation": report.get("rating_explanation", ""),
            "embedding": embedding,
        }


# end::summarization[]


async def main():
    records, _, _ = driver.execute_query(FETCH_COMMUNITIES_CYPHER)
    communities = [r.data() for r in records]
    print(f"Found {len(communities)} communities to summarize")

    if not communities:
        return

    sem = asyncio.Semaphore(4)

    tasks = [
        process_community(
            c["community_id"],
            c["node_summaries"],
            c["rel_descriptions"],
            sem,
        )
        for c in communities
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    rows = [r for r in results if isinstance(r, dict)]
    errors = [r for r in results if isinstance(r, BaseException)]
    if errors:
        print(f"⚠ {len(errors)} communities failed:")
        for e in errors:
            print(f"  {e}")

    if rows:
        driver.execute_query(STORE_COMMUNITY_SUMMARY_CYPHER, rows=rows)
        print(f"Stored {len(rows)} community summaries")

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
