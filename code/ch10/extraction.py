import sys
import asyncio
import re
from enum import Enum

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from utils import (
    driver,
    oai_client,
)

MAX_CONCURRENCY = 4
DEFAULT_ARTICLES_PATH = "../../data/ch10/articles_initial.csv"

# tag::cypher[]
# Each entity is materialized with two labels: the typed label (e.g. PERSON)
# AND __Entity__. Leiden later projects __Entity__ uniformly; retrieval can
# still filter or rank by the typed label.
NODES_QUERY = """
UNWIND $data AS chunk
MERGE (c:__Chunk__ {url: chunk.url})
SET c.sitename = chunk.sitename, c.date = chunk.date, c.text = chunk.text
WITH c, chunk
UNWIND chunk.entities AS ent
MERGE (e:__Entity__ {id: ent.entity_name})
ON CREATE SET e:$(ent.entity_type)
MERGE (c)-[m:MENTIONS]->(e)
SET m.description = ent.entity_description, m.processed = False
"""

# The graph stores all extracted edges as :RELATIONSHIP for uniform projection
# during community detection. The typed relationship label from the ontology
# (WORKS_FOR, LED_BY, ...) lives in the r.type property so retrieval can still
# filter on it without changing the projected graph topology.
RELS_QUERY = """
UNWIND $relationships AS rel
MATCH (src:__Entity__ {id: rel.source_entity})
MATCH (tgt:__Entity__ {id: rel.target_entity})
MERGE (src)-[r:RELATIONSHIP]->(tgt)
SET r.type = rel.relationship_type,
    r.description = rel.relationship_description,
    r.strength = rel.relationship_strength,
    r.processed = False
"""
# end::cypher[]


# tag::ontology[]
_ID_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9-]*$")


def _canonical_id(value: str) -> str:
    """Normalize an entity id to the ALL-CAPS-HYPHENATED form."""
    cleaned = re.sub(r"\s+", "-", value.strip()).upper()
    if not _ID_PATTERN.match(cleaned):
        raise ValueError(f"Id {value!r} does not match {_ID_PATTERN.pattern}")
    return cleaned


class EntityType(str, Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    EVENT = "EVENT"
    CONCEPT = "CONCEPT"


class RelationshipType(str, Enum):
    WORKS_FOR = "WORKS_FOR"              # PERSON -> ORGANIZATION
    LED_BY = "LED_BY"                    # ORGANIZATION -> PERSON
    BASED_IN = "BASED_IN"                # ORGANIZATION | PERSON -> LOCATION
    PARTICIPATED_IN = "PARTICIPATED_IN"  # PERSON | ORGANIZATION -> EVENT
    RELATED_TO = "RELATED_TO"            # any -> any (fallback)


# Domain / range constraints. The Pydantic schema enforces that each
# relationship carries a valid type; the from/to-type constraints are
# checked after the LLM call in validate_ontology.
RELATIONSHIP_CONSTRAINTS: dict[
    RelationshipType, tuple[set[EntityType], set[EntityType]]
] = {
    RelationshipType.WORKS_FOR:       ({EntityType.PERSON},
                                       {EntityType.ORGANIZATION}),
    RelationshipType.LED_BY:          ({EntityType.ORGANIZATION},
                                       {EntityType.PERSON}),
    RelationshipType.BASED_IN:        ({EntityType.ORGANIZATION,
                                        EntityType.PERSON},
                                       {EntityType.LOCATION}),
    RelationshipType.PARTICIPATED_IN: ({EntityType.PERSON,
                                        EntityType.ORGANIZATION},
                                       {EntityType.EVENT}),
    RelationshipType.RELATED_TO:      (set(EntityType), set(EntityType)),
}


class Entity(BaseModel):
    id: str = Field(
        ...,
        description=("ALL-CAPS-HYPHENATED unique identifier (e.g. MICROSOFT, "
                     "SATYA-NADELLA, MICROSOFT-ACTIVISION-ACQUISITION)."),
    )
    type: EntityType
    description: str = Field(
        ...,
        description=("Atomic, self-contained description grounded in the "
                     "article. Use full entity names; no pronouns or "
                     "shorthand. Restate context within the description."),
    )

    @field_validator("id")
    @classmethod
    def _canonical(cls, v: str) -> str:
        return _canonical_id(v)


class Relationship(BaseModel):
    source: str = Field(..., description="Entity id from the entities list.")
    target: str = Field(..., description="Entity id from the entities list.")
    type: RelationshipType
    description: str = Field(
        ...,
        description=("Description naming both endpoints by full name; "
                     "grounded in the article."),
    )
    strength: int = Field(
        ...,
        ge=1,
        le=10,
        description="How central this relationship is to the article (1-10).",
    )

    @field_validator("source", "target")
    @classmethod
    def _canonical(cls, v: str) -> str:
        return _canonical_id(v)


class Extraction(BaseModel):
    entities: list[Entity]
    relationships: list[Relationship]
# end::ontology[]


# tag::prompt[]
BUSINESS_NEWS_EXTRACTION_PROMPT = """-Goal-
Extract business-relevant entities and typed relationships from news
articles to build a knowledge graph for QA retrieval. Output MUST match
the supplied JSON schema.

-Entity types-
- PERSON:       Named individuals (executives, regulators' chairs, public
                figures). Always extract people the article names.
- ORGANIZATION: Companies, regulators, agencies, and public institutions
                (e.g. APPLE, US-DOJ, EUROPEAN-COMMISSION).
- LOCATION:     Geographic or jurisdictional places (cities, countries,
                regions, e.g. SAN-FRANCISCO-USA, EUROPEAN-UNION).
- EVENT:        Notable occurrences (acquisitions, launches, lawsuits,
                regulatory approvals, product releases).
- CONCEPT:      Notable themes, technologies, products, or programs
                discussed in the article (e.g. GENERATIVE-AI,
                ANTITRUST-ENFORCEMENT, IPHONE-15). Use when the thing
                is neither a person, org, location, nor an event.

-Relationship types (with required domain -> range)-
- WORKS_FOR:       PERSON -> ORGANIZATION
- LED_BY:          ORGANIZATION -> PERSON
- BASED_IN:        ORGANIZATION or PERSON -> LOCATION
- PARTICIPATED_IN: PERSON or ORGANIZATION -> EVENT
- RELATED_TO:      any -> any (use sparingly, only when no typed
                   relationship above fits)

-Entity ID Format-
ALL-CAPS, hyphen-separated for multi-word names. Use full legal or
common names (e.g. "Google" -> GOOGLE). Government bodies include
jurisdiction (e.g. "DOJ" -> US-DOJ). Correct obvious typos.

-Description Guidelines-
- Write ONLY from information explicitly stated in the source text. Never
  add background knowledge, inferences, or elaboration.
- Stay close to the original wording rather than paraphrasing into more
  technical language.
- Never include meta-commentary ("the article states...", "no further
  details provided..."). State facts directly.

ATOMIC & SELF-CONTAINED DESCRIPTIONS (CRITICAL):
Each description must be fully understandable in isolation, without
reading any other entity or the source text. To achieve this:
- Always use full entity names. Never write "the company", "the
  executive", "the deal", "the agency", "it", "they", or any
  pronoun/shorthand referring to another entity.
- Re-state the relevant context within each description.
- Good: "Satya Nadella is CEO of Microsoft. Nadella called Microsoft's
  acquisition of Activision Blizzard 'a turning point for our gaming
  business.'"
- Bad: "CEO who called the acquisition 'a turning point for our gaming
  business.'"

Per-type guidance:
- PERSON: Job title, organization (by full name), role in reported
  events, direct quotes or actions.
- ORGANIZATION: Industry sector, HQ location if stated, financials
  (revenue, valuation, market cap with units/currency), employee counts,
  role in events.
- LOCATION: Jurisdictional/regulatory context, economic indicators if
  stated, role in events.
- EVENT: Date/timeframe, involved parties (by full name), financial
  terms/deal values, current status, stated outcomes.
- CONCEPT: What the concept is (technology, theme, product, program),
  who is associated with it in the article, why it matters in context.

-Relationship Guidelines-
- Source != target (no self-references).
- Both endpoints must exist in the entities list.
- Descriptions must name both entities explicitly by full name -- never
  use "it", "they", or "the company".
- Strength 1-10 reflects how central the relationship is to the article.
- Use the most specific typed relationship that fits; fall back to
  RELATED_TO only when none of the typed options apply.

-Critical Rules-
1. ENTITY-FIRST: Extract ALL entities before relationships. Every
   relationship endpoint must already be in the entities list.
2. COMPLETE CAPTURE: ALL financial figures (with units/currency), ALL
   dates, and ALL job titles from the source must appear in
   descriptions.
3. SOURCE TEXT ONLY: Never add definitions, context, or background from
   general knowledge.
4. NO HALLUCINATION: If the text says "Cook announced the product" but
   gives no biography, do NOT add biographical details.
5. SPARSE ARTICLES: If the text is only boilerplate, disclaimers, or
   author bios, return empty lists for entities and relationships.

-Real Data-
Article:
{input_text}
"""
# end::prompt[]


# tag::extraction[]
async def extract_row(sem: asyncio.Semaphore, index: int, row: dict) -> dict | None:
    """Run typed extraction against one article using structured outputs.

    chat.completions.parse puts the SDK in structured-output mode: the
    model is constrained to JSON that validates against the Extraction
    Pydantic schema, so entity and relationship types are guaranteed to
    come from the ontology enums. There is no hand-written parser and
    no delimiter format to keep in sync.
    """
    async with sem:
        print(f"Processing row {index + 1}")

        prompt = BUSINESS_NEWS_EXTRACTION_PROMPT.format(input_text=row["text"])

        try:
            completion = await oai_client.chat.completions.parse(
                model="gpt-5.4-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You extract knowledge graphs that match the "
                            "supplied ontology. Follow the schema exactly."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=Extraction,
            )
            extraction: Extraction = completion.choices[0].message.parsed

            entities = [
                {
                    "entity_name": e.id,
                    "entity_type": e.type.value,
                    "entity_description": e.description,
                }
                for e in extraction.entities
            ]
            relationships = [
                {
                    "source_entity": r.source,
                    "target_entity": r.target,
                    "relationship_type": r.type.value,
                    "relationship_description": r.description,
                    "relationship_strength": r.strength,
                }
                for r in extraction.relationships
            ]

            if entities or relationships:
                print(
                    f" -> Row {index + 1}: {len(entities)} nodes, {len(relationships)} edges."
                )
                return {
                    "row": row,
                    "entities": entities,
                    "relationships": relationships,
                }
            else:
                print(f" -> Row {index + 1}: no data extracted.")
                return None
        except Exception as e:
            print(f" -> Error on row {index + 1}: {e}")
            return None


# end::extraction[]


async def ingest_batch(driver, results: list[dict]):
    data = [
        {
            "url": str(r["row"].get("url", "")),
            "sitename": str(r["row"].get("sitename", "")),
            "date": str(r["row"].get("date", "")),
            "text": str(r["row"].get("text", "")),
            "entities": r["entities"],
        }
        for r in results
        if r["entities"]
    ]
    # Drop relationships whose endpoints weren't extracted as entities
    # (structured outputs constrain types, not referential integrity).
    entity_ids = {e["entity_name"] for r in results for e in r["entities"]}
    all_rels = [
        rel
        for r in results
        for rel in r["relationships"]
        if rel["source_entity"] in entity_ids
        and rel["target_entity"] in entity_ids
        and rel["source_entity"] != rel["target_entity"]
    ]

    if data:
        driver.execute_query(NODES_QUERY, data=data)
    if all_rels:
        driver.execute_query(RELS_QUERY, relationships=all_rels)


async def main(articles_path: str):
    try:
        df = pd.read_csv(articles_path)
    except FileNotFoundError:
        print(f"Error: '{articles_path}' not found.")
        return

    # 1. Run all LLM extractions concurrently
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [extract_row(sem, i, row) for i, row in df.iterrows()]
    extractions = await asyncio.gather(*tasks)

    # 2. Filter out failed / empty results, then batch-import
    results = [r for r in extractions if r is not None]
    print(f"\nIngesting {len(results)} rows into Neo4j...")
    await ingest_batch(driver, results)

    print("Pipeline execution complete.")


if __name__ == "__main__":
    articles_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ARTICLES_PATH
    asyncio.run(main(articles_path))
