"""Ontology-guided triple extraction: worked example for the benchmark.

A small, self-contained implementation of the Chapter 7 "JSON generation
plus import" extraction blueprint, scored end-to-end by
``kg_construction_eval.py``:

    1. Define the ontology in code as Pydantic models. Entity and
       relationship types are enums; the LLM cannot return anything
       outside the enum because the SDK's structured-output mode rejects
       schema-violating completions before we ever see them.

    2. Run the extractor against the source text. The model proposes
       candidates *within* the schema: typed entities and typed
       relationships referenced by id.

    3. Validate the semantics that the JSON Schema cannot express:
       relationships must reference entities the extractor also produced
       (no dangling endpoints), entity ids must be unique (entity
       resolution), and the source/target types of every relationship
       must satisfy the ontology's domain/range constraints.

The Chapter 7 point lives in step 3: structured outputs give you a typed
shape, not semantic correctness. The ontology is the scaffolding; the
enforcement lives in code.
"""

from __future__ import annotations

import re
from enum import Enum

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator


# tag::canonicalise[]
_ID_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9-]*$")


def _canonical_id(value: str) -> str:
    """Normalise an entity id to the ALL-CAPS-HYPHENATED form.

    The model is told to emit canonical ids in the prompt, but the prompt
    does not strictly bind the model -- in practice we still see
    "SATYA NADELLA" sneak through when the only example in the source
    text uses spaces. Normalising in a Pydantic validator means a single
    canonical form lands in the graph regardless of model variance.
    """
    cleaned = re.sub(r"\s+", "-", value.strip()).upper()
    if not _ID_PATTERN.match(cleaned):
        raise ValueError(f"Id {value!r} does not match {_ID_PATTERN.pattern}")
    return cleaned
# end::canonicalise[]


# tag::ontology[]
class EntityType(str, Enum):
    ORGANIZATION = "ORGANIZATION"
    PERSON = "PERSON"
    EVENT = "EVENT"


class RelationshipType(str, Enum):
    ACQUIRED = "ACQUIRED"          # ORGANIZATION -> ORGANIZATION
    LED_BY = "LED_BY"              # ORGANIZATION -> PERSON
    INVOLVED_IN = "INVOLVED_IN"    # PERSON | ORGANIZATION -> EVENT
    REVIEWED = "REVIEWED"          # ORGANIZATION -> EVENT


# Domain / range constraints. The Pydantic schema can enforce that a
# relationship has a valid ``type``, but it cannot enforce that ACQUIRED's
# source is an ORGANIZATION; that is checked by ``validate_ontology``.
RELATIONSHIP_CONSTRAINTS: dict[
    RelationshipType, tuple[set[EntityType], set[EntityType]]
] = {
    RelationshipType.ACQUIRED:    ({EntityType.ORGANIZATION}, {EntityType.ORGANIZATION}),
    RelationshipType.LED_BY:      ({EntityType.ORGANIZATION}, {EntityType.PERSON}),
    RelationshipType.INVOLVED_IN: ({EntityType.PERSON, EntityType.ORGANIZATION},
                                   {EntityType.EVENT}),
    RelationshipType.REVIEWED:    ({EntityType.ORGANIZATION}, {EntityType.EVENT}),
}


class Entity(BaseModel):
    id: str = Field(
        ...,
        description=("ALL-CAPS-HYPHENATED unique identifier (e.g. MICROSOFT, "
                     "ACTIVISION-BLIZZARD, MICROSOFT-ACTIVISION-ACQUISITION)."),
    )
    type: EntityType
    description: str = Field(
        ...,
        description=("One-sentence factual description grounded in the article. "
                     "Do not add outside knowledge."),
    )

    @field_validator("id")
    @classmethod
    def _canonical(cls, v: str) -> str:
        return _canonical_id(v)


class Relationship(BaseModel):
    source: str = Field(..., description="Entity id from the entities list.")
    target: str = Field(..., description="Entity id from the entities list.")
    type: RelationshipType

    @field_validator("source", "target")
    @classmethod
    def _canonical(cls, v: str) -> str:
        return _canonical_id(v)


class Extraction(BaseModel):
    entities: list[Entity]
    relationships: list[Relationship]
# end::ontology[]


# tag::prompt[]
ONTOLOGY_PROMPT = """\
Extract a knowledge graph from the article below using the ontology.

Entity types:
- ORGANIZATION: companies AND regulators (e.g. UK-CMA, US-FTC, EUROPEAN-COMMISSION)
                AND public institutions. Always extract every regulator named in the
                text, even if it's only mentioned in passing as "approved" or "reviewed".
- PERSON:       any named individual, including executives mentioned by name+title
                (e.g. CEOs, board members, regulators' chairs). If the article names
                someone as the head of an organisation, extract them.
- EVENT:        notable occurrences (acquisitions, launches, lawsuits, regulatory
                approvals).

Relationship types (with required domain -> range):
- ACQUIRED:    ORGANIZATION -> ORGANIZATION
- LED_BY:      ORGANIZATION -> PERSON
- INVOLVED_IN: PERSON or ORGANIZATION -> EVENT
- REVIEWED:    ORGANIZATION -> EVENT (regulators reviewing an event;
               extract one REVIEWED relationship per regulator named).

ID rules:
- ALL-CAPS, HYPHEN-SEPARATED. No spaces. No periods.
- Use the form the article itself uses (full name if the article uses
  the full name, abbreviation if the article uses the abbreviation).

Other rules:
- For every regulator the article mentions, emit both the ORGANIZATION entity
  and the REVIEWED triple to the event.
- For every executive the article names, emit both the PERSON entity and the
  LED_BY triple from their organisation.
- Every relationship's source and target must reference an entity you
  also extract.
- Every description must be supported by the article. Do not add
  outside knowledge.

Article:
{article}
"""
# end::prompt[]


# tag::extract[]
async def extract_typed(
    article: str, client: AsyncOpenAI, model: str
) -> Extraction:
    """Run the ontology-guided extractor and parse the typed output.

    ``chat.completions.parse`` puts the SDK in structured-output mode:
    the model is constrained to JSON that validates against the
    ``Extraction`` schema, and the SDK returns it already parsed into a
    Pydantic instance. There is no hand-written parser, no delimiter
    string to keep in sync, and no chance of an enum value that isn't in
    the ontology.
    """
    completion = await client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system",
             "content": "You extract knowledge graphs that match the supplied ontology."},
            {"role": "user",
             "content": ONTOLOGY_PROMPT.format(article=article)},
        ],
        response_format=Extraction,
    )
    return completion.choices[0].message.parsed
# end::extract[]


# tag::validate[]
def validate_ontology(extraction: Extraction) -> dict:
    """Post-process checks the JSON Schema cannot express.

    Three failure modes worth flagging at the construction stage:

    * **Duplicate ids**: two extracted entities share an id, which is
      the entity-resolution failure that downstream retrieval will
      experience as fragmented neighbourhoods.
    * **Dangling endpoints**: a relationship references an id the
      extractor did not produce as an entity. The schema treats
      ``source`` and ``target`` as plain strings, so it cannot catch
      this on its own.
    * **Domain/range violations**: the relationship's source/target
      types do not satisfy the ontology's constraints (for example
      ``ACQUIRED`` between a PERSON and an ORGANIZATION). This is the
      semantic check Chapter 7 explicitly says LLMs cannot perform from
      the prompt alone.
    """
    seen: set[str] = set()
    duplicate_ids: list[str] = []
    for e in extraction.entities:
        if e.id in seen:
            duplicate_ids.append(e.id)
        seen.add(e.id)

    entity_types = {e.id: e.type for e in extraction.entities}
    dangling: list[dict] = []
    domain_range_violations: list[dict] = []

    for r in extraction.relationships:
        if r.source not in entity_types or r.target not in entity_types:
            dangling.append({
                "type":    r.type.value,
                "source":  r.source,
                "target":  r.target,
                "missing": [x for x in (r.source, r.target) if x not in entity_types],
            })
            continue

        allowed_source, allowed_target = RELATIONSHIP_CONSTRAINTS[r.type]
        if (entity_types[r.source] not in allowed_source
                or entity_types[r.target] not in allowed_target):
            domain_range_violations.append({
                "type":    r.type.value,
                "source":  f"{r.source} ({entity_types[r.source].value})",
                "target":  f"{r.target} ({entity_types[r.target].value})",
                "allowed": (f"{sorted(t.value for t in allowed_source)} -> "
                            f"{sorted(t.value for t in allowed_target)}"),
            })

    return {
        "valid": not (duplicate_ids or dangling or domain_range_violations),
        "duplicate_ids":           sorted(set(duplicate_ids)),
        "dangling_endpoints":      dangling,
        "domain_range_violations": domain_range_violations,
    }
# end::validate[]
