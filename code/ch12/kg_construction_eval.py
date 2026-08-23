"""Benchmark 2: evaluate the ontology-guided triple extractor.

The extractor in ``ontology_extractor.py`` follows the Chapter 7
blueprint: a Pydantic ontology compiled into the model's structured-
output schema, plus a post-processing validator for the semantics the
JSON Schema can't express. This benchmark scores it on three dimensions,
all of them deterministic and reference-based:

    completeness  - share of gold entities/relationships captured (recall)
    accuracy      - share of extracted items that match the gold set (precision)
    consistency   - duplicates, dangling references, domain/range

Both completeness and accuracy use the typed gold set from
``eval_data.py``: a relationship only counts as a hit if its source,
target, *and* type all match. Consistency defers to the ontology
validator the extractor ships with. The benchmark deliberately does not
use an LLM-as-judge for any metric: the gold set and the validator are
the entire scoring surface, so an unchanged extraction always produces
identical numbers.
"""

import asyncio
import json

from eval_data import GOLD_ARTICLE, GOLD_ENTITIES, GOLD_RELATIONSHIPS
from ontology_extractor import Extraction, extract_typed, validate_ontology
from utils import oai_client, TARGET_MODEL


# tag::extract[]
async def run_extraction(article: str) -> tuple[Extraction, list[dict], list[tuple]]:
    """Run the typed extractor and project the result into plain Python.

    The benchmark's completeness/precision helpers are dict-shaped, so we
    convert the Pydantic objects once here. The original ``Extraction``
    is also returned because the consistency check needs the typed form.
    """
    extraction = await extract_typed(article, oai_client, TARGET_MODEL)
    entities = [
        {"id": e.id, "type": e.type.value, "description": e.description}
        for e in extraction.entities
    ]
    relationships = [
        (r.source, r.type.value, r.target) for r in extraction.relationships
    ]
    return extraction, entities, relationships
# end::extract[]


# tag::completeness[]
def completeness(gold: list, predicted: list, key) -> dict:
    """Fraction of gold items whose key is present in the prediction.

    ``key`` extracts the comparable identifier: an entity id for
    entities, or a typed ``(source, type, target)`` triple for
    relationships. The function also returns the missing items, so a
    dropped score points at exactly which entity or triple was lost.
    """
    gold_keys = {key(g) for g in gold}
    pred_keys = {key(p) for p in predicted}
    hits = gold_keys & pred_keys
    missing = gold_keys - pred_keys
    return {
        "recall": len(hits) / len(gold_keys) if gold_keys else 1.0,
        "missing": sorted(missing),
    }
# end::completeness[]


# tag::precision[]
def precision(gold: list, predicted: list, key) -> dict:
    """Fraction of predicted items whose key is present in the gold set.

    The symmetric counterpart to ``completeness``: instead of "what
    should be there but isn't?", it asks "what's there that shouldn't
    be?". Returns the offending items so a dropped score is debuggable.
    Note that with a small gold set, any defensible extraction the
    curator didn't include will count as a false positive; the right
    response is to grow the gold set.
    """
    gold_keys = {key(g) for g in gold}
    pred_keys = {key(p) for p in predicted}
    hits = gold_keys & pred_keys
    extra = pred_keys - gold_keys
    return {
        "precision": len(hits) / len(pred_keys) if pred_keys else 1.0,
        "extra": sorted(extra),
    }
# end::precision[]


# tag::consistency[]
def consistency(extraction: Extraction) -> dict:
    """Defer to the ontology validator that ships with the extractor.

    The structured-output schema already guarantees that every entity
    has a valid ``EntityType`` and every relationship a valid
    ``RelationshipType``; those checks are free. What the schema
    cannot express is *semantic* correctness: that a relationship's
    source/target reference an entity in the same payload, that ids are
    unique, and that each relationship satisfies the ontology's
    domain/range. ``validate_ontology`` is the code that does that work.
    """
    return validate_ontology(extraction)
# end::consistency[]


async def main():
    extraction, ents, rels = await run_extraction(GOLD_ARTICLE)

    # tag::main[]
    report = {
        "entities_extracted":      len(ents),
        "relationships_extracted": len(rels),
        "entity_recall":           completeness(GOLD_ENTITIES, ents,
                                                key=lambda e: e["id"]),
        "entity_precision":        precision(GOLD_ENTITIES, ents,
                                             key=lambda e: e["id"]),
        "relationship_recall":     completeness(GOLD_RELATIONSHIPS, rels,
                                                key=lambda r: tuple(r)),
        "relationship_precision":  precision(GOLD_RELATIONSHIPS, rels,
                                             key=lambda r: tuple(r)),
        "consistency":             consistency(extraction),
    }
    print(json.dumps(report, indent=2))
    # end::main[]
    return report


if __name__ == "__main__":
    asyncio.run(main())
