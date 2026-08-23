"""Adopt OMG Consulting's existing knowledge graph as the long-term region
of a context graph.

`client.schema.adopt_existing_graph` (new in `neo4j-agent-memory` v0.2.0)
adds the `:Entity` super-label, a `type` property, and a stable `id` to
existing domain nodes so the library's MERGE on `(:Entity {name, type})`
finds them instead of creating duplicates.

Run this once after the chapter 3 import, before launching the agent.
The call is idempotent: re-running the script is a no-op for already-
adopted nodes.
"""
import asyncio

from neo4j_agent_memory import MemoryClient

from memory_settings import build_memory_settings


# tag::mapping[]
# OMG's six entity types from chapter 3. The mapping value becomes the
# `type` property on each adopted node; matching the label keeps the OMG
# schema readable.
LABEL_TO_TYPE = {
    "Person":     "Person",
    "Project":    "Project",
    "Client":     "Client",
    "Skill":      "Skill",
    "Technology": "Technology",
    "Industry":   "Industry",
}

# Per-label name-property overrides. OMG's :Person uses `full_name`;
# everything else has a `name` property already.
NAME_PROPERTY_PER_LABEL = {
    "Person": "full_name",
}


async def main(dry_run: bool = False) -> None:
    settings = build_memory_settings()
    async with MemoryClient(settings) as client:
        report = await client.schema.adopt_existing_graph(
            label_to_type=LABEL_TO_TYPE,
            name_property_per_label=NAME_PROPERTY_PER_LABEL,
            dry_run=dry_run,
        )
# end::mapping[]

    for r in report.by_label:
        print(
            f"  {r.label:<12}  migrated={r.migrated_count:<5}"
            f"  already_adopted={r.already_adopted_count:<5}"
            f"  skipped={r.skipped_count}"
        )
    print(f"{'(dry run) ' if dry_run else ''}adoption complete.")


if __name__ == "__main__":
    import sys
    asyncio.run(main(dry_run="--dry-run" in sys.argv))
