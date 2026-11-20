"""OMG-specific glue around `neo4j-agent-memory` (v0.7 or later).

In v0.1.x this module wrapped a great deal of the library's behavior: it
created a `:User` node by hand, wrote `(:User)-[:HAS_CONVERSATION]->(:Conversation)`
edges via raw Cypher, and instrumented `(:ReasoningStep)-[:TOUCHED]->(:Entity)`
edges after every tool call. v0.2.0 ships all of that natively, so this
module shrinks to one OMG-specific concern: mapping a tool call to the
entities it touched. We register that mapping as a hook on the library's
new `on_tool_call_recorded` event, and the library writes the `:TOUCHED`
edges itself.
"""
import re
from typing import Any

from neo4j_agent_memory import MemoryClient
from neo4j_agent_memory.schema import EntityRef


# tag::touched_inference[]
# ── Touched-entity inference for OMG tools ─────────────────────────────────


_NAME_RE = re.compile(r'"([^"]+)"\s*:\s*"([^"]+)"')


def _names_from_json(json_str: str, key: str) -> list[str]:
    if not isinstance(json_str, str):
        return []
    return [v for k, v in _NAME_RE.findall(json_str) if k == key]


def infer_touched_entities(
    tool_name: str, args: dict, result: Any
) -> list[EntityRef]:
    """Map an OMG tool call to the long-term entities it referenced.

    The agent already knows which arguments name which entity types --
    we encode that mapping here so the library's `on_tool_call_recorded`
    hook can fire `:TOUCHED` edges with no extra cost in the agent loop.
    """
    refs: list[EntityRef] = []

    if tool_name == "fulltext_search_clients":
        if name := args.get("search_term"):
            refs.append(EntityRef(name=name, type="Client"))

    elif tool_name == "vector_search_projects":
        for name in _names_from_json(result, key="name"):
            refs.append(EntityRef(name=name, type="Project"))

    elif tool_name == "recommend_team":
        if name := args.get("client_name"):
            refs.append(EntityRef(name=name, type="Client"))
        for tech in args.get("required_technologies") or []:
            refs.append(EntityRef(name=tech, type="Technology"))
        for name in _names_from_json(result, key="candidate"):
            refs.append(EntityRef(name=name, type="Person"))

    return refs


def register_touched_hook(client: MemoryClient) -> None:
    """Register the touched-entity hook with the library.

    After this call, every `record_tool_call` automatically writes
    `(:ReasoningStep)-[:TOUCHED]->(:Entity)` edges for the entities the
    OMG tool referenced. The agent loop doesn't need to know.
    """
    @client.reasoning.on_tool_call_recorded
    async def _on_tool_call(tool_call, ctx):
        for ref in infer_touched_entities(
            tool_call.tool_name, tool_call.arguments, tool_call.result
        ):
            await ctx.add_touched_edge(ref)
# end::touched_inference[]


# tag::bootstrap[]
# ── Bootstrap retrieval at session start ───────────────────────────────────


async def load_user_context(
    client: MemoryClient, user_email: str, industry: str | None = None
) -> dict:
    """Fetch the user's preferences and recent successful traces in the
    relevant industry. Folded into the system prompt before the first turn.

    Preferences come from the library's first-class API; the recent-traces
    half traverses the graph directly because there's no equivalent
    high-level helper for "successful traces touching :Industry X."
    """
    prefs = await client.long_term.get_preferences_for(
        user_identifier=user_email,
        applies_to=EntityRef(name=industry, type="Industry")
            if industry else None,
        active_only=True,
    )

    rows = await client.graph.execute_read(
        """
        MATCH (u:User {identifier: $email})-[:HAS_TRACE]->(rt:ReasoningTrace)
              -[:HAS_STEP]->(:ReasoningStep)-[:TOUCHED]->(:Client)
              -[:IN_INDUSTRY]->(i:Industry)
        WHERE rt.success = true
          AND ($industry IS NULL OR i.name = $industry)
          AND rt.completed_at >= datetime() - duration({days: 30})
        WITH DISTINCT rt
        ORDER BY rt.completed_at DESC
        LIMIT 5
        RETURN rt.task AS task, rt.outcome AS outcome
        """,
        {"email": user_email, "industry": industry},
    )

    return {
        "preferences": [p.preference for p in prefs],
        "recent_wins": [
            {"task": r["task"], "outcome": r["outcome"]} for r in rows
        ],
    }
# end::bootstrap[]
