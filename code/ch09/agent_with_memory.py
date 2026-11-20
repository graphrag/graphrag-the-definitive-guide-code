"""Agent loop extended with context-graph memory (neo4j-agent-memory v0.7+).

Diff against ch08/agent.py:

  + Loads the user's preferences and recent successful traces at the start
    of each turn (the bootstrap retrieval, query B in the Memory-Augmented
    Retrieval section) and folds them into the system prompt.
  + Records every user/assistant message into short-term memory.
  + Opens a ReasoningTrace per turn, writes a ReasoningStep + ToolCall per
    tool invocation, and closes the trace with a structured TraceOutcome.
  + The library's `on_tool_call_recorded` hook -- registered once at app
    startup via `memory_io.register_touched_hook` -- writes
    `(:ReasoningStep)-[:TOUCHED]->(:Entity)` automatically.

Everything else -- the system prompt rules, the while-loop shape, the
OpenAI chat.completions interface -- is unchanged from chapter 8.
"""
import json, os

from openai import OpenAI

from neo4j_agent_memory import MemoryClient
from neo4j_agent_memory.schema import TraceOutcome

from memory_io import load_user_context

BASE_SYSTEM_PROMPT = (
    "You have access to a Neo4j database via tools. "
    "NEVER guess or fabricate information. Either use tools or tell the user you don't know.\n\n"
    "Rules:\n"
    "1. Before ANY Cypher query, ALWAYS call `get-schema` first to understand the available node labels, "
    "relationship types, and property keys. Never assume the schema.\n"
    "2. Write Cypher queries based strictly on the schema returned — do not invent labels, types, or properties.\n"
    "3. If the tools return no results, say you don't know. Do not speculate.\n"
    "4. If the user's question cannot be answered with the available schema or tools, explicitly say so."
)


async def build_system_prompt(memory: MemoryClient, user_email: str, industry: str | None) -> str:
    ctx = await load_user_context(memory, user_email, industry=industry)
    if not ctx["preferences"] and not ctx["recent_wins"]:
        return BASE_SYSTEM_PROMPT

    extras = ["\n\nUser context (from memory):"]
    if ctx["preferences"]:
        extras.append("Preferences:")
        for p in ctx["preferences"]:
            extras.append(f"  - {p}")
    if ctx["recent_wins"]:
        extras.append("Recent successful tasks in this area:")
        for w in ctx["recent_wins"]:
            extras.append(f"  - {w['task']} → {w['outcome']}")
    return BASE_SYSTEM_PROMPT + "\n".join(extras)


async def agent_loop(
    query: str,
    *,
    openai_client: OpenAI,
    memory: MemoryClient,
    user_email: str,
    session_id: str,
    schemas: list[dict],
    executors: dict,
    industry_hint: str | None = None,
    on_tool_call=None,
    history: list[dict] | None = None,
) -> tuple[str, list[dict], list[dict]]:
    """Run one agent turn with memory recording.

    Returns (answer, tool_calls_log, messages). The OpenAI history is
    unchanged so the existing UI works without modification.
    """
    # tag::turn_open[]
    # Record the user message into short-term memory.
    user_msg = await memory.short_term.add_message(
        session_id=session_id, role="user", content=query
    )

    # Fold the user's preferences and recent successful traces into the prompt.
    system_prompt = await build_system_prompt(memory, user_email, industry_hint)

    # Open a ReasoningTrace scoped to this user; it will hold every step below.
    trace = await memory.reasoning.start_trace(
        session_id=session_id,
        task=query,
        triggered_by_message_id=str(user_msg.id),
        user_identifier=user_email,
    )
    # end::turn_open[]

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        *(history or []),
        {"role": "user", "content": query},
    ]
    tool_calls_log: list[dict] = []

    try:
        while True:
            resp = openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-5"),
                messages=messages,
                tools=schemas or None,
            )
            msg = resp.choices[0].message

            if not msg.tool_calls:
                # tag::turn_close[]
                answer = msg.content or ""
                messages.append({"role": "assistant", "content": answer})
                # Record the assistant message in short-term memory.
                await memory.short_term.add_message(
                    session_id=session_id, role="assistant", content=answer
                )
                # Close the trace with a structured outcome (success flag,
                # summary, metrics) — this is what indexes for case-based
                # retrieval in the Memory-Augmented Retrieval section.
                await memory.reasoning.complete_trace(
                    trace.id,
                    outcome=TraceOutcome(
                        success=True,
                        summary=_summarize(answer, tool_calls_log),
                        metrics={"tool_calls": len(tool_calls_log)},
                    ),
                )
                # end::turn_close[]
                return answer, tool_calls_log, messages

            messages.append(msg)
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)

                if on_tool_call:
                    on_tool_call(name, args)

                executor = executors.get(name)
                if executor is None:
                    status = "error"
                    error_kind = "UNKNOWN_TOOL"
                    error_msg = f"Unknown tool: {name}"
                    output = error_msg
                else:
                    try:
                        output = await executor(args)
                        status = "success"
                        error_kind = None
                        error_msg = None
                    except Exception as tool_exc:
                        status = "error"
                        error_kind = type(tool_exc).__name__
                        error_msg = str(tool_exc)
                        output = f"Tool {name} failed: {error_msg}"

                messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})

                # tag::tool_hooks[]
                step = await memory.reasoning.add_step(
                    trace.id,
                    thought=f"call `{name}` with {args}",
                    action=name,
                )
                # The on_tool_call_recorded hook (see memory_io.register_touched_hook)
                # fires automatically and writes :TOUCHED edges for the entities
                # this call referenced. The agent loop stays clean.
                await memory.reasoning.record_tool_call(
                    step_id=step.id,
                    tool_name=name,
                    arguments=args,
                    result=output,
                    status=status,
                    error=error_msg,
                    message_id=str(user_msg.id),
                )
                # end::tool_hooks[]

                tool_calls_log.append({"name": name, "args": args, "result": output, "status": status})

    except Exception as exc:
        await memory.reasoning.complete_trace(
            trace.id,
            outcome=TraceOutcome(
                success=False,
                summary=f"error: {exc}",
                error_kind=type(exc).__name__,
            ),
        )
        raise


def _summarize(answer: str, tool_calls: list[dict]) -> str:
    used = ", ".join(sorted({tc["name"] for tc in tool_calls}))
    head = answer.strip().splitlines()[0] if answer else ""
    return f"[tools: {used or 'none'}] {head[:200]}"
