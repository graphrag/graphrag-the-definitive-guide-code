import json, os
from openai import OpenAI

SYSTEM_PROMPT = (
    "You have access to a Neo4j database via tools. "
    "NEVER guess or fabricate information. Either use tools or tell the "
    "user you don't know.\n\n"
    "Rules:\n"
    "1. Most tools (vector search, fulltext search, recommendations) "
    "have their own typed inputs and must be called directly without "
    "any preceding `get_schema` call. The ONLY exception is "
    "`read-cypher`: before any `read-cypher` call, you must first call "
    "`get_schema` so you know the available node labels, relationship "
    "types, and property keys. Calling `get_schema` before a "
    "non-Cypher tool is a bug — do not do it.\n"
    "2. Write Cypher queries based strictly on the schema returned — do "
    "not invent labels, relationship types, or properties. If a "
    "relationship or property you'd like to query for is not in the "
    "schema, it does not exist; either rephrase the query against what "
    "the schema does have, or tell the user that information is not in "
    "the graph.\n"
    "3. Report what tools return verbatim. Never invent names, IDs, "
    "addresses, scores, dates, or other specifics that the tools did "
    "not actually return. If a tool surfaces partial information, "
    "report exactly what it returned and explicitly state which "
    "requested fields were not available.\n"
    "4. If the tools return no results, say you don't know. Do not speculate.\n"
    "5. If the user's question cannot be answered with the available "
    "schema or tools, explicitly say so."
)

async def agent_loop(
    query: str,
    client: OpenAI,
    schemas: list[dict],
    executors: dict,
    on_tool_call=None,
    history: list[dict] | None = None,
) -> tuple[str, list[dict], list[dict]]:
    """Run the agent loop returning (answer, tool_calls_log, messages).

    *history* should be the raw OpenAI messages list from prior turns
    (including tool_calls and tool responses) so the LLM has full context.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *(history or []),
        {"role": "user", "content": query},
    ]
    tool_calls_log: list[dict] = []

    while True:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5"),
            messages=messages,
            tools=schemas or None,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            return msg.content or "", tool_calls_log, messages

        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)

            if on_tool_call:
                on_tool_call(name, args)

            executor = executors.get(name)
            output = (
                await executor(args) if executor else f"Unknown tool: {name}"
            )
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": output}
            )

            tool_calls_log.append(
                {"name": name, "args": args, "result": output}
            )
