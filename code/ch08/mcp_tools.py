from mcp import ClientSession


async def register_mcp_tools(session: ClientSession) -> tuple[list[dict], dict]:
    """
    Ask the server what tools it has. Returns:
      - schemas:   list of OpenAI function-calling dicts
      - executors: dict mapping tool name -> async fn(args) -> str
    """
    resp = await session.list_tools()
    schemas, executors = [], {}

    for t in resp.tools:
        name = t.name
        params = {} if t.inputSchema == {"type": "object"} else t.inputSchema
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": t.description or "",
                    "parameters": params,
                },
            }
        )

        async def execute(args, _n=name, _s=session):
            result = await _s.call_tool(_n, arguments=args)
            return "\n".join(
                p.text for p in result.content if hasattr(p, "text")
            )

        executors[name] = execute

    return schemas, executors
