import os
from contextlib import AsyncExitStack

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

MCP_ENV = {
    **os.environ,
    "NEO4J_URI": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    "NEO4J_USERNAME": os.getenv("NEO4J_USERNAME", "neo4j"),
    "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD", "password"),
    "NEO4J_DATABASE": os.getenv("NEO4J_DATABASE", "neo4j"),
    "NEO4J_READ_ONLY": os.getenv("NEO4J_READ_ONLY", "true"),
}


async def connect_mcp(exit_stack: AsyncExitStack) -> ClientSession:
    """Launch neo4j-mcp as a subprocess and return a live session."""
    params = StdioServerParameters(
        command="uvx",
        args=["neo4j-mcp-server", "--neo4j-transport-mode", "stdio"],
        env=MCP_ENV,
    )
    read, write = await exit_stack.enter_async_context(stdio_client(params))
    session = await exit_stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    return session
