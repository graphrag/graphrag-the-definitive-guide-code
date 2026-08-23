import asyncio, json, os
from contextlib import AsyncExitStack

import nest_asyncio
import streamlit as st
from openai import OpenAI

from mcp_connect import connect_mcp
from mcp_tools import register_mcp_tools
from vector_tool import register_vector_tool
from fulltext_tool import register_fulltext_tool
from recommendation_tool import register_recommendation_tool
from agent import agent_loop

nest_asyncio.apply()

if "loop" not in st.session_state:
    st.session_state.loop = asyncio.new_event_loop()


def run_async(coro):
    return st.session_state.loop.run_until_complete(coro)


# ── MCP connection (once) ────────────────────────────────────────────────────


async def init_mcp():
    stack = AsyncExitStack()
    session = await connect_mcp(stack)
    schemas, executors = await register_mcp_tools(session)
    register_vector_tool(schemas, executors)
    register_fulltext_tool(schemas, executors)
    register_recommendation_tool(schemas, executors)
    return stack, schemas, executors


def ensure_connected():
    if "schemas" in st.session_state:
        return
    with st.spinner("Connecting to Neo4j MCP server..."):
        stack, schemas, executors = run_async(init_mcp())
        st.session_state.mcp_stack = stack
        st.session_state.schemas = schemas
        st.session_state.executors = executors


def render_tool_calls(tool_calls_log: list[dict]):
    """Render tool call history as collapsible expanders."""
    if not tool_calls_log:
        return
    for i, tc in enumerate(tool_calls_log, 1):
        with st.expander(f"⚡ Step {i}: `{tc['name']}`", expanded=False):
            st.markdown("**Arguments:**")
            st.code(json.dumps(tc["args"], indent=2), language="json")
            st.markdown("**Result:**")
            result = tc["result"]
            # Try to pretty-print JSON results, fall back to plain text
            try:
                parsed = json.loads(result) if isinstance(result, str) else result
                st.code(json.dumps(parsed, indent=2), language="json")
            except (json.JSONDecodeError, TypeError):
                st.code(str(result), language=None)


# ── UI ───────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Neo4j MCP Chat", page_icon="🗄️")
st.title("🗄️ Neo4j MCP Chat")

with st.sidebar:
    st.header("Connection")
    st.code(os.getenv("NEO4J_URI", "bolt://localhost:7687"), language=None)
    try:
        ensure_connected()
        st.success("Connected")
        st.header("Tools")
        for name in st.session_state.executors:
            st.caption(f"🔧 {name}")
    except Exception as e:
        st.error(f"Failed: {e}")
        st.stop()

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []  # for rendering in the UI
    st.session_state.llm_history = []  # full OpenAI messages (incl. tool calls)

for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        if msg.get("tool_calls_log"):
            render_tool_calls(msg["tool_calls_log"])
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your graph..."):
    st.session_state.display_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("Thinking...", expanded=True)

        answer, tool_calls_log, llm_messages = run_async(
            agent_loop(
                prompt,
                OpenAI(),
                st.session_state.schemas,
                st.session_state.executors,
                on_tool_call=lambda name, args: status.write(f"⚡ `{name}`"),
                history=st.session_state.llm_history,
            )
        )

        # Keep full message history (skip system prompt) for next turn
        st.session_state.llm_history = llm_messages[1:]  # exclude system prompt

        status.update(label="Done", state="complete", expanded=False)
        render_tool_calls(tool_calls_log)
        st.markdown(answer)

    st.session_state.display_messages.append(
        {"role": "assistant", "content": answer, "tool_calls_log": tool_calls_log}
    )
