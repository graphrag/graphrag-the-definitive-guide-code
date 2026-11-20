"""Streamlit chat app for the OMG agent with context-graph memory.

Diff against ch08/app_agent.py:

  + A "Who are you?" sidebar selector picks the user (Sara, Sarah Chen).
  + Past sessions for that user appear in the sidebar; clicking one loads it.
  + A preferences panel shows the user's recorded preferences.
  + Each agent response collapses a "Reasoning trace" expander populated
    from the graph itself, not just from an in-memory log.
"""
import asyncio, os, uuid
from contextlib import AsyncExitStack

import nest_asyncio
import streamlit as st
from openai import OpenAI
from neo4j_agent_memory import MemoryClient

# Reuse ch08's tool registration verbatim
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch08"))
from mcp_connect import connect_mcp                  # noqa: E402
from mcp_tools import register_mcp_tools             # noqa: E402
from vector_tool import register_vector_tool         # noqa: E402
from fulltext_tool import register_fulltext_tool     # noqa: E402
from recommendation_tool import register_recommendation_tool  # noqa: E402

from memory_settings import build_memory_settings
from memory_io import register_touched_hook
from agent_with_memory import agent_loop


nest_asyncio.apply()

USERS = {
    "Sara Patel (Director of Staffing)":   ("sara@omg.com",        {"name": "Sara Patel",    "role": "Director of Staffing"}),
    "Sarah Chen (Senior Consultant)":      ("sarah.chen@omg.com",  {"name": "Sarah Chen",    "role": "Senior Consultant"}),
    "Emily Johnson (Principal Consultant)":("emily.johnson@omg.com",{"name": "Emily Johnson","role": "Principal Consultant"}),
}

if "loop" not in st.session_state:
    st.session_state.loop = asyncio.new_event_loop()


def run_async(coro):
    return st.session_state.loop.run_until_complete(coro)


# ── One-time setup ──────────────────────────────────────────────────────────


async def init_all():
    stack = AsyncExitStack()
    mcp_session = await connect_mcp(stack)
    schemas, executors = await register_mcp_tools(mcp_session)
    register_vector_tool(schemas, executors)
    register_fulltext_tool(schemas, executors)
    register_recommendation_tool(schemas, executors)

    memory = await stack.enter_async_context(MemoryClient(build_memory_settings()))

    # Library-native :User upsert; replaces the v0.1.x hand-rolled Cypher.
    for email, attrs in USERS.values():
        await memory.users.upsert_user(identifier=email, attributes=attrs)

    # Register the on_tool_call_recorded hook once. From now on, every
    # record_tool_call writes :TOUCHED edges for the OMG entities the
    # call referenced.
    register_touched_hook(memory)

    return stack, schemas, executors, memory


def ensure_initialized():
    if "schemas" in st.session_state:
        return
    with st.spinner("Connecting to Neo4j MCP server and memory..."):
        stack, schemas, executors, memory = run_async(init_all())
        st.session_state.app_stack = stack
        st.session_state.schemas = schemas
        st.session_state.executors = executors
        st.session_state.memory = memory


# ── Memory-driven sidebar widgets ──────────────────────────────────────────


async def list_user_conversations(memory: MemoryClient, email: str) -> list[dict]:
    rows = await memory.graph.execute_read(
        """
        MATCH (u:User {identifier: $email})-[:HAS_CONVERSATION]->(c:Conversation)
        RETURN c.session_id AS session_id,
               c.title AS title,
               c.updated_at AS updated_at
        ORDER BY c.updated_at DESC
        LIMIT 10
        """,
        {"email": email},
    )
    return rows


async def list_user_preferences(memory: MemoryClient, email: str) -> list[dict]:
    """Native preferences API plus a tiny extra read for scope."""
    prefs = await memory.long_term.get_preferences_for(
        user_identifier=email, active_only=True
    )
    if not prefs:
        return []
    rows = await memory.graph.execute_read(
        """
        MATCH (p:Preference) WHERE p.id IN $ids
        OPTIONAL MATCH (p)-[:APPLIES_TO]->(scope)
        RETURN p.id AS id,
               p.preference AS preference,
               p.category AS category,
               coalesce(scope.name, '*') AS scope
        """,
        {"ids": [str(p.id) for p in prefs]},
    )
    return rows


async def fetch_trace_for_message(memory: MemoryClient, message_id: str) -> dict | None:
    rows = await memory.graph.execute_read(
        """
        MATCH (m:Message {id: $mid})<-[:INITIATED_BY]-(rt:ReasoningTrace)
        OPTIONAL MATCH (rt)-[:HAS_STEP]->(step:ReasoningStep)
        OPTIONAL MATCH (step)-[:USES_TOOL]->(tc:ToolCall)
        OPTIONAL MATCH (step)-[:TOUCHED]->(e:Entity)
        WITH rt, step, tc, collect(DISTINCT e.name + ' (' + e.type + ')') AS touched
        ORDER BY step.step_number ASC
        RETURN rt.task AS task,
               rt.outcome AS outcome,
               rt.success AS success,
               collect({
                 step: step.step_number,
                 thought: step.thought,
                 tool: tc.tool_name,
                 args: tc.arguments,
                 status: tc.status,
                 touched: touched
               }) AS steps
        """,
        {"mid": message_id},
    )
    return rows[0] if rows else None


# ── UI ─────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="OMG Memory Chat", page_icon="🧠")
st.title("🧠 OMG Memory Chat")

ensure_initialized()
memory = st.session_state.memory

# tag::sidebar_additions[]
with st.sidebar:
    st.header("Who are you?")
    user_label = st.selectbox("User", list(USERS.keys()))
    email, attrs = USERS[user_label]
    st.caption(f"{attrs['name']} — {attrs['role']}")

    st.header("Your preferences")
    prefs = run_async(list_user_preferences(memory, email))
    if prefs:
        for p in prefs:
            st.markdown(f"- **{p['scope']}**: {p['preference']}")
    else:
        st.caption("(none recorded yet)")

    st.header("Past sessions")
    convs = run_async(list_user_conversations(memory, email))
    for c in convs:
        if st.button(c["title"] or c["session_id"][:8], key=c["session_id"]):
            st.session_state.session_id = c["session_id"]
            st.session_state.display_messages = []
            st.session_state.llm_history = []
# end::sidebar_additions[]

# Open a fresh conversation if none is active. The library writes the
# (:User)-[:HAS_CONVERSATION]->(:Conversation) edge automatically when
# user_identifier is provided.
if "session_id" not in st.session_state or st.session_state.get("session_user") != email:
    conv = run_async(memory.short_term.create_conversation(
        session_id=str(uuid.uuid4()),
        title="New conversation", user_identifier=email,
    ))
    st.session_state.session_id = conv.session_id
    st.session_state.session_user = email
    st.session_state.display_messages = []
    st.session_state.llm_history = []

# Render history
for m in st.session_state.display_messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("trace"):
            with st.expander("Reasoning trace"):
                for s in m["trace"]["steps"]:
                    st.markdown(
                        f"- *{s['thought']}* → `{s['tool']}` "
                        f"(touched: {', '.join(s['touched']) or '—'})"
                    )

if prompt := st.chat_input("Ask about OMG staffing..."):
    st.session_state.display_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("Thinking...", expanded=True)

        answer, _, llm_messages = run_async(
            agent_loop(
                prompt,
                openai_client=OpenAI(),
                memory=memory,
                user_email=email,
                session_id=st.session_state.session_id,
                schemas=st.session_state.schemas,
                executors=st.session_state.executors,
                on_tool_call=lambda n, a: status.write(f"⚡ `{n}`"),
                history=st.session_state.llm_history,
            )
        )

        st.session_state.llm_history = llm_messages[1:]
        status.update(label="Done", state="complete", expanded=False)
        st.markdown(answer)

        # tag::trace_panel[]
        # Read the just-written trace back from memory so the expander
        # shows the same data an auditor would see in cypher-shell.
        last_user_msg_id = run_async(
            memory.graph.execute_read(
                """
                MATCH (c:Conversation {session_id: $sid})
                      -[:HAS_MESSAGE]->(m:Message {role:'user'})
                RETURN m.id AS id ORDER BY m.timestamp DESC LIMIT 1
                """,
                {"sid": st.session_state.session_id},
            )
        )
        trace = None
        if last_user_msg_id:
            trace = run_async(
                fetch_trace_for_message(memory, last_user_msg_id[0]["id"])
            )
        if trace:
            with st.expander("Reasoning trace"):
                for s in trace["steps"]:
                    st.markdown(
                        f"- *{s['thought']}* → `{s['tool']}` "
                        f"(touched: {', '.join(s['touched']) or '—'})"
                    )
        # end::trace_panel[]

    st.session_state.display_messages.append(
        {"role": "assistant", "content": answer, "trace": trace}
    )
