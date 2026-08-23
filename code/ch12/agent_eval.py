"""Benchmark 1: end-to-end evaluation of the Chapter 8 agent.

Re-uses Chapter 8's agent loop (``agent.py``), its three custom tools
(``vector_search_projects``, ``fulltext_search_clients``, ``recommend_team``),
and the same neo4j-mcp subprocess that ch08 itself talks to for
``get_neo4j_schema`` and ``read_neo4j_cypher``. ``NEO4J_READ_ONLY=true`` in
``mcp_connect.MCP_ENV`` keeps the benchmark from mutating the graph it
scores against.

Each question is run through the agent; the full tool-call trace is
captured; then the trace + final answer are scored on two families of
metrics, in this order:

  Answer-quality (the typical end-to-end metrics):
    correctness        - claims in the answer align with the reference (0-5)
    completeness       - answer covers the facets the reference covers (0-5)
    relevance          - answer addresses the question that was asked  (0-5)

  Process (agent-specific, expanding on answer quality):
    task_completion    - did the agent produce a final answer at all?
    trajectory_match   - at least one expected tool appeared in the trace?
    tool_correctness   - fraction of calls that aren't in forbidden_tools (0-1)
    n_tool_calls       - efficiency proxy: how many tool hops did it take?

The three answer-quality metrics are LLM-as-judge scores; everything in
the process group is a deterministic check over the trace.
"""

import asyncio
import importlib.util
import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from openai import OpenAI

from agent_eval_data import EVAL_QUESTIONS
from utils import judge, JUDGE_MODEL, TARGET_MODEL

# Pull in Chapter 8's agent loop, MCP wiring, and custom tools. ch08 code
# uses ``from utils import ...`` which collides with ch12/utils.py, so we
# register ch08's utils under the name ``utils`` just long enough to load
# the modules.
_CH08 = Path(__file__).resolve().parent.parent / "ch08"


def _load(alias: str, path: Path):
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


_ch12_utils_saved = sys.modules.get("utils")
_load("utils", _CH08 / "utils.py")
_ch08_agent = _load("ch08_agent", _CH08 / "agent.py")
_ch08_mcp_connect = _load("ch08_mcp_connect", _CH08 / "mcp_connect.py")
_ch08_mcp_tools = _load("ch08_mcp_tools", _CH08 / "mcp_tools.py")
_ch08_vector = _load("ch08_vector_tool", _CH08 / "vector_tool.py")
_ch08_fulltext = _load("ch08_fulltext_tool", _CH08 / "fulltext_tool.py")
_ch08_recommend = _load("ch08_recommendation_tool", _CH08 / "recommendation_tool.py")
if _ch12_utils_saved is not None:
    sys.modules["utils"] = _ch12_utils_saved

agent_loop = _ch08_agent.agent_loop
connect_mcp = _ch08_mcp_connect.connect_mcp
register_mcp_tools = _ch08_mcp_tools.register_mcp_tools


# tag::setup[]
async def setup_tools(exit_stack: AsyncExitStack) -> tuple[list[dict], dict]:
    """Assemble the full tool registry the agent will see during the eval.

    Schema + Cypher come from the same neo4j-mcp subprocess ch08 uses in
    production; the three specialised tools are registered in-process from
    ch08's modules. Running both through the real MCP wiring means the
    benchmark exercises the deployed tool surface, not a re-implementation
    that could drift from it.
    """
    session = await connect_mcp(exit_stack)
    schemas, executors = await register_mcp_tools(session)
    _ch08_vector.register_vector_tool(schemas, executors)
    _ch08_fulltext.register_fulltext_tool(schemas, executors)
    _ch08_recommend.register_recommendation_tool(schemas, executors)
    return schemas, executors


# end::setup[]


# tag::answer_judge_prompts[]
CORRECTNESS_PROMPT = """Grade the factual correctness of a candidate answer against a reference answer.

Score whether the claims the candidate makes are factually right. A short answer that gets
its facts right is correct; missing facts are scored separately as completeness. Wrong
claims, hallucinations, or contradictions of the reference all hurt this score.

Return JSON: {{"score": <0-5 int>, "reason": "<one sentence>"}}

Scoring: 5=all claims correct; 3=mostly correct with minor inaccuracies; 1=more wrong than right; 0=completely wrong or refused.

-Reference-
{reference}
-Candidate-
{candidate}
"""


COMPLETENESS_PROMPT = """Grade how completely a candidate answer covers the facets in the reference.

List the distinct facts or facets in the reference, then check which of them the candidate
mentions. The score reflects the share of facets covered. Extra detail beyond the reference
neither helps nor hurts.

Return JSON: {{"score": <0-5 int>, "reason": "<one sentence>"}}

Scoring: 5=covers every facet; 3=covers most facets, some omissions; 1=covers a small fraction; 0=covers nothing.

-Reference-
{reference}
-Candidate-
{candidate}
"""


RELEVANCE_PROMPT = """Grade whether a candidate answer addresses the question that was asked.

A relevant answer responds to the question, even if it does so incompletely or with mistakes.
An irrelevant answer drifts off-topic, refuses without justification, or answers a different question.

Return JSON: {{"score": <0-5 int>, "reason": "<one sentence>"}}

Scoring: 5=directly addresses the question; 3=addresses most of it; 1=loosely related; 0=off-topic or refused.

-Question-
{question}
-Candidate-
{candidate}
"""
# end::answer_judge_prompts[]


async def score_correctness(reference: str, candidate: str) -> dict:
    return await judge(CORRECTNESS_PROMPT.format(reference=reference, candidate=candidate))


async def score_completeness(reference: str, candidate: str) -> dict:
    return await judge(COMPLETENESS_PROMPT.format(reference=reference, candidate=candidate))


async def score_relevance(question: str, candidate: str) -> dict:
    return await judge(RELEVANCE_PROMPT.format(question=question, candidate=candidate))


# tag::process_metrics[]
def trajectory_match(expected_any: list[str], calls: list[dict]) -> bool:
    """Any-order match: at least one of the expected tools appeared in the trace."""
    if not expected_any:
        return True
    called = {c["name"] for c in calls}
    return any(e in called for e in expected_any)


def tool_correctness(forbidden: list[str], calls: list[dict]) -> float:
    """Fraction of tool calls that aren't in the per-question forbidden set.

    Tool correctness is a deterministic check: each question lists the tools
    that would be inappropriate to call (e.g. ``get_neo4j_schema`` for a
    question that doesn't need Cypher). 1.0 means every call was permitted;
    lower values flag wasted hops.
    """
    if not calls:
        return 1.0
    forbidden_set = set(forbidden or [])
    return sum(1 for c in calls if c["name"] not in forbidden_set) / len(calls)


# end::process_metrics[]


# tag::run[]
async def run_one(q: dict, client: OpenAI, schemas: list[dict], executors: dict) -> dict:
    """Run one question through the agent and score the trace.

    Answer-quality metrics (correctness, completeness, relevance) come
    first; they tell us whether the agent actually answered the question
    well. Process metrics come second; they tell us *how* it got there,
    which is what makes an expensive agent operable in production.
    """
    answer, calls, _ = await agent_loop(q["question"], client, schemas, executors)

    correctness, completeness, relevance = await asyncio.gather(
        score_correctness(q["reference"], answer),
        score_completeness(q["reference"], answer),
        score_relevance(q["question"], answer),
    )

    return {
        "id":                  q["id"],
        # Answer-quality (typical end-to-end metrics, LLM-judged).
        "correctness":         correctness["score"],
        "correctness_reason":  correctness.get("reason", ""),
        "completeness":        completeness["score"],
        "completeness_reason": completeness.get("reason", ""),
        "relevance":           relevance["score"],
        # Agent process (deterministic checks over the trace).
        "task_completion":     bool(answer and answer.strip()),
        "trajectory_match":    trajectory_match(q.get("expected_tools_any", []), calls),
        "tool_correctness":    tool_correctness(q.get("forbidden_tools", []), calls),
        "n_tool_calls":        len(calls),
        # Full call trace (name + args) so two eval runs can be diffed:
        # which arguments shifted when a model or prompt change moves a score?
        "tool_calls":          [{"name": c["name"], "args": c["args"]} for c in calls],
    }
# end::run[]


async def main():
    client = OpenAI()
    # The ch08 agent_loop reads OPENAI_MODEL, not TARGET_MODEL. Mirror our
    # ch12 convention so the judge is always stronger than the system under test.
    os.environ["OPENAI_MODEL"] = TARGET_MODEL

    async with AsyncExitStack() as stack:
        schemas, executors = await setup_tools(stack)

        results = []
        for q in EVAL_QUESTIONS:
            try:
                r = await run_one(q, client, schemas, executors)
            except Exception as e:
                r = {"id": q["id"], "error": str(e)}
            results.append(r)
            if "error" in r:
                print(f"{q['id']:<32} ERROR: {r['error']}")
            else:
                print(
                    f"{q['id']:<32} "
                    f"correctness={r['correctness']} "
                    f"completeness={r['completeness']} "
                    f"relevance={r['relevance']} "
                    f"tool_calls={r['n_tool_calls']}"
                )

    print("\n=== Summary ===")
    print(json.dumps(_summarise(results), indent=2))


def _summarise(results: list[dict]) -> dict:
    good = [r for r in results if "error" not in r]
    if not good:
        return {"runs": len(results), "errors": len(results)}
    return {
        "runs":                  len(good),
        # Answer quality first.
        "correctness_avg":       sum(r["correctness"]  for r in good) / len(good),
        "completeness_avg":      sum(r["completeness"] for r in good) / len(good),
        "relevance_avg":         sum(r["relevance"]    for r in good) / len(good),
        # Then process.
        "task_completion":       sum(r["task_completion"]  for r in good) / len(good),
        "trajectory_match":      sum(r["trajectory_match"] for r in good) / len(good),
        "tool_correctness_avg":  sum(r["tool_correctness"] for r in good) / len(good),
        "avg_tool_calls":        sum(r["n_tool_calls"]     for r in good) / len(good),
    }


if __name__ == "__main__":
    asyncio.run(main())
