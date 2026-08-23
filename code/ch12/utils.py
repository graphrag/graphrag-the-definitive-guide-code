import os
import json
import asyncio
from openai import AsyncOpenAI
from neo4j import GraphDatabase

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

oai_client = AsyncOpenAI()

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-5")
TARGET_MODEL = os.getenv("TARGET_MODEL", "gpt-5.4-mini")


# tag::judge[]
async def judge(prompt: str, model: str | None = None) -> dict:
    """Run an LLM-as-judge prompt and return parsed JSON output.

    The judge runs on a different model than the system under test so that
    the judge's own biases don't inflate scores for responses from the same
    model. GPT-5-class judge models don't accept a custom ``temperature``,
    so we rely on the default and accept some per-run variance. The
    practical response is to report noise bands rather than raw deltas.
    """
    resp = await oai_client.chat.completions.create(
        model=model or JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


# end::judge[]


async def get_embedding(
    text: str, model: str = "text-embedding-3-small"
) -> list[float]:
    resp = await oai_client.embeddings.create(input=[text], model=model)
    return resp.data[0].embedding


async def answer(
    question: str,
    context: str,
    model: str | None = None,
) -> str:
    """Generate a grounded answer from a question and retrieval context."""
    prompt = (
        "Answer the user's question using ONLY the provided context. "
        "If the context does not contain the answer, say so explicitly.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    resp = await oai_client.chat.completions.create(
        model=model or TARGET_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def gather_with_concurrency(limit: int, *tasks):
    """Run coroutines with a concurrency cap."""
    sem = asyncio.Semaphore(limit)

    async def wrapped(t):
        async with sem:
            return await t

    return asyncio.gather(*(wrapped(t) for t in tasks))
