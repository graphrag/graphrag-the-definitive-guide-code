# tag::quick_eval[]
"""Quick GraphRAG vs. vector-only RAG evaluation on Ch2's cliffhanger questions.

A deliberately small harness: one pointwise correctness judge on a 0-5 scale,
applied to the two retrievers from `graphrag_retrievers.py`. See Chapter 12
for the full evaluation methodology (process metrics, multi-judge scoring,
KG construction benchmark).
"""
import json
from statistics import mean

import openai

from graphrag_retrievers import GraphRAGApp, driver

# The canonical questions from Chapter 2 that the plain-RAG assistant struggled
# with, plus one easy single-entity baseline ("Who is Emily Johnson?").
QUESTIONS = [
    {
        "question": "Who is Emily Johnson?",
        "reference": (
            "Emily Johnson is a Principal Consultant in Conversational AI at "
            "OMG Consulting, based in Boston, with 9+ years of experience in "
            "NLP, dialogue systems, and intelligent automation."
        ),
    },
    {
        "question": (
            "Who has worked on the Cybersecurity Transformation project?"
        ),
        "reference": (
            "Ryan Carter (Project Lead), Alexander Schmidt (Security Lead), "
            "David Kim (Cloud Security), and Daniel Singh (DevSecOps) worked "
            "on the Cybersecurity Transformation project for "
            "SecureFinancial Group."
        ),
    },
    {
        "question": (
            "How many people have both GenAI and Healthcare experience?"
        ),
        "reference": (
            "Based on the OMG demo data, several consultants combine GenAI "
            "and Healthcare experience; the exact count depends on the loaded "
            "dataset but is in the low single digits."
        ),
    },
]

JUDGE_PROMPT = """\
Grade the factual correctness of a candidate answer against a reference answer.

Score whether the claims the candidate makes are factually right. A short answer
that gets its facts right is correct; wrong claims, hallucinations, or
contradictions of the reference all hurt this score.

Return JSON: {{"score": <0-5 int>, "reason": "<one sentence>"}}

Scoring: 5=all claims correct; 3=mostly correct with minor inaccuracies;
1=more wrong than right; 0=completely wrong or refused.

-Question-
{question}
-Reference-
{reference}
-Candidate-
{candidate}
"""


def judge(question: str, reference: str, candidate: str) -> dict:
    """Run the pointwise correctness judge and return parsed JSON."""
    prompt = JUDGE_PROMPT.format(
        question=question, reference=reference, candidate=candidate
    )
    resp = openai.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def evaluate(app: GraphRAGApp, method: str, label: str) -> dict:
    results = []
    for item in QUESTIONS:
        out = app.answer_question(item["question"], retrieval_method=method)
        verdict = judge(item["question"], item["reference"], out["answer"])
        results.append({
            "question": item["question"],
            "answer": out["answer"],
            "score": verdict["score"],
            "reason": verdict.get("reason", ""),
        })
    return {"retriever": label, "results": results,
            "average": mean(r["score"] for r in results)}


if __name__ == "__main__":
    app = GraphRAGApp(driver)
    runs = [
        evaluate(app, "vector", "vector"),
        evaluate(app, "vector_entity", "graphrag"),
    ]
    print(f"\n{'Question':<60} {'vector':>8} {'graphrag':>10}")
    print("-" * 80)
    for i, q in enumerate(QUESTIONS):
        v = runs[0]["results"][i]["score"]
        g = runs[1]["results"][i]["score"]
        print(f"{q['question'][:60]:<60} {v:>8} {g:>10}")
    print("-" * 80)
    print(f"{'Average':<60} {runs[0]['average']:>8.2f} "
          f"{runs[1]['average']:>10.2f}")
    driver.close()
# end::quick_eval[]
