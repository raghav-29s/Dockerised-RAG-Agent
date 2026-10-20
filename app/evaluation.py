"""Evaluation suite for the RAG agent.

Retrieval metrics (hit rate, MRR) are deterministic: a question is a hit when one of
its evidence strings appears in a retrieved chunk. Evidence strings are used instead of
chunk IDs so the test set stays valid when the chunking strategy changes.

Faithfulness, answer relevance and correctness are scored by an LLM judge.

Run standalone inside the container:
    docker compose exec rag-api python -m app.evaluation
"""
import argparse
import json
import re
from pathlib import Path

from app import llm
from app.rag import answer_question, format_context, retrieve

TEST_SET_PATH = Path(__file__).resolve().parent.parent / "eval" / "test_set.json"

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of a retrieval-augmented question answering system.
Score the generated answer on three criteria, each a number between 0 and 1:

- faithfulness: the fraction of claims in the answer that are supported by the retrieved context. An answer that only says the information could not be found scores 1.
- answer_relevance: how directly the answer addresses the question (1 = answers it directly, 0.5 = partially, 0 = off-topic or declines to answer).
- correctness: agreement with the reference answer (1 = same facts, 0.5 = partially correct or incomplete, 0 = wrong or missing).

Respond with JSON only: {"faithfulness": x, "answer_relevance": x, "correctness": x, "reason": "<one sentence>"}"""


def _norm(text):
    return re.sub(r"\s+", " ", text).lower()


def _first_hit_rank(chunks, evidence):
    for chunk in chunks:
        if any(_norm(e) in _norm(chunk["text"]) for e in evidence):
            return chunk["rank"]
    return None


def _judge(question, reference, chunks, answer):
    raw = llm.chat(
        [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Reference answer: {reference}\n\n"
                    f"Retrieved context:\n{format_context(chunks)}\n\n"
                    f"Generated answer: {answer}"
                ),
            },
        ],
        json_mode=True,
    )
    # Some OpenAI-compatible servers wrap JSON in prose or code fences
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    scores = json.loads(match.group(0) if match else raw)

    result = {
        key: min(max(float(scores[key]), 0.0), 1.0)
        for key in ("faithfulness", "answer_relevance", "correctness")
    }
    result["judge_reason"] = str(scores.get("reason", ""))
    return result


def _mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def run_evaluation(top_k=5, retrieval_only=False):
    tests = json.loads(TEST_SET_PATH.read_text())
    use_llm = llm.enabled() and not retrieval_only
    results = []

    for test in tests:
        chunks = retrieve(test["question"], top_k)
        answerable = bool(test["evidence"])
        rank = _first_hit_rank(chunks, test["evidence"]) if answerable else None

        row = {
            "id": test["id"],
            "question": test["question"],
            "answerable": answerable,
            "hit": (rank is not None) if answerable else None,
            "hit_rank": rank,
            "retrieved": [f"{c['source']} p.{c['page']} #{c['chunk_index']}" for c in chunks],
        }

        if use_llm:
            try:
                answer, _ = answer_question(test["question"], chunks)
                row["answer"] = answer
                row.update(_judge(test["question"], test["ground_truth"], chunks, answer))
                if not answerable:
                    row["abstained"] = "could not find" in answer.lower()
            except Exception as exc:  # keep evaluating the remaining questions
                row["error"] = f"{type(exc).__name__}: {exc}"

        results.append(row)

    answerable_rows = [r for r in results if r["answerable"]]

    report = {
        "num_questions": len(results),
        "num_answerable": len(answerable_rows),
        "top_k": top_k,
        "hit_rate": _mean([float(r["hit"]) for r in answerable_rows]),
        "mrr": _mean([1 / r["hit_rank"] if r["hit_rank"] else 0.0 for r in answerable_rows]),
        "faithfulness": _mean([r.get("faithfulness") for r in results]),
        "answer_relevance": _mean([r.get("answer_relevance") for r in answerable_rows]),
        "answer_correctness": _mean([r.get("correctness") for r in answerable_rows]),
        "abstention_accuracy": _mean([float(r["abstained"]) for r in results if "abstained" in r]),
        "llm_model": llm.LLM_MODEL if use_llm else None,
    }

    if retrieval_only:
        report["note"] = "Retrieval-only run: answers and LLM-as-judge metrics were skipped."
    elif not use_llm:
        report["note"] = (
            "No LLM is configured (OPENAI_API_KEY is empty), so only retrieval metrics were computed. "
            "Set the LLM settings in .env to enable answers and the LLM-as-judge metrics."
        )

    errors = sum("error" in r for r in results)
    if errors:
        report["errors"] = errors

    report["results"] = results
    return report


def _print_markdown(report):
    def fmt(value):
        return "n/a" if value is None else f"{value:.2f}"

    print("| Metric | Score |")
    print("| --- | --- |")
    for key, label in [
        ("hit_rate", f"Hit Rate@{report['top_k']}"),
        ("mrr", "MRR"),
        ("faithfulness", "Faithfulness"),
        ("answer_relevance", "Answer Relevance"),
        ("answer_correctness", "Answer Correctness"),
        ("abstention_accuracy", "Abstention Accuracy (out-of-scope)"),
    ]:
        print(f"| {label} | {fmt(report[key])} |")

    print()
    print("| # | Question | Hit rank | Faithfulness | Relevance | Correctness |")
    print("| --- | --- | --- | --- | --- | --- |")
    for r in report["results"]:
        rank = "out of scope" if not r["answerable"] else (r["hit_rank"] or "miss")
        print(
            f"| {r['id']} | {r['question']} | {rank} | {fmt(r.get('faithfulness'))} "
            f"| {fmt(r.get('answer_relevance'))} | {fmt(r.get('correctness'))} |"
        )

    if report.get("note"):
        print(f"\nNote: {report['note']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG evaluation suite.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="print the full JSON report")
    parser.add_argument("--retrieval-only", action="store_true", help="skip LLM answers and judging")
    args = parser.parse_args()

    report = run_evaluation(args.top_k, args.retrieval_only)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_markdown(report)
