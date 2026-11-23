"""Ask the compliance corpus a question. Answers come from retrieved passages only.

Usage:
    python -m src.query "What due diligence is required for a politically exposed person?"

The output lists the top 5 retrieved passages with their regulatory category and
relevance score. Answer generation is deliberately left to the caller: plug any chat
model in front of these passages. If no passage is relevant, the correct behaviour is
to say so, not to answer from memory.
"""

import sys

from src.vector_store import ensure_ingested, query_rules


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m src.query "your question"')
        raise SystemExit(1)
    question = " ".join(sys.argv[1:])
    count = ensure_ingested()
    print(f"Corpus: {count} chunks indexed.\n")
    print(f"Question: {question}\n")
    hits = query_rules(question)
    for i, hit in enumerate(hits, start=1):
        print(f"[{i}] {hit['rule_category']} — {hit['source']} "
              f"(relevance {hit['relevance_score']})")
        print(f"    {hit['text']}\n")
    best = hits[0]["relevance_score"] if hits else 0.0
    if best < 0.3:
        print("No passage is strongly relevant — the honest answer is "
              "'not covered by the indexed rules'.")


if __name__ == "__main__":
    main()
