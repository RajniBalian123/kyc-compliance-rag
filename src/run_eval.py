"""Run the golden-question eval set and report recall@5.

For each question, the check is: does the expected regulatory category appear in the
top 5 retrieved chunks, and do the expected key phrases appear in the retrieved text?
Retrieval quality is measured, not assumed.

Usage:
    python -m src.run_eval
"""

import json

from src.config import EVALS_FILE
from src.vector_store import query_rules


def main() -> None:
    questions = json.loads(EVALS_FILE.read_text(encoding="utf-8"))
    passed = 0
    print(f"{'Question':60s} {'Category':9s} {'Phrases':7s}")
    print("-" * 80)
    for item in questions:
        hits = query_rules(item["question"])
        categories = {h["rule_category"] for h in hits}
        joined = " ".join(h["text"] for h in hits).lower()
        category_ok = item["expected_category"] in categories
        phrases_ok = all(p.lower() in joined for p in item["expected_phrases"])
        ok = category_ok and phrases_ok
        passed += int(ok)
        print(f"{item['question'][:58]:60s} "
              f"{'PASS' if category_ok else 'FAIL':9s} "
              f"{'PASS' if phrases_ok else 'FAIL':7s}")
    print("-" * 80)
    print(f"{passed}/{len(questions)} questions fully recalled in the top 5.")


if __name__ == "__main__":
    main()
