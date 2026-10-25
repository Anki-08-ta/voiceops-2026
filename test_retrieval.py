
from pathlib import Path
import pandas as pd

from src.retriever import search_documents

PROJECT_ROOT = Path(__file__).resolve().parent.parent

QUESTIONS_CSV = (
    PROJECT_ROOT
    / "data"
    / "voiceops_dataset"
    / "Capstone_Project_Dataset_FTTT_2026"
    / "practice"
    / "questions.csv"
)


def parse_gold_sections(gold_evidence):
    """Extract expected section IDs, excluding product-table evidence."""
    if not gold_evidence.strip():
        return set()

    sections = set()

    for item in gold_evidence.split(";"):
        item = item.strip()

        if not item or item.lower().startswith("products.csv:"):
            continue

        sections.add(item)

    return sections


def main():
    questions = pd.read_csv(
        QUESTIONS_CSV, dtype=str
    ).fillna("")

    total = 0
    answerable = 0
    no_gold = 0
    hit_count = 0
    recall_sum = 0.0
    misses = []

    for _, row in questions.iterrows():
        question_id = row["id"]
        question = row["question"]
        gold = parse_gold_sections(row["gold_evidence"])

        results = search_documents(question, top_k=3)

        retrieved_ids = {
            result["section_id"]
            for result in results
            if result["section_id"]
        }

        total += 1

        if not gold:
            no_gold += 1

            # A question with no annotated evidence should not
            # retrieve a section unless the dataset annotation
            # or the retrieval policy requires further review.
            if results:
                misses.append(
                    (
                        question_id,
                        "NO GOLD EVIDENCE",
                        ", ".join(sorted(retrieved_ids)),
                    )
                )
            continue

        answerable += 1
        relevant_found = gold & retrieved_ids

        if relevant_found:
            hit_count += 1
        else:
            misses.append(
                (
                    question_id,
                    "MISSED GOLD SECTION",
                    ", ".join(sorted(retrieved_ids)),
                )
            )

        recall_sum += len(relevant_found) / len(gold)

    print("\n" + "=" * 65)
    print("RETRIEVAL EVALUATION")
    print("=" * 65)
    print(f"Total questions: {total}")
    print(f"Questions with gold sections: {answerable}")
    print(f"Questions without gold evidence: {no_gold}")

    if answerable:
        hit_at_3 = hit_count / answerable
        recall_at_3 = recall_sum / answerable

        print(f"Hit@3:    {hit_count}/{answerable} = {hit_at_3:.1%}")
        print(f"Recall@3:  {recall_at_3:.1%}")

    print(f"\nQuestions to review: {len(misses)}")

    for question_id, reason, retrieved in misses:
        print(f"\n{question_id} | {reason}")
        print(f"Retrieved section IDs: {retrieved or '(none)'}")

    print("\nEvaluation complete. No Gemini API calls were made.")


if __name__ == "__main__":
    main()
