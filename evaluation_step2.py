
import argparse
from pathlib import Path

import pandas as pd

from src.agent import answer_question


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET = (
    PROJECT_ROOT
    / "data"
    / "voiceops_dataset"
    / "Capstone_Project_Dataset_FTTT_2026"
    / "practice"
    / "questions.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "sample_outputs"
OUTPUT = OUTPUT_DIR / "step2_rag.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the Step 2 RAG agent."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Questions to evaluate; use 0 for all questions.",
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be 0 or a positive integer.")

    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    questions = pd.read_csv(DATASET, dtype=str).fillna("")

    if args.limit > 0:
        questions = questions.head(args.limit)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    total = len(questions)
    print(f"Running Step 2 RAG evaluation on {total} question(s).")
    print("This test calls src.agent.answer_question().\n")

    for index, (_, row) in enumerate(questions.iterrows(), start=1):
        question_id = row["id"]
        question = row["question"]

        print(f"[{index}/{total}] Question ID: {question_id}")
        print(f"Question: {question}")

        try:
            result = answer_question(question)
            answer = result.get("answer", "")
            evidence = result.get("evidence", [])
            products = result.get("products", [])

            if not answer:
                answer = str(result)

            status = "success"
            error_message = ""

            print(f"Answer: {answer}")
            print(f"Evidence returned: {len(evidence)}")
            print(f"Product records returned: {len(products)}")

        except Exception as error:
            answer = ""
            evidence = []
            products = []
            status = "error"
            error_message = f"{type(error).__name__}: {error}"
            print(f"Error: {error_message}")

        results.append({
            "id": question_id,
            "question": question,
            "expected_decision": row.get("expected_decision", ""),
            "answerable": row.get("answerable", ""),
            "gold_evidence": row.get("gold_evidence", ""),
            "reference_answer": row.get("reference_answer", ""),
            "generated_answer": answer,
            "evidence_count": len(evidence),
            "product_count": len(products),
            "status": status,
            "error_message": error_message,
        })

        print("-" * 60)

    pd.DataFrame(results).to_csv(
        OUTPUT, index=False, encoding="utf-8-sig"
    )

    successful = sum(r["status"] == "success" for r in results)
    print("\nStep 2 evaluation finished.")
    print(f"Questions attempted: {total}")
    print(f"Successful responses: {successful}")
    print(f"Failed responses: {total - successful}")
    print(f"Saved results to: {OUTPUT}")


if __name__ == "__main__":
    main()
