
import argparse
import time
from pathlib import Path

import pandas as pd
from google import genai
from google.genai.errors import APIError

from src.config import GEMINI_API_KEY, validate_gemini_key


MODEL_NAME = "gemini-3.8-flash"

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
OUTPUT = OUTPUT_DIR / "step1_baseline.csv"

SYSTEM_INSTRUCTION = """
You are the AI customer-support assistant for ZentroMart,
an online store in India selling groceries and electronics.

Be polite, clear, and concise.
Do not invent company policies, order details, or delivery dates.
If you do not know an answer, say so honestly and offer help
from a human colleague when appropriate.
You cannot access customer orders or create support tickets.
At the beginning of a new conversation, identify yourself as an AI.
"""


def generate_with_retry(client, question, attempts=3):
    """Call the baseline LLM and retry temporary API errors."""
    for attempt in range(attempts):
        try:
            return client.interactions.create(
                model=MODEL_NAME,
                input=question,
                system_instruction=SYSTEM_INSTRUCTION,
            )
        except APIError as error:
            code = getattr(error, "code", None)

            if code not in (429, 500, 502, 503, 504):
                raise

            if attempt == attempts - 1:
                raise

            wait_seconds = 2 ** (attempt + 1)
            print(
                f"Temporary API error ({code}). "
                f"Retrying in {wait_seconds} seconds..."
            )
            time.sleep(wait_seconds)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the Step 1 ZentroMart baseline."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of questions to evaluate; use 0 for all questions.",
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be 0 or a positive integer.")

    validate_gemini_key()

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Practice questions CSV not found: {DATASET}"
        )

    questions = pd.read_csv(DATASET, dtype=str).fillna("")

    required_columns = [
        "id",
        "question",
        "expected_decision",
        "answerable",
        "gold_evidence",
        "reference_answer",
    ]
    missing = [
        column for column in required_columns
        if column not in questions.columns
    ]
    if missing:
        raise ValueError(
            f"questions.csv is missing required columns: {missing}"
        )

    if args.limit > 0:
        questions = questions.head(args.limit)

    if questions.empty:
        print("No questions found to evaluate.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=GEMINI_API_KEY)
    results = []

    total = len(questions)
    print(f"Running Step 1 baseline on {total} question(s).")
    print("This baseline does not use retrieval.\n")

    for index, (_, row) in enumerate(
        questions.iterrows(), start=1
    ):
        question_id = row["id"]
        question = row["question"]

        print(f"[{index}/{total}] Question ID: {question_id}")
        print(f"Question: {question}")

        try:
            response = generate_with_retry(client, question)
            answer = response.output_text or ""
            status = "success"
            error_message = ""
        except Exception as error:
            answer = ""
            status = "error"
            error_message = f"{type(error).__name__}: {error}"
            print(f"Error: {error_message}")

        results.append({
            "id": question_id,
            "question": question,
            "expected_decision": row["expected_decision"],
            "answerable": row["answerable"],
            "gold_evidence": row["gold_evidence"],
            "reference_answer": row["reference_answer"],
            "generated_answer": answer,
            "status": status,
            "error_message": error_message,
        })

        if answer:
            print(f"Answer: {answer}")
        print("-" * 60)

    results_df = pd.DataFrame(results)

    # Preserve earlier results if a new run has errors.
    # Each run writes a complete new result file.
    results_df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    successful = sum(item["status"] == "success" for item in results)
    failed = total - successful

    print("\nEvaluation finished.")
    print(f"Questions attempted: {total}")
    print(f"Successful responses: {successful}")
    print(f"Failed responses: {failed}")
    print(f"Saved results to: {OUTPUT}")


if __name__ == "__main__":
    main()
