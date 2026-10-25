
import re
from pathlib import Path

import pandas as pd
from google import genai

from src.config import validate_gemini_key, GEMINI_API_KEY
from src.retriever import search_documents


MODEL_NAME = "gemini-3.8-flash"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "voiceops_dataset"
    / "Capstone_Project_Dataset_FTTT_2026"
)

PRODUCTS_CSV = DATASET_ROOT / "data" / "products.csv"


SYSTEM_INSTRUCTION = """
You are the AI customer-support assistant for ZentroMart,
an online store in India selling groceries and electronics.

Rules:
1. Be polite, clear, and concise.
2. Answer using only the supplied product data and policy evidence.
3. Do not invent product details, company policies, or delivery dates.
4. If the evidence does not answer the question, say so honestly.
5. Never claim an action was completed unless a real tool confirms it.
6. At the beginning of a new conversation, identify yourself as an AI.
7. Do not assume that a general policy applies to a product unless
   the supplied product data supports that conclusion.
8. If product data and policy evidence are both supplied, use both
   when they are relevant to the question.
"""


def extract_skus(question):
    """Find SKU-like identifiers such as ELE-0012 in a question."""
    return list(
        dict.fromkeys(
            re.findall(r"\b[A-Z]{2,}-\d+\b", question.upper())
        )
    )


def lookup_products(skus, question=""):
    """Look up exact product rows by SKU or product name."""
    if not PRODUCTS_CSV.exists():
        raise FileNotFoundError(
            f"Products CSV not found: {PRODUCTS_CSV}"
        )

    products = pd.read_csv(PRODUCTS_CSV, dtype=str).fillna("")

    # First, look up exact SKUs when the question contains them.
    matches = products[
        products["sku"].str.upper().isin(skus)
    ]

    if not matches.empty:
        return matches.to_dict(orient="records")

    # If no SKU matched, look for a distinctive product name.
    question_lower = question.lower()
    name_matches = products[
        products["title"].str.lower().apply(
            lambda title: bool(title) and title in question_lower
        )
    ]

    return name_matches.to_dict(orient="records")


def answer_question(question, top_k=4):
    """Answer using retrieved policy sections and exact product records."""
    validate_gemini_key()

    # Step A: Retrieve relevant current knowledge-base sections.
    documents = search_documents(question, top_k=top_k)

    evidence_parts = []

    for doc in documents:
        evidence_parts.append(
            f"Source file: {doc['filename']}\n"
            f"Section ID: {doc['section_id']}\n"
            f"Section title: {doc['section_title']}\n"
            f"Content:\n{doc['text']}"
        )

    # Step B: Look up any exact SKU mentioned by the customer.
    skus = extract_skus(question)
    products = lookup_products(skus, question)

    for product in products:
        product_text = "\n".join(
            f"{key}: {value}"
            for key, value in product.items()
        )

        evidence_parts.append(
            "Source: products.csv\n"
            f"Exact product record:\n{product_text}"
        )

    evidence = "\n\n---\n\n".join(evidence_parts)

    if not evidence:
        evidence = (
            "No relevant current policy evidence or exact product "
            "record was found."
        )

    # Step C: Ask Gemini to answer from the supplied evidence.
    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.interactions.create(
        model=MODEL_NAME,
        input=(
            f"Customer question:\n{question}\n\n"
            f"Evidence:\n{evidence}\n\n"
            "Answer the customer using only evidence that directly "
            "supports the answer. For a product-specific question, "
            "use the exact product record if available and combine "
            "it with the relevant policy. If the evidence is missing "
            "or insufficient, explain what cannot be confirmed. "
            "Do not guess."
        ),
        system_instruction=SYSTEM_INSTRUCTION,
    )

    return {
        "question": question,
        "answer": response.output_text or "",
        "evidence": [
            {
                "filename": doc["filename"],
                "section_id": doc["section_id"],
                "section_title": doc["section_title"],
                "score": doc["score"],
            }
            for doc in documents
        ],
        "products": [
            {
                "sku": product.get("sku", ""),
                "title": product.get("title", ""),
                "assured": product.get("assured", ""),
                "warranty_months": product.get("warranty_months", ""),
            }
            for product in products
        ],
    }


if __name__ == "__main__":
    test_questions = [
        "How fast are refunds to the ZentroMart Wallet?",
        "Does SKU ELE-0012 have ZentroMart Assured warranty?",
        "Can I schedule a delivery at midnight?",
        "Does SKU UNKNOWN-9999 have a warranty?",
    ]

    for question in test_questions:
        print("\n" + "=" * 65)
        print(f"Question: {question}")

        try:
            result = answer_question(question)

            print(f"\nAnswer: {result['answer']}")

            print("\nRetrieved policy evidence:")
            if not result["evidence"]:
                print("No policy sections retrieved.")

            for item in result["evidence"]:
                print(
                    f"- {item['filename']} | "
                    f"{item['section_id']} | "
                    f"{item['section_title']}"
                )

            print("\nProduct lookup:")
            if not result["products"]:
                print("No matching product record found.")

            for product in result["products"]:
                print(
                    f"- SKU: {product['sku']} | "
                    f"Title: {product['title']} | "
                    f"Assured: {product['assured']} | "
                    f"Warranty months: {product['warranty_months']}"
                )

        except Exception as error:
            print(f"\nError: {error}")
            print("Share the complete error message for debugging.")
