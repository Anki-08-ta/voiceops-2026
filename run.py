
from google import genai
from src.config import validate_gemini_key, GEMINI_API_KEY


SYSTEM_INSTRUCTION = """
You are the AI customer-support assistant for ZentroMart,
an online store in India selling groceries and electronics.

Rules:
1. At the beginning of a new conversation, clearly identify
   yourself as an AI customer-support assistant.
2. Be polite, clear, and helpful.
3. Keep responses concise and easy to understand.
4. Do not invent order details, delivery dates, refund eligibility,
   or company policies.
5. In this first version, you cannot access customer orders,
   create support tickets, or perform account actions.
6. If you do not know the answer, say so honestly.
7. Do not claim that an action has been completed when you
   have not performed it.
"""


def main():
    validate_gemini_key()

    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.interactions.create(
        model="gemini-3.8-flash",
        input=(
            "A customer says: Hello! Who am I speaking with? "
            "Please respond as the customer-support assistant."
        ),
        system_instruction=SYSTEM_INSTRUCTION,
    )

    print(response.output_text)


if __name__ == "__main__":
    main()
