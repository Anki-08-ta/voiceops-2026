
import os
from dotenv import load_dotenv

# Load environment variables from the project's .env file
load_dotenv()

# Read the Gemini API key without displaying it
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Read the Gnani API key for the later voice stage
GNANI_API_KEY = os.getenv("GNANI_API_KEY")


def validate_gemini_key():
    """Check that the Gemini API key exists."""
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is missing. Check your project's .env file."
        )
