
from pathlib import Path

from gnani.stt.client import GnaniSTTClient
from gnani.tts.client import GnaniTTSClient

from src.config import GNANI_API_KEY
from src.agent import answer_question


def validate_gnani_key():
    """Ensure the Gnani API key is configured."""
    if not GNANI_API_KEY:
        raise ValueError(
            "GNANI_API_KEY is missing. Check your project's .env file."
        )


def transcribe_audio(audio_path):
    """
    Convert a customer's recorded speech into text using Gnani STT.
    """
    validate_gnani_key()

    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    client = GnaniSTTClient()
    result = client.transcribe(
        str(audio_path),
        language_code="en-IN",
        format="verbatim",
    )

    transcript = result.get("transcript", "").strip()

    if not transcript:
        raise ValueError(
            "Gnani STT returned an empty transcript."
        )

    return transcript


def synthesize_speech(text, output_path="outputs/response.wav"):
    """
    Convert the assistant's response into audio using Gnani TTS.
    """
    validate_gnani_key()

    if not text or not text.strip():
        raise ValueError("Cannot synthesize empty text.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = GnaniTTSClient()
    client.synthesize(
        text=text,
        voice="Pranav",
        output_file=str(output_path),
    )

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(
            "Gnani TTS did not produce a non-empty audio file."
        )

    return output_path


def answer_by_voice(audio_path, output_path="outputs/response.wav"):
    """
    Complete voice pipeline:
    customer audio -> transcript -> Gemini answer -> spoken response.
    """
    transcript = transcribe_audio(audio_path)
    result = answer_question(transcript)
    answer = result["answer"]
    audio_output = synthesize_speech(answer, output_path)

    return {
        "transcript": transcript,
        "answer": answer,
        "audio_path": str(audio_output),
        "evidence": result.get("evidence", []),
    }
