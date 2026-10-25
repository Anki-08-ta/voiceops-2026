
import streamlit as st
from pathlib import Path
from src.voice import answer_by_voice

# Project output folder
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="ZentroMart VoiceOps",
    page_icon="🎧",
    layout="centered",
)

st.title("🎧 ZentroMart VoiceOps")
st.caption("AI-powered customer support using Gemini and Gnani")

st.write(
    "Upload a customer's voice recording. The system will transcribe "
    "it, generate an evidence-grounded answer, and speak the response."
)

uploaded_audio = st.file_uploader(
    "Upload customer audio",
    type=["wav", "mp3", "m4a", "ogg", "flac"],
)

if st.button("Process voice query", type="primary", disabled=uploaded_audio is None):
    input_path = None

    try:
        extension = Path(uploaded_audio.name).suffix.lower() or ".wav"
        input_path = OUTPUT_DIR / f"customer_upload{extension}"
        response_path = OUTPUT_DIR / "voiceops_response.wav"

        input_path.write_bytes(uploaded_audio.getvalue())

        with st.spinner("Transcribing audio and generating the response..."):
            result = answer_by_voice(
                str(input_path),
                str(response_path),
            )

        st.success("Voice query processed successfully!")

        st.subheader("1. Customer transcript")
        st.write(result["transcript"])

        st.subheader("2. AI customer-support answer")
        st.write(result["answer"])

        st.subheader("3. Spoken response")
        if Path(result["audio_path"]).is_file():
            st.audio(
                Path(result["audio_path"]).read_bytes(),
                format="audio/wav",
            )
        else:
            st.warning("The response audio file was not found.")

        st.subheader("4. Retrieved evidence")
        evidence = result.get("evidence", [])

        if evidence:
            for item in evidence:
                st.markdown(
                    f"- **{item['section_id']}** — "
                    f"{item['section_title']} "
                    f"(`{item['filename']}`)"
                )
        else:
            st.warning("No supporting policy sections were retrieved.")

    except Exception as exc:
        st.error(f"Could not process the voice query: {exc}")
        st.info(
            "Check your API configuration, audio format, and terminal "
            "output before trying again."
        )

    finally:
        if input_path is not None and input_path.exists():
            input_path.unlink(missing_ok=True)

st.divider()
st.caption("Prototype for the VoiceOps 2026 capstone.")
