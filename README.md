# VoiceOps 2026 — ZentroMart Voice Assistant

## Overview

VoiceOps is a voice-enabled customer-support assistant for ZentroMart, an online store in India. It answers customer questions using retrieved knowledge-base policies and product catalogue information.

The project combines retrieval-augmented generation (RAG), a Gemini language model, and Gnani speech services to support voice-based customer interactions.

## Features

- **Policy retrieval:** Retrieves relevant sections from the supplied knowledge base.
- **Product lookup:** Looks up exact product records by SKU.
- **Grounded responses:** Generates answers using retrieved evidence and product data.
- **Voice interaction:** Uses Gnani speech-to-text and text-to-speech services.
- **Web interface:** Provides a Streamlit interface for submitting audio and viewing responses.
- **Evaluation:** Includes baseline, RAG, and retrieval evaluation scripts.

## Project Structure

```text
voiceops-2026/
├── app.py
├── run.py
├── run.sh
├── requirements.txt
├── .env.example
├── src/
│   ├── agent.py
│   ├── config.py
│   ├── retriever.py
│   ├── voice.py
│   ├── evaluation.py
│   ├── evaluation_step2.py
│   └── test_retrieval.py
├── sample_outputs/
├── outputs/
└── README.md
```

*Verify the filenames against the final repository before submission.*

## Requirements

- Python 3.10 or a compatible version supported by the installed dependencies
- Gemini API access
- Gnani API credentials for speech services

## Setup

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file using `.env.example` as a template.
5. Add your own API credentials to `.env`. Never commit this file.

The supplied capstone dataset and knowledge-base files must be available at the paths expected by the application.

## Running the Application

Launch the Streamlit interface using the project's verified launch command. For example:

```bash
streamlit run app.py
```

Upload a customer audio recording, submit it, and review the transcribed question, generated response, and audio output.

## Evaluation

The project includes:

- Step 1: Baseline evaluation without retrieval.
- Step 2: RAG-based answer evaluation.
- Retrieval evaluation using the supplied gold-evidence annotations.

Evaluation results should be reported using the actual generated outputs. API-dependent evaluations may require available Gemini quota.

## Limitations

- Responses depend on the quality and relevance of the supplied evidence.
- Questions unsupported by the available policies or catalogue may not be answerable.
- API availability and rate limits can affect evaluation and runtime.
- Voice quality and transcription accuracy depend on the audio and speech-service responses.

## Security

- Keep API credentials in `.env`.
- Commit `.env.example`, not `.env`.
- Do not include virtual environments, API keys, or private credentials in the repository or submission ZIP.
- Follow the capstone rules regarding dataset redistribution.

## Author 

Ankita Kumari 
