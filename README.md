# PDF Clinical Summarizer

This project provides a Streamlit-powered web interface to parse PDF files and generate structured clinical summaries using OpenAI models (via LiteLLM). It includes end-to-end evaluation with ROUGE and hallucination-free scorers, plus an interactive feedback system (thumbs up/down and notes) logged through Weights & Biases Weave.

---

## Setup

Create a `.env` file with your API keys:

```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
WANDB_API_KEY=your_wandb_api_key
WANDB_ENTITY=your_wandb_team_name
```

Install dependencies (recommended: use a conda environment):

```bash
conda activate pdf_summarizer_env
pip install -r requirements.txt
```

---

## File Descriptions

### `model.py`
Contains core model classes and utility functions for PDF parsing, LLM chat interaction, and summarization logic. Defines the `ChatModel`, `AuthoringModel`, and related Weave Ops used by both the Streamlit app and evaluation script.

**How to use:**  
This file is imported by `streamlit.py` and `evaluation.py` and is not meant to be run directly.

---

### `streamlit.py`
A Streamlit web application for uploading PDF files, generating clinical summaries using LLMs, and collecting user feedback. The app parses PDFs, summarizes their content with a configurable system prompt, and displays results with interactive feedback buttons.

**How to run:**
```bash
streamlit run streamlit.py
```

---

### `evaluation.py`
A script for automated evaluation of different LLM summarization models on a clinical dataset. It uses Weave's evaluation framework, computes ROUGE-L scores, and leverages hallucination and summarization scorers. Results are logged for comparison across models.

**How to run:**
```bash
python evaluation.py
```

---

## Notes

- Ensure you have the proper API keys set up if you're evaluating both OpenAI and Anthropic models.
- All scripts require dependencies listed in `requirements.txt` and are best run inside the provided conda environment.