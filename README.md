# Assignment 12 — Document AI RAG Assistant

A Streamlit-based Retrieval-Augmented Generation (RAG) application for Data Science and AI Batch-06.

## RAG Pipeline

PDF → Load → Chunk → Embed → Store → Retrieve → Generate

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Enter your Groq API key in the sidebar.

## Deployment

Deploy `app.py` on Streamlit Community Cloud.

Add the secret:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

Do not upload your API key to GitHub.
