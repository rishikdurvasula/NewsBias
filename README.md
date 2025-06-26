# 📰 News Bias Analyzer

A full-stack AI-powered web application that **summarizes news articles** and detects their **political bias** (Left, Right, or Neutral) using large language models from Hugging Face.

## 🚀 Features

- 🔍 **URL or Text Input**: Paste a news article **URL** or raw **text**, and the app will automatically process it.
- 🧠 **LLM Summarization**: Uses `facebook/bart-large-cnn` to generate concise summaries of the content.
- 📊 **Bias Detection**: Employs zero-shot classification via `facebook/bart-large-mnli` or `joeddav/xlm-roberta-large-xnli` to classify political bias.
- 🌐 **Intuitive UI**: Built with Streamlit for a fast, responsive user experience.
- ⚡ **FastAPI Backend**: Powers the NLP inference pipeline through an efficient REST API.
- 📦 **Transformer Caching**: Supports disk-based caching for large models using `TRANSFORMERS_CACHE`.

---

## 🧰 Tech Stack

| Layer       | Technology                              |
|-------------|------------------------------------------|
| **Frontend** | Streamlit                               |
| **Backend**  | FastAPI                                 |
| **NLP Models** | Hugging Face Transformers (`BART`, `XLM-RoBERTa`) |
| **Web Scraping** | Newspaper3k                          |
| **Deployment** | Local (can be Dockerized/hosted)      |
| **Others**    | Requests, Python 3.9, Uvicorn, Git     |

---
