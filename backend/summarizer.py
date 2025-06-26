from transformers import pipeline
from urllib.parse import urlparse

# Load once when FastAPI starts
# ✅ Define summarizer pipeline
summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")  # or smaller model

# ✅ Define zero-shot classifier
classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")


def summarize_text(text):
    summary = summarizer_pipeline(text, max_length=150, min_length=40, do_sample=False)
    return summary[0]['summary_text']
# Optional: known source-level biases
SOURCE_BIAS = {
    "cnn.com": "Left",
    "foxnews.com": "Right",
    "reuters.com": "Neutral",
    "nytimes.com": "Left",
    "wsj.com": "Right",
    "bbc.com": "Neutral"
}

def get_domain_bias(url: str) -> str:
    try:
        domain = urlparse(url).netloc.replace("www.", "")
        return SOURCE_BIAS.get(domain, "Unknown")
    except:
        return "Unknown"

def get_bias_score(text: str) -> str:
    candidate_labels = ["Democratic", "Republican", "Neutral"]
    result = classifier(text, candidate_labels=candidate_labels)
    return result["labels"][0]  # Most likely label