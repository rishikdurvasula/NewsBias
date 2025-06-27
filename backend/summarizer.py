from transformers import pipeline
from urllib.parse import urlparse

# ✅ Define the summarizer correctly
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# ✅ Known source-level biases
SOURCE_BIAS = {
    "cnn.com": "Left",
    "foxnews.com": "Right",
    "reuters.com": "Neutral",
    "nytimes.com": "Left",
    "wsj.com": "Right",
    "bbc.com": "Neutral"
}

# ✅ Generate summary
def generate_summary(text: str) -> str:
    if len(text.split()) > 1024:
        text = ' '.join(text.split()[:1024])
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# ✅ Infer bias from domain
def get_domain_bias(url: str) -> str:
    try:
        domain = urlparse(url).netloc.replace("www.", "")
        return SOURCE_BIAS.get(domain, "Unknown")
    except:
        return "Unknown"
