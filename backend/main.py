from fastapi import FastAPI
from pydantic import BaseModel
from summarizer import summarize_text, get_bias_score, get_domain_bias

app = FastAPI()

class Article(BaseModel):
    content: str
    source_url: str = ""  # Optional

@app.post("/analyze")
def analyze(article: Article):
    summary = summarize_text(article.content)
    bias = get_bias_score(article.content)

    # Optional: add source bias info
    source_bias = get_domain_bias(article.source_url)

    return {
        "summary": summary,
        "bias": bias,
        "source_bias": source_bias
    }
