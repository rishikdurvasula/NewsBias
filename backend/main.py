from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
from summarizer import generate_summary  # assumes you already have this module

app = FastAPI()


from pathlib import Path
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Dynamically get the path to the model directory
MODEL_PATH = Path(__file__).parent / "fine_tune_bias" / "bias-classifier"

# Use string paths when calling Hugging Face APIs
tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODEL_PATH))
model = DistilBertForSequenceClassification.from_pretrained(str(MODEL_PATH))

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Input schema
class AnalyzeRequest(BaseModel):
    content: str
    source_url: str = ""

@app.post("/analyze")
def analyze_article(request: AnalyzeRequest):
    content = request.content

    # Tokenize and predict
    inputs = tokenizer(content, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    # Label mapping (ensure it matches your fine-tuning labels)
    label_map = {
        0: "left",
        1: "lean left",
        2: "center",
        3: "lean right",
        4: "right"
    }
    bias_label = label_map.get(prediction, "Unknown")

    # Get summary (calls your existing summarizer pipeline)
    summary = generate_summary(content)

    return {
        "summary": summary,
        "bias": bias_label,
        "source_bias": "Unknown"  # You can implement domain-based source bias detection later
    }
