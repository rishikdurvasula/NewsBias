from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

# Load data
df = pd.read_csv("backend/fine_tune_bias/bias_dataset.csv")
label2id = {"left": 0, "center": 1, "right": 2}
id2label = {v: k for k, v in label2id.items()}
df["label"] = df["label"].map(label2id)

# Tokenize
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
dataset = Dataset.from_pandas(df)
dataset = dataset.map(lambda e: tokenizer(e["text"], padding=True, truncation=True), batched=True)

# Train/test split
split = dataset.train_test_split(test_size=0.2)
train_ds, test_ds = split["train"], split["test"]

# Load model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

# Train config
training_args = TrainingArguments(
    output_dir="backend/fine_tune_bias/results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_total_limit=1,
)

# Train!
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds
)
trainer.train()

# Save model
model.save_pretrained("backend/fine_tune_bias/bias-classifier")
tokenizer.save_pretrained("backend/fine_tune_bias/bias-classifier")
print("✅ Model saved to backend/fine_tune_bias/bias-classifier")
