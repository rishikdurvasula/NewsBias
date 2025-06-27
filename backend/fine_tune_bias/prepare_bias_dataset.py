import pandas as pd

# Load the raw Kaggle dataset
df = pd.read_csv("backend/fine_tune_bias/Political_Bias.csv")

# Clean up columns
df = df.rename(columns=str.strip)
df = df[["Text", "Bias"]].dropna()
df.columns = ["text", "label"]

# Normalize bias labels
def normalize_label(x):
    x = x.lower().strip()
    if "lean left" in x or x == "left":
        return "left"
    elif "lean right" in x or x == "right":
        return "right"
    elif "center" in x:
        return "center"
    else:
        return None

df["label"] = df["label"].map(normalize_label)
df = df[df["label"].notnull()]

# Remove very short texts
df = df[df["text"].str.len() > 200]

# Sample if too large
df = df.sample(n=min(2000, len(df)), random_state=42)

# Save cleaned dataset
df.to_csv("backend/fine_tune_bias/bias_dataset.csv", index=False)
print(f"✅ Saved cleaned dataset with {len(df)} rows.")
