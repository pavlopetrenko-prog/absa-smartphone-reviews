import pandas as pd
import math

df = pd.read_csv(
    "/Users/macos/Documents/sentiment_project/validation/main_category.csv"
)


def get_sample_size(df, confidence=0.95, margin_error=0.05, p=0.5, verbose=True):
    N = len(df)

    # z-scores for common confidence levels
    z_table = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_table.get(confidence, 1.96)

    # initial infinite-population sample size
    n0 = (z**2) * p * (1 - p) / (margin_error**2)
    # finite-population correction
    n = n0 / (1 + ((n0 - 1) / N))
    n = math.ceil(n)

    if verbose:
        print(f"📊 Dataset size: {N}")
        print(f"🔹 Confidence level: {int(confidence * 100)}%")
        print(f"🔹 Margin of error: ±{margin_error * 100}%")
        print(f"🔹 Estimated proportion (p): {p}")
        print(f"✅ Required sample size: {n}")

    return n


n = get_sample_size(df, confidence=0.95, margin_error=0.04)


df_sample = df.sample(n=n, random_state=42)


model_columns = [
    "overall_sentiment",
    "aspect_sentiment",
    "aspect_summary",
    "high_category",
    "main_category",
]

# Create empty validation columns with suffix "_validate"
for col in model_columns:
    df_sample[f"{col}_validate"] = ""

# Optional: reorder columns (model columns → validation columns)
ordered_cols = model_columns + [f"{col}_validate" for col in model_columns]
df_sample = df_sample[
    [c for c in df_sample.columns if c not in ordered_cols] + ordered_cols
]


df_sample.to_csv("validation_sample.csv", index=False)
