import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import f1_score, accuracy_score

df = pd.read_csv(
    "/Users/macos/Documents/sentiment_project/validation/validation_sample_mapped.csv"
)


def infer_true_sentiment(row, sentiment_col, validate_col):
    if row[validate_col] == 1:
        return row[sentiment_col]
    else:
        return "negative" if row[sentiment_col] == "positive" else "positive"


# For aspect sentiment
df["aspect_sentiment_true"] = df.apply(
    infer_true_sentiment,
    axis=1,
    sentiment_col="aspect_sentiment",
    validate_col="aspect_sentiment_validate",
)

# For overall sentiment
df["overall_sentiment_true"] = df.apply(
    infer_true_sentiment,
    axis=1,
    sentiment_col="overall_sentiment",
    validate_col="overall_sentiment_validate",
)


df["high_category_true"] = df.apply(
    lambda row: row["high_category"] if row["high_category_validate"] == 1 else "",
    axis=1,
)

df["main_category_true"] = df.apply(
    lambda row: row["main_category"] if row["main_category_validate"] == 1 else "",
    axis=1,
)

# MANUAL MAPPING TRUE VALUES WITH VALIDATION = 0

# RE-READ MANUALLY MAPPED DATASET
df = pd.read_csv(
    "/Users/macos/Documents/sentiment_project/validation/validation_sample_mapped.csv"
)


column_order = [
    "row_id",
    "review_id",  # identifiers first
    # Overall sentiment
    "overall_sentiment",
    "overall_sentiment_validate",
    "overall_sentiment_true",
    # Aspect sentiment
    "aspect_sentiment",
    "aspect_sentiment_validate",
    "aspect_sentiment_true",
    # Aspect summary (textual)
    "aspect_summary",
    "aspect_summary_validate",
    # Category hierarchy
    "high_category",
    "high_category_validate",
    "high_category_true",
    "main_category",
    "main_category_validate",
    "main_category_true",
]

df = df[column_order]


df.to_csv(
    "/Users/macos/Documents/sentiment_project/validation/validation_sample_mapped.csv",
    index=False,
)


id_cols = ["row_id", "review_id"]

pred_true_pairs = [
    ("overall_sentiment", "overall_sentiment_true"),
    ("aspect_sentiment", "aspect_sentiment_true"),
    ("high_category", "high_category_true"),
    ("main_category", "main_category_true"),
]

validate_cols = [
    "overall_sentiment_validate",
    "aspect_sentiment_validate",
    "aspect_summary_validate",
    "high_category_validate",
    "main_category_validate",
]

required = (
    id_cols
    + [p for p, _ in pred_true_pairs]
    + [t for _, t in pred_true_pairs]
    + validate_cols
    + ["aspect_summary"]
)
missing = [c for c in required if c not in df.columns]
assert not missing, f"Missing columns: {missing}"

dups = df.duplicated(subset=id_cols, keep=False).sum()
print(f"Duplicate rows by {id_cols}: {dups}")
print(df[required].isna().sum().sort_values(ascending=False).head(20))


for c in validate_cols:
    assert set(df[c].dropna().unique()) <= {0, 1}, f"{c} has values outside 0/1"
    df[c] = df[c].astype("int8")

checks = []
for pred, true in pred_true_pairs:
    vcol = f"{pred}_validate"
    if vcol in df.columns:
        ok1 = ((df[vcol] == 1) & (df[pred] != df[true])).sum() == 0
        ok0 = ((df[vcol] == 0) & (df[pred] == df[true])).sum() == 0
        checks.append((pred, ok1, ok0))
for pred, ok1, ok0 in checks:
    assert ok1 and ok0, f"Inconsistent validate logic for {pred}"
print("✅ Validate ↔ pred/true consistency passed.")


acc_table = df[validate_cols].mean().rename("accuracy").to_frame()
print(acc_table)

for pred, true in pred_true_pairs:
    print(f"\nClass distribution (TRUE) for {true}:")
    print(df[true].value_counts(dropna=False).to_frame("count"))


def report_task(pred_col, true_col, title=None):
    y_true = df[true_col].astype(str)
    y_pred = df[pred_col].astype(str)
    print("\n" + (title or f"{true_col} vs {pred_col}"))
    print(classification_report(y_true, y_pred, digits=3))
    labels = sorted(pd.unique(y_true))
    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=pd.Index(labels, name="TRUE"),
        columns=pd.Index(labels, name="PRED"),
    )
    return cm


cms = {}
for pred, true in pred_true_pairs:
    cms[pred] = report_task(pred, true, title=f"== {pred.upper()} ==")

# Example: inspect confusion matrix for main_category
cms["main_category"].head()


by_high = df.groupby("high_category_true")[validate_cols].mean().sort_index()
by_main = (
    df.groupby("main_category_true")[validate_cols]
    .mean()
    .sort_values("main_category_validate")
)

print("\nAccuracy by high_category_true:")
print(by_high)

print("\nAccuracy by main_category_true (sorted by main_category_validate):")
print(by_main)


def per_class_f1(pred_col, true_col):
    y_true = df[true_col].astype(str)
    y_pred = df[pred_col].astype(str)
    labels = sorted(pd.unique(y_true))
    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    out = pd.DataFrame(
        {"precision": p, "recall": r, "f1": f1, "support": s}, index=labels
    )
    return out.sort_values("f1")


print("\nPer-class F1 for main_category:")
print(per_class_f1("main_category", "main_category_true"))


majority_aspect_true = (
    df.groupby("review_id")["aspect_sentiment_true"]
    .agg(lambda s: s.value_counts().idxmax())
    .rename("aspect_majority_true")
)

overall_true = df.drop_duplicates("review_id").set_index("review_id")[
    "overall_sentiment_true"
]

# Align both Series by their review_id index
common_ids = majority_aspect_true.index.intersection(overall_true.index)

agree = (majority_aspect_true.loc[common_ids] == overall_true.loc[common_ids]).mean()

print(f"\nAgreement: majority aspect TRUE vs overall TRUE = {agree:.3f}")


corr = df[validate_cols].corr()
print("\nValidation flag correlations:")
print(corr.round(3))


any_fail = (
    df[
        [
            "aspect_sentiment_validate",
            "high_category_validate",
            "main_category_validate",
        ]
    ]
    == 0
).any(axis=1)
errors = df[any_fail].copy()

# 7.2 Show the most frequent TRUE classes with low F1
weak_main = per_class_f1("main_category", "main_category_true").head(5).index.tolist()
errors_weak_main = errors[errors["main_category_true"].isin(weak_main)]

# 7.3 Sample some rows to read
cols_show = [
    "row_id",
    "review_id",
    "aspect_summary",
    "aspect_summary_validate",
    "aspect_sentiment",
    "aspect_sentiment_true",
    "aspect_sentiment_validate",
    "high_category",
    "high_category_true",
    "high_category_validate",
    "main_category",
    "main_category_true",
    "main_category_validate",
]
sample_to_review = errors_weak_main.sample(
    min(20, len(errors_weak_main)), random_state=42
)[cols_show]
sample_to_review.head(10)


def stratified_macro_f1(pred_col, true_col, slice_col):
    out = []
    for grp, part in df.groupby(slice_col):
        y_true = part[true_col].astype(str)
        y_pred = part[pred_col].astype(str)
        score = f1_score(y_true, y_pred, average="macro", zero_division=0)
        out.append((grp, score, len(part)))
    return pd.DataFrame(out, columns=[slice_col, "macro_f1", "n"]).sort_values(
        "macro_f1"
    )


print("\nMacro F1 for main_category per high_category_true:")
print(stratified_macro_f1("main_category", "main_category_true", "high_category_true"))


summary_rows = []
for pred, true in pred_true_pairs:
    # accuracy from validate flag (safe & quick)
    v = f"{pred}_validate"
    acc = df[v].mean() if v in df.columns else np.nan

    y_true = df[true].astype(str)
    y_pred = df[pred].astype(str)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    summary_rows.append(
        {
            "task": pred,
            "accuracy_validate": round(float(acc), 4),
            "macro_f1": round(float(macro), 4),
            "weighted_f1": round(float(weighted), 4),
            "n": len(df),
        }
    )

summary = (
    pd.DataFrame(summary_rows)
    .set_index("task")
    .sort_values("macro_f1", ascending=False)
)
print("\n=== Executive Summary ===")
print(summary)
