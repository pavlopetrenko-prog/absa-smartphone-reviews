import pandas as pd
import gcsfs

bucket_path = "sent_proj"
fs = gcsfs.GCSFileSystem(token="google_default")

# List all files in the bucket that match 'results_task_*.csv'
all_files = fs.ls(bucket_path)
task_files = [
    f"gs://{file}"
    for file in all_files
    if "results_task_" in file and file.endswith(".csv")
]

print(f"Found {len(task_files)} task files.")

# Read all CSVs directly from GCS
dfs = [pd.read_csv(f) for f in task_files]


# Concatenate all DataFrames
final_df = pd.concat(dfs, ignore_index=True)
print(f"Merged DataFrame shape: {final_df.shape}")

review_data_path = f"gs://{bucket_path}/review_data.csv"
review_df = pd.read_csv(review_data_path)[["review_id", "ground_truth_sentiment"]]
final_df = final_df.merge(review_df, on="review_id", how="left")


final_df = (
    final_df[
        [
            "review_id",
            "overall_sentiment",
            "ground_truth_sentiment",
            "aspect_sentiment",
            "aspect_summary",
        ]
    ]
    .sort_values("review_id")
    .reset_index(drop=True)
)

final_df["row_id"] = range(len(final_df))


final_df = (
    final_df[
        [
            "row_id",
            "review_id",
            "overall_sentiment",
            "ground_truth_sentiment",
            "aspect_sentiment",
            "aspect_summary",
        ]
    ]
    .sort_values("review_id")
    .reset_index(drop=True)
)

# Save merged CSV back to GCS
final_df.to_csv(f"gs://{bucket_path}/initial_extraction.csv", index=False)
