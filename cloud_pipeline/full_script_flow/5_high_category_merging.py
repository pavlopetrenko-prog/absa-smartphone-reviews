import pandas as pd
import gcsfs

bucket_path = "sent_proj"
fs = gcsfs.GCSFileSystem()

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


data_path = f"gs://{bucket_path}/initial_extraction.csv"
review_df = pd.read_csv(data_path)
final_df = review_df.merge(final_df, on="row_id", how="left")


final_df = final_df[
    [
        "row_id",
        "review_id",
        "overall_sentiment",
        "ground_truth_sentiment",
        "aspect_sentiment",
        "aspect_summary",
        "high_category",
    ]
]


# Save merged CSV back to GCS
final_df.to_csv(f"gs://{bucket_path}/high_category.csv", index=False)
print("✅ Saved final merged CSV to GCS")
