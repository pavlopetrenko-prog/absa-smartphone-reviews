import pandas as pd
import gcsfs

bucket_path = "sent_proj"
fs = gcsfs.GCSFileSystem(token="google_default")

all_files = fs.ls(bucket_path)
task_files = [
    f"gs://{file}"
    for file in all_files
    if "results_task_" in file and file.endswith(".csv")
]

for file_path in task_files:
    # Remove 'gs://' prefix for gcsfs
    path_without_prefix = file_path.replace("gs://", "")
    fs.rm(path_without_prefix)
    print(f"Deleted {file_path}")
