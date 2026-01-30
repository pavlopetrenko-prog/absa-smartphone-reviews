import json
import logging
import math
import os
import sys
import threading
import time
from collections import deque
import pandas as pd

import asyncio
import nest_asyncio
from jsonschema import validate, ValidationError
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

import google.generativeai as genai

###############################################################
api_key = os.environ.get("MY_ENV_VAR")
if not api_key:
    raise ValueError("API key not found in environment variable MY_ENV_VAR")
print("✅ Secret key accessed successfully")
genai.configure(api_key=api_key)
generation_config = {
    "response_mime_type": "application/json",
}
model = genai.GenerativeModel("gemini-2.5-pro", generation_config=generation_config)

###############################################################
task_index = int(os.getenv("CLOUD_RUN_TASK_INDEX", 0))  # e.g. 0,1,2...
task_count = int(os.getenv("CLOUD_RUN_TASK_COUNT", 1))  # total number of tasks

# 2. Load your dataset (for example, from GCS or local volume)
df = pd.read_csv("gs://sent_proj/review_data.csv")
df = df.head(10000)


# 3. Split dataframe into N roughly equal chunks
def split_dataframe(df, n_chunks):
    chunk_size = math.ceil(len(df) / n_chunks)
    return [df[i * chunk_size : (i + 1) * chunk_size] for i in range(n_chunks)]


chunks = split_dataframe(df, task_count)

# 4. Pick the chunk for *this* task
my_chunk = chunks[task_index]
df = my_chunk
