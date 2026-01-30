import os
import math
import asyncio
import nest_asyncio
import json
import pandas as pd
import google.generativeai as genai
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from jsonschema import validate, ValidationError
import logging
import sys
import time
from collections import deque
import threading

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

logger = logging.getLogger("initial_extraction")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


call_timestamps = deque()
lock = threading.Lock()


def record_model_call():
    now = time.time()
    with lock:
        call_timestamps.append(now)
        # Keep only last 60 seconds of calls
        while call_timestamps and now - call_timestamps[0] > 60:
            call_timestamps.popleft()


def get_calls_last_minute():
    with lock:
        return len(call_timestamps)


def start_usage_logger():
    def log_usage():
        while True:
            time.sleep(60)
            cpm = get_calls_last_minute()
            logger.info(
                json.dumps(
                    {
                        "event": "usage_frequency",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "calls_last_minute": cpm,
                        "percent_of_limit": round(100 * cpm / MAX_CALLS_PER_MIN, 1),
                    }
                )
            )

    threading.Thread(target=log_usage, daemon=True).start()


# Set your limit manually or dynamically if known
MAX_CALLS_PER_MIN = 150
start_usage_logger()

###############################################################
# 1. Extract env vars from Cloud Run Job
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
###############################################################

feature_extraction_prompt = """
You are given reviews.  
For each review, return an entry where the **key is the review index** (starting from 0)  
and the value is the following JSON schema:

{{
  "overall_sentiment": "positive" | "negative",
  "aspects": [
    {{
      "sentiment": "positive" | "negative",
      "summary": "Brief, concrete quote from the review describing a feature, service, or problem."
    }}
  ]
}}

The output should be one JSON object with multiple keys (one per review).  

Reviews to analyze:
{reviews_text}
"""
###############################################################

review_schema = {
    "type": "object",
    "properties": {
        "overall_sentiment": {"type": "string", "enum": ["positive", "negative"]},
        "aspects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sentiment": {"type": "string", "enum": ["positive", "negative"]},
                    "summary": {"type": "string"},
                },
                "required": ["sentiment", "summary"],
            },
        },
    },
    "required": ["overall_sentiment", "aspects"],
}


def validate_schema_batch(resolved_output, schema):
    invalid = {}
    for rid, review_obj in resolved_output.items():
        try:
            validate(instance=review_obj, schema=schema)
        except ValidationError as e:
            invalid[rid] = str(e)
    return invalid


# Find missing rows
def find_missing_in_batch(batch, resolved_output):
    missing = []
    for row_id in batch["review_id"]:
        if row_id not in resolved_output or not resolved_output[row_id]:
            missing.append(row_id)
    return missing


# Clean model output (similar to your classification version)
def clean_llm_output_batch(raw_output):
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        import ast

        try:
            return ast.literal_eval(raw_output)
        except Exception:
            return {}


async def extract_sentiment_batch(batch, model):
    local_map = {i: row_id for i, row_id in enumerate(batch["review_id"].tolist())}
    reviews_text = "\n".join(
        [f"{i}. {text}" for i, text in enumerate(batch["review"].tolist())]
    )
    prompt = feature_extraction_prompt.format(reviews_text=reviews_text)

    response = None
    try:
        response = await asyncio.to_thread(
            lambda: (record_model_call(), model.generate_content(prompt))[1]
        )
        raw_text = getattr(response, "text", str(response)).strip()

        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0)
        output_tokens = getattr(usage, "candidates_token_count", 0)
        total_tokens = getattr(usage, "total_token_count", 0)

        if input_tokens <= 200_000:
            input_price_per_million = 1.25
            output_price_per_million = 10.00
        else:
            input_price_per_million = 2.50
            output_price_per_million = 15.00

        cost_usd = (input_tokens / 1_000_000) * input_price_per_million + (
            output_tokens / 1_000_000
        ) * output_price_per_million
        logger.info(
            json.dumps(
                {
                    "event": "usage_stats",
                    "task_index": task_index,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "est_cost_usd": round(cost_usd, 6),
                }
            )
        )

        print(
            f"📊 Batch usage — input: {input_tokens}, output: {output_tokens}, "
            f"total: {total_tokens}, est. cost: ${cost_usd:.4f}"
        )

    except Exception as e:
        raw_text = f"ERROR: {e}"

    batch_output = clean_llm_output_batch(raw_text)

    # Map local indices → real review_id
    resolved_output = {
        local_map[int(local_idx)]: review_obj
        for local_idx, review_obj in batch_output.items()
        if int(local_idx) in local_map
    }

    return resolved_output


async def retry_missing_rows_sentiment(
    batch, resolved_output, rowid_to_review, model, max_retries=3, batch_id=None
):
    attempt = 0
    missing_ids = find_missing_in_batch(batch, resolved_output)
    schema_invalid = validate_schema_batch(resolved_output, review_schema)

    if schema_invalid:
        for rid in schema_invalid.keys():
            resolved_output.pop(rid, None)
        missing_ids = list(set(missing_ids) | set(schema_invalid.keys()))

    while (missing_ids or schema_invalid) and attempt < max_retries:
        attempt += 1
        missing_rows_before = len(missing_ids)
        error_type = None
        try:
            retry_batch = pd.DataFrame(
                {
                    "review_id": missing_ids,
                    "review": [rowid_to_review[rid] for rid in missing_ids],
                }
            )

            retry_output = await extract_sentiment_batch(retry_batch, model)

            resolved_output.update(retry_output)

            # Re-check missing and schema
            missing_ids = find_missing_in_batch(batch, resolved_output)
            schema_invalid = validate_schema_batch(resolved_output, review_schema)
            missing_rows_after = len(missing_ids)

        except Exception as e:
            error_type = str(type(e).__name__)
            missing_rows_after = missing_rows_before
            logger.error(f"❌ Retry attempt {attempt} failed: {e}")

        finally:
            logger.info(
                json.dumps(
                    {
                        "event": "retry_step",
                        "task_index": task_index,
                        "batch_id": batch_id,
                        "attempt": attempt,
                        "missing_rows_before": missing_rows_before,
                        "missing_rows_after": missing_rows_after,
                        "error_type": error_type
                        or ("schema_invalid" if schema_invalid else None),
                    }
                )
            )
        if missing_ids or schema_invalid:
            print(
                f"🔄 Batch {batch_id} retry {attempt}, still missing: {missing_ids}, schema errors: {list(schema_invalid.keys())}"
            )

    if missing_ids or schema_invalid:
        print(f"❌ Batch {batch_id} failed after {max_retries} retries")
    else:
        print(f"✅ Batch {batch_id} succeeded after {attempt} retries")

    return resolved_output, attempt


nest_asyncio.apply()
rowid_to_review = dict(zip(df["review_id"], df["review"]))

MAX_CONCURRENT = 7
BATCH_SIZE = 50

# Shared list to collect all batch results
all_results = []


# Process a single batch with retry support
async def process_batch_sentiment(batch, model, semaphore, pbar, batch_id=None):
    async with semaphore:
        start_time = time.time()
        batch = batch.reset_index(drop=True)
        missing_rows_count = 0
        status = "success"

        try:
            resolved_output = await extract_sentiment_batch(batch, model)

            resolved_output, retries = await retry_missing_rows_sentiment(
                batch,
                resolved_output,
                rowid_to_review,
                model,
                max_retries=3,
                batch_id=batch_id,
            )

            after_retry_missing = find_missing_in_batch(batch, resolved_output)
            missing_rows_count = len(after_retry_missing)

            if after_retry_missing:
                status = "partial_fail"

        except Exception as e:
            status = "error"
            missing_rows_count = len(batch)
            logger.error(f"❌ Batch {batch_id} failed: {e}")

        finally:
            duration_sec = round(time.time() - start_time, 2)

            logger.info(
                json.dumps(
                    {
                        "event": "batch",
                        "task_index": task_index,
                        "batch_id": batch_id,
                        "status": status,
                        "retries": retries,
                        "duration_sec": duration_sec,
                        "missing_rows_count": missing_rows_count,
                    }
                )
            )

            if status == "error":
                all_results.append({})
            else:
                all_results.append(resolved_output)

            pbar.update(1)


# Run all batches asynchronously
async def run_all_batches_sentiment(
    df, model, batch_size=BATCH_SIZE, max_concurrent=MAX_CONCURRENT
):
    semaphore = asyncio.Semaphore(max_concurrent)
    num_batches = (len(df) + batch_size - 1) // batch_size

    logger.info(
        json.dumps(
            {
                "event": "task_start",
                "task_index": task_index,
                "total_reviews": len(df),
                "num_batches": num_batches,
            }
        )
    )

    with tqdm(total=num_batches, desc="Processing Batches") as pbar:
        tasks = []
        for i in range(num_batches):
            batch = df.iloc[i * batch_size : (i + 1) * batch_size]
            task = asyncio.create_task(
                process_batch_sentiment(batch, model, semaphore, pbar, batch_id=i + 1)
            )
            tasks.append(task)
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(run_all_batches_sentiment(df, model))
    print("✅ All tasks finished")

    rows = []
    for batch_dict in all_results:
        for rid, result in batch_dict.items():
            overall = result.get("overall_sentiment")
            for aspect in result.get("aspects", []):
                sentiment = aspect.get("sentiment")
                summary = aspect.get("summary")
                rows.append(
                    {
                        "review_id": rid,
                        "overall_sentiment": overall,
                        "aspect_sentiment": sentiment,
                        "aspect_summary": summary,
                    }
                )

    df_results = pd.DataFrame(rows)

    bucket_path = "gs://sent_proj"
    file_name = f"results_task_{task_index}.csv"
    full_path = f"{bucket_path}/{file_name}"

    df_results.to_csv(full_path, index=False)
    print(f"✅ Task {task_index} results saved to {full_path}")
