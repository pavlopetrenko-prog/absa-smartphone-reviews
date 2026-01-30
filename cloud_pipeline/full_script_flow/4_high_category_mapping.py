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
from typing import Dict, Tuple, List, Any
import ast
import time
import re
from collections import deque
import threading
import logging
import sys


###############################################################
api_key = os.environ.get("MY_ENV_VAR")
if not api_key:
    raise ValueError("API key not found in environment variable MY_ENV_VAR")
print("✅ Secret key accessed successfully")
genai.configure(api_key=api_key)
generation_config = {"response_mime_type": "application/json"}
model = genai.GenerativeModel("gemini-2.5-pro", generation_config=generation_config)
###############################################################

logger = logging.getLogger("high_category")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


call_timestamps = deque()
lock = threading.Lock()


class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()

    async def wait(self):
        while len(self.calls) >= self.max_calls:
            now = time.time()
            earliest = self.calls[0]
            wait_time = self.period - (now - earliest)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            else:
                self.calls.popleft()
        self.calls.append(time.time())


rate_limiter = RateLimiter(max_calls=7, period=60)

###############################################################
# 1. Extract env vars from Cloud Run Job
task_index = int(os.getenv("CLOUD_RUN_TASK_INDEX", 0))  # e.g. 0,1,2...
task_count = int(os.getenv("CLOUD_RUN_TASK_COUNT", 1))  # total number of tasks

# 2. Load your dataset (for example, from GCS or local volume)
df = pd.read_csv("gs://sent_proj/initial_extraction.csv")


# 3. Split dataframe into N roughly equal chunks
def split_dataframe(df, n_chunks):
    chunk_size = math.ceil(len(df) / n_chunks)
    return [df[i * chunk_size : (i + 1) * chunk_size] for i in range(n_chunks)]


chunks = split_dataframe(df, task_count)

# 4. Pick the chunk for *this* task
my_chunk = chunks[task_index]
df = my_chunk
ready_df = df[["row_id", "aspect_summary"]]
###############################################################

categorisation_prompt_raw = """
You are a text classification assistant.  
Classify each customer review into **exactly one** of the following high-level categories:

1. **INFRASTRUCTURE** → Reviews focusing on delivery, customer service, refunds, company operations, or related processes.  
   Examples: "Delivery was late", "Customer support is unhelpful", "Refund not received."

2. **PHONE** → Reviews that focus on the physical product or its features such as battery, camera, display, performance, sound, or design.  
   Examples: "Battery drains quickly", "Camera is bad", "Screen is bright", "The phone lags."

3. **GENERAL_FEEDBACK** → Reviews that express **overall satisfaction or dissatisfaction** with the product or company,  
   but **do not mention a specific feature or service**.  
   Examples: "Good phone", "Worst phone ever", "Awesome product", "Very bad", "Love it", "Not worth it."

---

**Important rules**:
- If a review only expresses general sentiment (positive/negative) or mentions the word “phone” or “product” **without describing a specific feature**, classify it as **GENERAL_FEEDBACK**.  
- Only classify as **PHONE** when the review clearly refers to a feature or part of the device (battery, camera, screen, sound, etc.).  
- Only classify as **INFRASTRUCTURE** when the review refers to service, delivery, or company-related issues.  
- If you are uncertain, choose **GENERAL_FEEDBACK** (default fallback).

Return the results in **valid JSON format** without extra text.

Here are the reviews to classify:
{reviews}
"""


def clean_llm_output_batch(raw_output: str) -> Dict[int, str]:
    MAIN_CATEGORIES = {"INFRASTRUCTURE", "PHONE", "GENERAL_FEEDBACK"}
    result = {}

    try:
        if not raw_output or not isinstance(raw_output, str):
            return {}

        # --- basic cleaning ---
        text = raw_output.strip()
        text = re.sub(r"^```(?:json)?|```$", "", text).strip()
        text = re.sub(r"(?m)(?<=\{|,)\s*(\d+)\s*:", r'"\1":', text)  # fix unquoted keys

        # --- extract inner JSON if model wrapped it with text ---
        if not text.startswith(("{", "[")):
            match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
            if match:
                text = match.group(1)

        # --- try parsing ---
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(text)

        # --- Case 1: dict form {"0":"PHONE"} ---
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                if (
                    str(k).isdigit()
                    and isinstance(v, str)
                    and v.strip().upper() in MAIN_CATEGORIES
                ):
                    result[int(k)] = v.strip().upper()

        # --- Case 2: list form [{"...": ...}] ---
        elif isinstance(parsed, list):
            for i, item in enumerate(parsed):
                if isinstance(item, str):
                    try:
                        item = json.loads(item)
                    except Exception:
                        continue
                if not isinstance(item, dict):
                    continue

                # detect ID-like field (integer or digit string)
                id_val = None
                for val in item.values():
                    if isinstance(val, (int, str)) and str(val).isdigit():
                        id_val = int(val)
                        break
                if id_val is None:
                    id_val = i  # fallback to index

                # detect category-like field (value from MAIN_CATEGORIES)
                cat_val = None
                for val in item.values():
                    if isinstance(val, str) and val.strip().upper() in MAIN_CATEGORIES:
                        cat_val = val.strip().upper()
                        break

                if cat_val:
                    result[id_val] = cat_val

        return result

    except Exception as e:
        print(f"⚠️ clean_llm_output_batch general failure: {e}")
        return {}


def find_missing_in_batch(batch, resolved_output):
    missing = []
    for row_id in batch["row_id"]:
        # Check: row_id not in resolved_output OR category is None/empty
        if row_id not in resolved_output or not resolved_output[row_id]:
            missing.append(row_id)
    return missing


rowid_to_review = dict(zip(ready_df["row_id"], ready_df["aspect_summary"]))


async def retry_missing_rows(
    batch,
    resolved_output,
    rowid_to_review,
    model,
    semaphore,
    pbar,
    max_retries=3,
    batch_id=None,
) -> Tuple[Dict[int, str], List[int]]:
    attempt = 0
    missing_ids = find_missing_in_batch(batch, resolved_output)

    if missing_ids:
        print(f"⚠️ Batch {batch_id}: missing {missing_ids}")

    while missing_ids and attempt < max_retries:
        attempt += 1
        missing_rows_before = len(missing_ids)
        error_type = None

        try:
            retry_batch = pd.DataFrame(
                {
                    "row_id": missing_ids,
                    "aspect_summary": [rowid_to_review[mid] for mid in missing_ids],
                }
            )

            retry_output = await classify_and_parse(retry_batch, model)
            resolved_output.update(retry_output)

            missing_ids = find_missing_in_batch(batch, resolved_output)
            missing_rows_after = len(missing_ids)
        except Exception as e:
            error_type = str(type(e).__name__)
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
                        "error_type": error_type,
                    }
                )
            )

        if missing_ids:
            print(f"🔄 Batch {batch_id}: retry {attempt}, still missing {missing_ids}")

    if missing_ids:
        print(
            f"❌ Batch {batch_id}: still missing after {max_retries} retries: {missing_ids}"
        )
    else:
        print(f"✅ Batch {batch_id}: all rows classified after {attempt} retries")

    return resolved_output, attempt


async def classify_and_parse(batch, model) -> Dict[int, str]:
    # Build local index → row_id map
    local_map = {i: int(row_id) for i, row_id in enumerate(batch["row_id"].tolist())}

    # Prepare prompt with local indices
    reviews_text = "\n".join(
        [f"{i}. {text}" for i, text in enumerate(batch["aspect_summary"].tolist())]
    )
    prompt = categorisation_prompt_raw.format(reviews=reviews_text)

    # Call model
    try:
        await rate_limiter.wait()
        response = await asyncio.to_thread(model.generate_content, prompt)

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
                    "event": "retry_cost",
                    "task_index": task_index,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "est_cost_usd": round(cost_usd, 6),
                }
            )
        )

        print(
            f"📊 Retry Cost — input: {input_tokens}, output: {output_tokens}, "
            f"total: {total_tokens}, est. cost: ${cost_usd:.4f}"
        )

    except Exception as e:
        raw_text = f"ERROR: {e}"

    batch_output = clean_llm_output_batch(raw_text)

    resolved_output = {
        local_map[int(local_idx)]: category
        for local_idx, category in batch_output.items()
        if int(local_idx) in local_map
    }
    return resolved_output


raw_outputs = []


async def process_batch(batch, semaphore, model, pbar, shared_list, batch_id):
    async with semaphore:
        start_time = time.time()
        batch = batch.reset_index(drop=True)
        missing_rows_count = 0
        final_missing_ids = []

        local_map = {
            i: int(row_id) for i, row_id in enumerate(batch["row_id"].tolist())
        }

        reviews_text = "\n".join(
            [f"{i}. {text}" for i, text in enumerate(batch["aspect_summary"].tolist())]
        )
        prompt = categorisation_prompt_raw.format(reviews=reviews_text)

        try:
            await rate_limiter.wait()
            response = await asyncio.to_thread(model.generate_content, prompt)
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

        finally:
            duration_sec = round(time.time() - start_time, 2)

        batch_output = clean_llm_output_batch(raw_text)

        resolved_output = {
            local_map[int(local_idx)]: category
            for local_idx, category in batch_output.items()
            if int(local_idx) in local_map
        }

        missing_ids = find_missing_in_batch(batch, resolved_output)
        missing_rows_count = len(missing_ids)

        if missing_rows_count > 0:
            logger.info(
                json.dumps(
                    {
                        "event": "debug_raw_output",
                        "task_index": task_index,
                        "batch_id": batch_id,
                        "raw_output_preview": raw_text,
                    }
                )
            )

        resolved_output, retries = await retry_missing_rows(
            batch,
            resolved_output,
            rowid_to_review,
            model,
            semaphore,
            pbar,
            batch_id=batch_id,
        )

        final_missing_ids = find_missing_in_batch(batch, resolved_output)
        missing_rows_after = len(final_missing_ids)
        shared_list.append(resolved_output)

        pbar.update(1)

        logger.info(
            json.dumps(
                {
                    "event": "batch",
                    "task_index": task_index,
                    "batch_id": batch_id,
                    "retries": retries,
                    "duration_sec": duration_sec,
                    "missing_rows_count": missing_rows_count,
                    "missing_rows_after": missing_rows_after,
                }
            )
        )


async def run_all_batches(
    ready_df, model, batch_size=50, max_concurrent=7, shared_list=None
):
    num_batches = (len(ready_df) + batch_size - 1) // batch_size
    semaphore = asyncio.Semaphore(max_concurrent)

    with tqdm(total=num_batches, desc="Processing Batches") as pbar:
        tasks = []
        for i in range(num_batches):
            batch = ready_df.iloc[i * batch_size : (i + 1) * batch_size]
            task = asyncio.create_task(
                process_batch(batch, semaphore, model, pbar, shared_list, batch_id=i)
            )
            tasks.append(task)
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    nest_asyncio.apply()
    raw_outputs = []

    asyncio.run(
        run_all_batches(
            ready_df,
            model,
            batch_size=50,
            max_concurrent=7,
            shared_list=raw_outputs,
        )
    )

    flat_list = [
        (row_id, category)
        for batch_dict in raw_outputs
        for row_id, category in batch_dict.items()
    ]

    df_classified = pd.DataFrame(flat_list, columns=["row_id", "high_category"])

    bucket_path = "gs://sent_proj"
    file_name = f"results_task_{task_index}.csv"
    full_path = f"{bucket_path}/{file_name}"

    df_classified.to_csv(full_path, index=False)
    print(f"✅ Task {task_index} results saved to {full_path}")
