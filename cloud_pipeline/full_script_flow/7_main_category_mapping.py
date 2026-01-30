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

########################################################################################
api_key = os.environ.get("MY_ENV_VAR")
if not api_key:
    raise ValueError("API key not found in environment variable MY_ENV_VAR")
print("✅ Secret key accessed successfully")
genai.configure(api_key=api_key)
generation_config = {"response_mime_type": "application/json"}
model = genai.GenerativeModel("gemini-2.5-pro", generation_config=generation_config)


########################################################################################
# 1. Extract env vars from Cloud Run Job
task_index = int(os.getenv("CLOUD_RUN_TASK_INDEX", 0))  # e.g. 0,1,2...
task_count = int(os.getenv("CLOUD_RUN_TASK_COUNT", 1))  # total number of tasks

# 2. Load your dataset (for example, from GCS or local volume)
df = pd.read_csv("gs://sent_proj/high_category.csv")


# 3. Split dataframe into N roughly equal chunks
def split_dataframe(df, n_chunks):
    chunk_size = math.ceil(len(df) / n_chunks)
    return [df[i * chunk_size : (i + 1) * chunk_size] for i in range(n_chunks)]


chunks = split_dataframe(df, task_count)

# 4. Pick the chunk for *this* task
my_chunk = chunks[task_index]
df = my_chunk
df = df[["row_id", "aspect_summary", "high_category"]]

########################################################################################

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
########################################################################################

phone_features_promp_raw = """
You are a text classification assistant.  
Classify each PHONE-related customer review into **exactly one** of the following categories:

BATTERY → charging speed and methods (wired, wireless, fast charging), battery life, standby time, power consumption, and temperature/thermal management. Excludes performance lag (→ PERFORMANCE) or physical battery build quality (→ DESIGN).
CAMERA → photo and video capabilities: image and video quality, sharpness, low-light performance, focus, zoom, flash, stabilization, and camera features. Excludes durability (→ DESIGN) or storage for media files (→ STORAGE).
PERFORMANCE → overall device responsiveness: speed, lag, multitasking, app loading, crashes, freezes, and gaming performance. Excludes software bugs in specific apps (→ SOFTWARE) or overheating (→ BATTERY).
DISPLAY → screen-related aspects: brightness, resolution, color accuracy, sharpness, refresh rate, touch response, outdoor visibility, and durability (scratches, cracks). Excludes device size/weight (→ DESIGN).
CONNECTIVITY → all device connections: mobile network signal, SIM functionality, Wi-Fi, Bluetooth, GPS, hotspot, USB/data transfer, and carrier compatibility. Excludes call sound quality (→ AUDIO).
DESIGN → physical characteristics: aesthetics, look and feel, size, weight, ergonomics, build materials, durability, button placement, and safety. Excludes internal performance (→ PERFORMANCE) or camera operation (→ CAMERA).
SOFTWARE → operating system and applications: OS updates, UI/UX, pre-installed apps, app store, app functionality, customization, and system bugs. Excludes performance speed (→ PERFORMANCE) or hardware faults (→ DESIGN, BATTERY).
STORAGE → memory and data handling: internal storage capacity, usable space, file management, expandability (memory cards), and cloud sync. Excludes performance slowdown (→ PERFORMANCE).
AUDIO → sound quality and features: speakers, microphone, headphone jack, headphones, ringtones, notifications, call audio quality, and overall sound performance. Excludes connectivity features (→ CONNECTIVITY) and accessory design (→ DESIGN).  


The reviews are numbered from 0 to N.  
Return a JSON object where keys are the review indices (0, 1, 2...) and values are ONLY the category names as strings. Do not include the review text or any additional information.

Here are the reviews to classify:
{reviews}
"""

infrastructure_features_promp_raw = """
You are a text classification assistant.  
Classify each infrastructure-related customer review into **exactly one** of the following categories:

RETURNS & EXCHANGES → ease and clarity of returning items, exchange process, fairness and application of return/exchange policies. Excludes financial reimbursement (→ REFUNDS & CASHBACK) or delivery issues (→ DELIVERY).
REFUNDS & CASHBACK → speed, reliability, and transparency of refunds or cashback processing. Excludes physical delivery or item condition (→ DELIVERY) and customer support interactions (→ CUSTOMER SUPPORT).
DELIVERY → logistics-related aspects: delivery speed, packaging quality, product condition on arrival, and damage during transit. Excludes tracking accuracy (→ ORDER TRACKING & COMMUNICATION) or policy issues (→ POLICY CLARITY & TRUSTWORTHINESS).
ORDER TRACKING & COMMUNICATION → accuracy of tracking, status updates, and delivery notifications. Excludes physical delivery quality (→ DELIVERY) or financial reimbursements (→ REFUNDS & CASHBACK).
CUSTOMER SUPPORT → responsiveness, helpfulness, professionalism, and problem resolution during service interactions. Excludes financial reimbursements (→ REFUNDS & CASHBACK), returns/exchanges process (→ RETURNS & EXCHANGES), or delivery/packaging quality (→ DELIVERY).
POLICY CLARITY & TRUSTWORTHINESS → transparency, clarity, and consistency of terms, hidden conditions, and application of policies. Excludes service speed (→ CUSTOMER SUPPORT) or logistics (→ DELIVERY).

The reviews are numbered from 0 to N.  
Return a JSON object where keys are the review indices (0, 1, 2...) and values are ONLY the category names as strings. Do not include the review text or any additional information.

Here are the reviews to classify:
{reviews}
"""

general_features_promp_raw = """
You are a text classification assistant.  
Classify each general feedack - related customer review into **exactly one** of the following categories:

PRICE & VALUE → affordability, fairness of pricing, and perceived value for money. Excludes general satisfaction (→ OVERALL SATISFACTION) or brand reputation (→ BRAND TRUST & REPUTATION).
BRAND TRUST & REPUTATION → reliability, credibility, brand image, and overall brand perception. Excludes pricing concerns (→ PRICE & VALUE) or individual product performance (→ PRODUCT FEATURES).
OVERALL SATISFACTION → general impressions, happiness or unhappiness with the product/service, and likelihood to recommend. Excludes specific product or service categories (→ PRODUCT FEATURES / INFRASTRUCTURE).
LOYALTY & REPEAT PURCHASE → willingness to buy again and long-term commitment to the brand. Excludes immediate satisfaction (→ OVERALL SATISFACTION) or competitor comparison (→ COMPETITOR COMPARISON).
MARKETING & PROMOTIONS → reactions to advertisements, campaigns, special deals, or promotional offers. Excludes pricing evaluation (→ PRICE & VALUE) or product performance (→ PRODUCT FEATURES).
COMPETITOR COMPARISON → direct comparisons with other brands, products, or services. Excludes evaluations of the reviewed product alone (→ PRODUCT FEATURES / OVERALL SATISFACTION).
ETHICS & SOCIAL RESPONSIBILITY → sustainability, eco-friendliness, fairness, and corporate responsibility. Excludes product performance (→ PRODUCT FEATURES) or service logistics (→ INFRASTRUCTURE).

The reviews are numbered from 0 to N.  
Return a JSON object where keys are the review indices (0, 1, 2...) and values are ONLY the category names as strings. Do not include the review text or any additional information.

Here are the reviews to classify:
{reviews}
"""

PROMPT_MAP = {
    "PHONE": phone_features_promp_raw,
    "INFRASTRUCTURE": infrastructure_features_promp_raw,
    "GENERAL_FEEDBACK": general_features_promp_raw,
}
########################################################################################


FINAL_RESULTS = []


def clean_llm_output_batch(raw_output: str, high_category: str) -> Dict[int, str]:
    CATEGORY_SETS = {
        "PHONE": {
            "BATTERY",
            "CAMERA",
            "PERFORMANCE",
            "DISPLAY",
            "CONNECTIVITY",
            "DESIGN",
            "SOFTWARE",
            "STORAGE",
            "AUDIO",
        },
        "INFRASTRUCTURE": {
            "RETURNS & EXCHANGES",
            "REFUNDS & CASHBACK",
            "DELIVERY",
            "ORDER TRACKING & COMMUNICATION",
            "CUSTOMER SUPPORT",
            "POLICY CLARITY & TRUSTWORTHINESS",
        },
        "GENERAL_FEEDBACK": {
            "PRICE & VALUE",
            "BRAND TRUST & REPUTATION",
            "OVERALL SATISFACTION",
            "LOYALTY & REPEAT PURCHASE",
            "MARKETING & PROMOTIONS",
            "COMPETITOR COMPARISON",
            "ETHICS & SOCIAL RESPONSIBILITY",
        },
    }

    MAIN_CATEGORIES = CATEGORY_SETS.get(high_category, set())
    result = {}

    try:
        if not raw_output or not isinstance(raw_output, str):
            return {}

        text = raw_output.strip()
        text = re.sub(r"^```(?:json)?|```$", "", text).strip()
        text = re.sub(r"(?m)(?<=\{|,)\s*(\d+)\s*:", r'"\1":', text)

        if not text.startswith(("{", "[")):
            match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
            if match:
                text = match.group(1)

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(text)

        if isinstance(parsed, dict):
            for k, v in parsed.items():
                if str(k).isdigit() and isinstance(v, str):
                    cat = v.strip().upper()
                    if cat in MAIN_CATEGORIES:
                        result[int(k)] = cat

        elif isinstance(parsed, list):
            for i, item in enumerate(parsed):
                if isinstance(item, str):
                    try:
                        item = json.loads(item)
                    except Exception:
                        continue
                if not isinstance(item, dict):
                    continue

                id_val = None
                for val in item.values():
                    if isinstance(val, (int, str)) and str(val).isdigit():
                        id_val = int(val)
                        break
                if id_val is None:
                    id_val = i

                cat_val = None
                for val in item.values():
                    if isinstance(val, str):
                        cat = val.strip().upper()
                        if cat in MAIN_CATEGORIES:
                            cat_val = cat
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


async def retry_missing_rows(
    batch,
    resolved_output,
    rowid_to_review,
    model,
    semaphore,
    pbar,
    prompt_template,
    high_category,
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

            retry_output = await classify_and_parse(
                retry_batch, model, prompt_template, high_category
            )
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


async def classify_and_parse(
    batch, model, prompt_template, high_category
) -> Dict[int, str]:
    # Build local index → row_id map
    local_map = {i: int(row_id) for i, row_id in enumerate(batch["row_id"].tolist())}

    # Prepare prompt with local indices
    reviews_text = "\n".join(
        [f"{i}. {text}" for i, text in enumerate(batch["aspect_summary"].tolist())]
    )
    prompt = prompt_template.format(reviews=reviews_text)

    # Call model
    try:
        await rate_limiter.wait()
        response = await asyncio.to_thread(model.generate_content, prompt)

        raw_text = getattr(response, "text", str(response)).strip()
    except Exception as e:
        raw_text = f"ERROR: {e}"

    # Parse LLM response
    batch_output = clean_llm_output_batch(raw_text, high_category)

    # Map local indices → global row_ids
    resolved_output = {
        local_map[int(local_idx)]: category
        for local_idx, category in batch_output.items()
        if int(local_idx) in local_map
    }
    return resolved_output


# Process batch function
async def process_batch(
    batch,
    semaphore,
    model,
    pbar,
    shared_list,
    batch_id,
    prompt_template,
    rowid_to_review,
    high_category,
):
    async with semaphore:
        start_time = time.time()
        batch = batch.reset_index(drop=True)
        missing_rows_count = 0
        final_missing_ids = []

        # Build local index → row_id map
        local_map = {
            i: int(row_id) for i, row_id in enumerate(batch["row_id"].tolist())
        }

        # Prepare prompt with local indices
        reviews_text = "\n".join(
            [f"{i}. {text}" for i, text in enumerate(batch["aspect_summary"].tolist())]
        )
        prompt = prompt_template.format(reviews=reviews_text)

        # Call model
        try:
            await rate_limiter.wait()
            response = await asyncio.to_thread(model.generate_content, prompt)

            raw_text = getattr(response, "text", str(response)).strip()
        except Exception as e:
            raw_text = f"ERROR: {e}"
        finally:
            duration_sec = round(time.time() - start_time, 2)

        # Parse LLM response
        batch_output = clean_llm_output_batch(raw_text, high_category)

        # Map local indices → global row_ids
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
            prompt_template,
            high_category,
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
    df_not_null,
    model,
    prompt_template,
    batch_size=50,
    max_concurrent=7,
    shared_list=None,
    rowid_to_review=None,
    high_category=None,
):
    num_batches = (len(df_not_null) + batch_size - 1) // batch_size
    semaphore = asyncio.Semaphore(max_concurrent)

    with tqdm(total=num_batches, desc="Processing Batches") as pbar:
        tasks = []
        for i in range(num_batches):
            batch = df_not_null.iloc[i * batch_size : (i + 1) * batch_size]
            task = asyncio.create_task(
                process_batch(
                    batch,
                    semaphore,
                    model,
                    pbar,
                    shared_list,
                    batch_id=i,
                    prompt_template=prompt_template,
                    rowid_to_review=rowid_to_review,
                    high_category=high_category,
                )
            )
            tasks.append(task)
        await asyncio.gather(*tasks)


########################################################################################


async def main():
    nest_asyncio.apply()

    for category, sub_df in df.groupby("high_category"):
        print(f"▶️ Processing category: {category} ({len(sub_df)} rows)")

        prompt_template = PROMPT_MAP.get(category)
        if not prompt_template:
            print(f"⚠️ No prompt template found for {category}, skipping.")
            continue

        rowid_to_review = dict(zip(sub_df["row_id"], sub_df["aspect_summary"]))

        # directly append into FINAL_RESULTS
        await run_all_batches(
            sub_df[["row_id", "aspect_summary"]],
            model,
            prompt_template,
            batch_size=50,
            max_concurrent=7,
            shared_list=FINAL_RESULTS,
            rowid_to_review=rowid_to_review,
            high_category=category,
        )


########################################################################################
if __name__ == "__main__":
    asyncio.run(main())
    merged_dict = {
        row_id: category
        for batch in FINAL_RESULTS
        for row_id, category in batch.items()
    }
    main_cat_df = pd.DataFrame(
        list(merged_dict.items()), columns=["row_id", "main_category"]
    )

    bucket_path = "gs://sent_proj"
    file_name = f"results_task_{task_index}.csv"
    full_path = f"{bucket_path}/{file_name}"

    main_cat_df.to_csv(full_path, index=False)
    print(f"✅ Task {task_index} results saved to {full_path}")
