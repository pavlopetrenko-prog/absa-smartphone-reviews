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


def find_missing_in_batch(batch, resolved_output):
    missing = []
    for row_id in batch["review_id"]:
        if row_id not in resolved_output or not resolved_output[row_id]:
            missing.append(row_id)
    return missing


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

        try:
            retry_batch = pd.DataFrame(
                {
                    "review_id": missing_ids,
                    "review": [rowid_to_review[rid] for rid in missing_ids],
                }
            )

            retry_output = await extract_sentiment_batch(retry_batch, model)
            resolved_output.update(retry_output)

            missing_ids = find_missing_in_batch(batch, resolved_output)
            schema_invalid = validate_schema_batch(resolved_output, review_schema)
            missing_rows_after = len(missing_ids)

        except Exception:
            missing_rows_after = missing_rows_before

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

all_results = []


async def process_batch_sentiment(batch, model, semaphore, pbar, batch_id=None):
    async with semaphore:
        batch = batch.reset_index(drop=True)
        missing_rows_count = 0
        status = "success"
        retries = 0

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

        except Exception:
            status = "error"
            missing_rows_count = len(batch)

        finally:
            if status == "error":
                all_results.append({})
            else:
                all_results.append(resolved_output)

            pbar.update(1)
