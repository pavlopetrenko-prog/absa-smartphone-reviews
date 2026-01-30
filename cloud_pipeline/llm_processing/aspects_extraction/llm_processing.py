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

    try:
        response = await asyncio.to_thread(lambda: model.generate_content(prompt))
        raw_text = getattr(response, "text", str(response)).strip()
    except Exception as e:
        raw_text = f"ERROR: {e}"

    batch_output = clean_llm_output_batch(raw_text)

    resolved_output = {
        local_map[int(local_idx)]: review_obj
        for local_idx, review_obj in batch_output.items()
        if int(local_idx) in local_map
    }

    return resolved_output


nest_asyncio.apply()
rowid_to_review = dict(zip(df["review_id"], df["review"]))
all_results = []


async def process_batch_sentiment(batch, model):
    batch = batch.reset_index(drop=True)
    resolved_output = await extract_sentiment_batch(batch, model)
    all_results.append(resolved_output)


async def run_all_batches_sentiment(df, model, batch_size=BATCH_SIZE):
    num_batches = (len(df) + batch_size - 1) // batch_size

    for i in range(num_batches):
        batch = df.iloc[i * batch_size : (i + 1) * batch_size]
        await process_batch_sentiment(batch, model)


if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(run_all_batches_sentiment(df, model))
    print("✅ All tasks finished")
