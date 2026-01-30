async def process_batch_sentiment(batch, model, semaphore, batch_id=None):
    """Process a single sentiment batch with concurrency control."""
    async with semaphore:
        batch = batch.reset_index(drop=True)

        # only call the model and return its result
        resolved_output = await extract_sentiment_batch(batch, model)

        return resolved_output


# Run all batches asynchronously


async def run_all_batches_sentiment(
    df, model, batch_size=BATCH_SIZE, max_concurrent=MAX_CONCURRENT
):
    """Split dataframe into batches and schedule async processing."""
    semaphore = asyncio.Semaphore(max_concurrent)

    num_batches = math.ceil(len(df) / batch_size)

    tasks = []
    for i in range(num_batches):
        batch = df.iloc[i * batch_size : (i + 1) * batch_size]
        task = asyncio.create_task(
            process_batch_sentiment(batch, model, semaphore, batch_id=i + 1)
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    return results


# Entrypoint

if __name__ == "__main__":
    nest_asyncio.apply()
    final_results = asyncio.run(run_all_batches_sentiment(df, model))
    print("✅ All batches completed.")
