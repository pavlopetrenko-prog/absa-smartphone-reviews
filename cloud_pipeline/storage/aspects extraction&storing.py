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
