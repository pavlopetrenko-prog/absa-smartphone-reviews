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
