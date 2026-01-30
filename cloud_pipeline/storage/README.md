# Process of storing data after each project step.
## After each running job (aspect extraction,high/main category) : 
data for each job is stored under name results_task_{task_index}.csv


so that it is possible to automatically merge dataset running one script(merging folder)

## I added part for only high category storing, because for main category the process is the same, but the difference is only in 
```
main_cat_df = pd.DataFrame(
        list(merged_dict.items()), columns=["row_id", "main_category"]
    )
```
