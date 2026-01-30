## 1. Schema Validation (per-review validation)
### Goal:
Guarantee that every returned object has:

overall_sentiment

aspects (a list)

each aspect with:
sentiment and summary


### If a review does not match the schema:

that review is marked as invalid - its entry is removed from the batch output - it is added to the list of rows to retry

This ensures the pipeline never accepts malformed JSON.

## 2. Missing-Row Detection

### Even if the model returns valid JSON, it may skip some reviews.
Step 1 - Review Id Indexing

```
local index → review_id
0            324
1            325
2            326
3            327
```
Step 2 - Reviews Indexing and Sent to the Model
```
0. "Great phone!"
1. "Battery drains fast."
2. "Camera is excellent."
3. "Screen is too dim."
```
Step 3 - Model Output
```
{
  "0": {...},          
  "1": {...},          
  "3": {...}...         

"2" is missing
}
```
Step 4 — Map Local Indices → Real review_id

```
local index 0 → review_id 324
local index 1 → review_id 325
local index 3 → review_id 327
```
Mapped Output Now:
```
{
  324: {...},
  325: {...},
  327: {...}...
}
```
Step 5 - Missing-Row Detection
```
expected review_ids in batch:
[324, 325, 326, 327,...]

Mapped output contains:
[324, 325, 327,...]
```
Final Missing List
```[326,...]```
### 3. Retry Logic (max 3 retries)

After gathering all invalid + missing rows, a mini-batch is created containing only those review_ids.

This mini-batch is sent back to the model again.

After each retry:

the newly returned rows are merged into the existing batch output

the pipeline re-checks the batch for both:

schema invalid rows

missing rows

Retries continue until:

all rows are valid
or

retry limit is reached

Max retries: 3 attempts

If after 3 retries some rows still:

fail schema validation
or

are not returned by the model

then:

The batch is marked as “partial fail”
Those specific rows remain missing in the final output
The batch still continues and the rest of the valid data is stored.

### 4. Final Outcome

At the end of the process, each batch produces:

a dictionary of valid review_id → extracted data

(optionally) some missing rows if they failed all retries

### The batch output is appended to all_results, and later merged across tasks.

