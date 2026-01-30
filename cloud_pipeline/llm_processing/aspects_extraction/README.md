### LLM Processing

This section is responsible for initial aspects extraction from raw reviews and \
explains how each Cloud Run task interacts with the LLM (Gemini 2.5 Pro) 

The focus here is on the conceptual flow — how prompts are built, how batches are processed, and how outputs are mapped — without going into validation or missing-row logic (which is handled separately in the llm_output_validation module).

---

### What this module does

The goal of the LLM processing stage is to transform unstructured natural-language reviews into structured JSON objects that follow a predefined schema.

Each Cloud Run task:
- Receives a chunk of the dataset (already split earlier)
- Processes that chunk in multiple batches
- Sends each batch to Gemini using a carefully designed prompt
- Receives model output for every review in the batch
- Maps local batch indices back to the global review IDs

This module is responsible only for:
- Preparing the prompt
- Sending the prompt to the LLM
- Receiving and cleaning the response
- Mapping outputs to the correct review identifiers

It does **not** perform retries or schema checks — those belong to llm_output_validation.

---

### How prompting works

### How Review ID Mapping Works

#### 1. Local indexing inside the batch
Before sending a batch to the LLM, each review is temporarily assigned a local index:
```
local_map = {
    0: 145233,
    1: 145234,
    2: 145235
}
```
And the prompt is built like:
```
0. "Battery life great"
1. "Camera terrible"
2. "Screen cracked"
```
The LLM always returns JSON keyed by **0, 1, 2**.

---

#### 2. Model output (local indices)
Example LLM output:
```
{
  "0": {...},
  "1": {...},
  "2": {...}
}
```
---

#### 3. Merge back into global review IDs
We convert local indices back to real review IDs:
```
resolved_output = {
    145233: output["0"],
    145234: output["1"],
    145235: output["2"]
}
```
---

#### Main Idea
Local indices make the prompt simple for the LLM, and the mapping ensures the final structured output is restored to the correct original `review_id` values.



Its purpose is to convert raw text → structured JSON using a deterministic prompt and a consistent output mapping system.

It does **not** validate, retry, or filter the results — it only produces the initial structured output for each batch.
