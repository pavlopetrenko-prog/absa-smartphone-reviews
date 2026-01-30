```
feature_extraction_prompt = """
You are given reviews.
For each review, return an entry where the key is the review index (starting from 0)
and the value is the following JSON schema:

{
  "overall_sentiment": "positive" | "negative",
  "aspects": [
    {
      "sentiment": "positive" | "negative",
      "summary": "Brief quote describing a feature, service, or problem."
    }
  ]
}

Reviews to analyze:
{reviews_text}
"""
```
