ready_df = df[["row_id", "aspect_summary"]]
\
###############################################################
\
```
categorisation_prompt_raw = """
You are a text classification assistant.  
Classify each customer review into **exactly one** of the following high-level categories:

1. **INFRASTRUCTURE** → Reviews focusing on delivery, customer service, refunds, company operations, or related processes.  
   Examples: "Delivery was late", "Customer support is unhelpful", "Refund not received."

2. **PHONE** → Reviews that focus on the physical product or its features such as battery, camera, display, performance, sound, or design.  
   Examples: "Battery drains quickly", "Camera is bad", "Screen is bright", "The phone lags."

3. **GENERAL_FEEDBACK** → Reviews that express **overall satisfaction or dissatisfaction** with the product or company,  
   but **do not mention a specific feature or service**.  
   Examples: "Good phone", "Worst phone ever", "Awesome product", "Very bad", "Love it", "Not worth it."



**Important rules**:
- If a review only expresses general sentiment (positive/negative) or mentions the word “phone” or “product” **without describing a specific feature**, classify it as **GENERAL_FEEDBACK**.  
- Only classify as **PHONE** when the review clearly refers to a feature or part of the device (battery, camera, screen, sound, etc.).  
- Only classify as **INFRASTRUCTURE** when the review refers to service, delivery, or company-related issues.  
- If you are uncertain, choose **GENERAL_FEEDBACK** (default fallback).

Return the results in **valid JSON format** without extra text.

Here are the reviews to classify:
{reviews}
"""
```
