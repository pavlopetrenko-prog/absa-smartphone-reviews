 # **Aspect-Based Sentiment Analysis of Smartphone Customer Reviews**

This project is a **complete end-to-end cloud pipeline** that transforms **10,000 raw smartphone customer reviews** into structured, actionable insights using **LLM-powered Aspect-Based Sentiment Analysis (ABSA)**.  
It combines **Google Cloud**, **Python**, and **Gemini 2.5 Pro** to extract aspects, classify sentiment, detect entities, and generate a clean analytical dataset.  
The final results are visualized through **interactive Tableau dashboards** that reveal sentiment trends, category breakdowns, and detailed aspect-level insights.

## Repository Structure

```
project_root
├── data
│   
├── cloud_pipeline
│   ├── splitting
│   ├── batching
│   ├── llm_processing
│   │   ├── aspects_extraction
│   │   └── high\main category
│   ├── llm_output_validation
│   │   ├── aspect_extraction_validation
│   │   └── high\main_category_validation
│   ├── storage
│   ├── merging
│   ├── logging
│   ├── full_script_flow
│   └── final_dataset_validation
│       ├── semantical_validation
│       └── statistical_validation
|    
└── analytics
    └── 
```


 **Key Features**
- End-to-end cloud-based text analytics pipeline  
- Parallel batch processing using Cloud Run Jobs  
- ABSA modeling with Gemini 2.5 Pro (aspects, sentiment, entities, categories)  
- Schema validation + automatic backfilling of missing fields  
- Distributed → merged dataset consolidation  
- Statistical & manual validation  
- Tableau dashboards with category drilldowns and MoM metrics  

 ### **Technologies Used**

### **Programming**
- **Python 3.10**

### **Machine Learning / LLM**
- **Gemini 2.5 Pro** (aspect extraction, sentiment scoring, entity detection, classification)

### **Google Cloud Platform**
- **Cloud Run Jobs**  
- **Cloud Build**  
- **Google Cloud Storage (GCS)**  
- **Artifact Registry**  
- **Cloud Logging**  
- **BigQuery (log analysis only)**  

**Visualization**
**Tableau**

### **Pipeline Flow**

### **1. Ingestion & Task Splitting**
- Raw reviews uploaded to **GCS**
- Cloud Run Jobs automatically assign task indexes for distributed processing

### **2. Batch Processing**
Each Cloud Run task performs:
- dynamic batching  
- Gemini 2.5 Pro calls  
- schema validation  
- missing field detection and re-calls  
- saving outputs back to GCS  

### **3. Merging Final Dataset**
- All task-level outputs consolidated into one unified dataset  
- Clean schema and consistent ordering  

### **4. Validation**
- Statistically significant sampling  
- Manual review  
- Consistency checks  

### **5. Visualization**
- Tableau dashboards for insights, drilldowns, testing, and tracking changes  

