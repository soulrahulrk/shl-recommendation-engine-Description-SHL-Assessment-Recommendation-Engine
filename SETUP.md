# SHL Assessment Recommendation Engine - Setup Guide

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/shl-recommendation-engine.git
cd shl-recommendation-engine
```

### 2. Install Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 3. Configure API Key
Get a free Groq API key from https://console.groq.com/keys

Create `.env` file:
```bash
GROQ_API_KEY=gsk_your_api_key_here
```

### 4. Run the Application

**Option A: Streamlit Web App**
```bash
streamlit run app.py
```
Access at http://localhost:8501

**Option B: FastAPI Backend**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

## Project Components

### Data Pipeline
- `src/crawler.py` - Crawl SHL catalog (already done, 377 tests)
- `data/catalog_enriched.json` - Crawled assessment data
- `data/vector_store.faiss` - FAISS vector index
- `data/metadata.pkl` - Metadata for quick lookup

### Recommendation System
- `src/vector_store.py` - Vector store with semantic search
- `llm/rerank.py` - LLM re-ranking with Groq
- `pipeline.py` - End-to-end pipeline

### Evaluation & Submission
- `evaluate.py` - Compute Mean Recall@10
- `generate_submission.py` - Generate predictions CSV
- `submission.csv` - Test set predictions

### APIs
- `api.py` - FastAPI backend
- `app.py` - Streamlit frontend

## API Usage

### POST /recommend
```python
import requests

response = requests.post("http://localhost:8000/recommend", json={
    "query": "Looking for Java developers with collaboration skills"
})

print(response.json())
```

### Response Format
```json
{
  "recommended_assessments": [
    {
      "name": "Core Java (Entry Level)",
      "url": "https://www.shl.com/products/...",
      "description": "Assesses Java programming skills",
      "test_type": ["Knowledge & Skills"],
      "duration": 30,
      "remote_support": "Yes",
      "adaptive_support": "Yes"
    }
  ]
}
```

## Evaluation

Run evaluation on training data:
```bash
python evaluate.py --dataset "Gen_AI Dataset.xlsx"
```

Expected output:
- Semantic Only: 20.56% Mean Recall@10
- Semantic + LLM: 22.11% Mean Recall@10

## Generate Submission

Generate predictions for test set:
```bash
python generate_submission.py --dataset "Gen_AI Dataset.xlsx" --output submission.csv
```

Format: Query, Assessment_url (one row per recommendation)

## Architecture

```
Query → Semantic Search (FAISS) → Top 20 Candidates
                                        ↓
                           LLM Re-ranking (Groq Llama-3.3-70B)
                                        ↓
                              Top 10 Balanced Results
                           (Hard Skills + Soft Skills)
```

## Technologies Used

- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS (IndexFlatIP)
- **LLM**: Groq (Llama-3.3-70B) - Free API
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Data Processing**: pandas, BeautifulSoup

## Troubleshooting

### Issue: Import errors
```bash
pip install -r requirements.txt
```

### Issue: API key not found
Create `.env` file with your Groq API key

### Issue: Vector store not found
The vector store files are pre-built and included in `data/` folder

### Issue: Port already in use
Change port in command:
```bash
uvicorn api:app --port 8001
streamlit run app.py --server.port 8502
```

## Development

### Rebuild Vector Index
```python
python -c "from src.vector_store import SHLVectorStore; s = SHLVectorStore(); s.build_index(); s.save()"
```

### Add New Assessments
1. Update `data/catalog_enriched.json`
2. Rebuild vector index
3. Restart API

## License

MIT License

## Contact

For questions about this project, please refer to the README.md
