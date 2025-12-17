# SHL Assessment Recommendation Engine

An intelligent recommendation system for SHL assessments using semantic search and LLM re-ranking.

## ✅ Requirements Checklist

| Requirement | Status | Details |
|------------|--------|---------|
| Crawl SHL Catalog | ✅ Done | 377 Individual Test Solutions crawled |
| Build Recommendation Engine | ✅ Done | Semantic search + LLM re-ranking |
| Natural Language Query Support | ✅ Done | Text/JD input supported |
| Return 1-10 Assessments | ✅ Done | Returns top 10 relevant assessments |
| Assessment Name + URL | ✅ Done | Full metadata included |
| API Endpoint (JSON) | ✅ Done | FastAPI at `/recommend` |
| Web Frontend | ✅ Done | Streamlit app |
| GitHub Repository | ✅ Done | Ready for submission |
| Submission CSV | ✅ Done | `submission.csv` generated |
| Mean Recall@10 Evaluation | ✅ Done | 22.11% with LLM reranking |
| LLM Integration | ✅ Done | Groq Llama-3.3-70B (Free) |
| Balanced Recommendations | ✅ Done | Hard + Soft skills mix |

## 🌟 Features

- **Semantic Search**: sentence-transformers embeddings (all-MiniLM-L6-v2)
- **LLM Re-ranking**: Groq Llama-3.3-70B for intelligent re-ordering
- **377 Assessments**: Complete SHL Individual Test Solutions catalog
- **REST API**: FastAPI backend with OpenAPI docs
- **Web Interface**: Streamlit frontend for testing

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/shl-recommendation-engine.git
cd shl-recommendation-engine

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key (Optional)

For LLM re-ranking, get a **free** HuggingFace API key:

1. Go to https://huggingface.co and create a free account
2. Go to **Settings** → **Access Tokens** (https://huggingface.co/settings/tokens)
3. Click **"New token"** → Name it → Select **"Read"** → Create
4. Copy the token (starts with `hf_...`)

```bash
# Create .env file
echo "HF_API_KEY=hf_your_token_here" > .env
```

### 3. Run the Application

**Option A: Streamlit Frontend**
```bash
streamlit run app.py
```
Opens at http://localhost:8501

**Option B: FastAPI Backend**
```bash
uvicorn api:app --reload
```
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

## 📊 API Usage

### POST /recommend

```python
import requests

response = requests.post("http://localhost:8000/recommend", json={
    "query": "Looking for Java developers with team collaboration skills",
    "top_k": 10,
    "use_llm": True
})

print(response.json())
```

### Response Format

```json
{
  "query": "Looking for Java developers...",
  "method": "semantic+llm_rerank",
  "recommendations": [
    {
      "rank": 1,
      "assessment_name": "Core Java (Entry Level) - New",
      "url": "https://www.shl.com/products/...",
      "test_types": ["Knowledge & Skills"],
      "reason": "Directly assesses Java programming skills"
    }
  ],
  "processing_time_ms": 250.5
}
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  User Query     │───▶│  Semantic Search │───▶│  LLM Re-ranking │
│  (Job Desc/URL) │    │  (FAISS + MiniLM)│    │  (Gemini Flash) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  Top-20 Candidates│───▶│  Top-10 Balanced│
                       │  (by similarity)  │    │  (hard+soft mix)│
                       └──────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
shl-recommendation-engine/
├── api.py                        # FastAPI backend (POST /recommend, GET /health)
├── app.py                        # Streamlit frontend
├── pipeline.py                   # End-to-end pipeline
├── evaluate.py                   # Evaluation script (Mean Recall@10)
├── generate_submission.py        # Generate submission CSV
├── submission.csv                # Predictions for test set
├── requirements.txt              # Dependencies
├── README.md                     # Documentation
├── .env.example                  # Environment template
├── data/
│   ├── catalog_enriched.json     # 377 Individual Test Solutions
│   ├── vector_store.faiss        # FAISS vector index
│   └── metadata.pkl              # Test metadata
├── src/
│   ├── crawler.py                # Web crawler for SHL catalog
│   └── vector_store.py           # Vector store implementation
└── llm/
    ├── prompt.txt                # LLM re-ranking prompt
    └── rerank.py                 # Groq LLM re-ranking module
```

## 📈 Evaluation Results

Mean Recall@10 on the training dataset:

| Method | Mean Recall@10 |
|--------|----------------|
| Semantic Only | 20.56% |
| **Semantic + LLM** | **22.11%** |

Run evaluation:
```bash
python evaluate.py --dataset "Gen_AI Dataset.xlsx"
```

Generate submission:
```bash
python generate_submission.py --dataset "Gen_AI Dataset.xlsx" --output submission.csv
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Groq API key (FREE at console.groq.com) | For LLM re-ranking |

### Vector Store Settings

- Model: `all-MiniLM-L6-v2` (384 dimensions)
- Index: FAISS IndexFlatIP (cosine similarity)
- Candidate pool: 20 for re-ranking, configurable

## 📝 Assessment Types

| Code | Type | Description |
|------|------|-------------|
| K | Knowledge & Skills | Technical/hard skills (programming, etc.) |
| P | Personality | Personality assessments (OPQ, etc.) |
| B | Behavioral | Behavioral competencies |
| A | Ability | Cognitive abilities |
| S | Simulation | Job simulations |

## 🛠️ Development

### Rebuild Vector Index

```bash
python -c "from src.vector_store import SHLVectorStore; s = SHLVectorStore(); s.build_index(); s.save()"
```

### Run Tests

```bash
python -m pytest tests/
```

## 📜 License

MIT License

## 🙏 Acknowledgments

- [SHL](https://www.shl.com) for the comprehensive assessment catalog
- [Sentence Transformers](https://www.sbert.net/) for the embedding models
- [Groq](https://groq.com/) for providing free LLM API access
