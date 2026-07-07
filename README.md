# 🩺 MediAssist RAG AI Agent

A fully local, offline-capable Retrieval-Augmented Generation (RAG) health assistant
that answers questions about common African health conditions — grounded in a trusted
medical knowledge base, not LLM memory.


*![Intro_Image](https://github.com/Uzo-Hill/MediAssist-RAG-AI-Agent-/blob/main/Project_Image/RAG_Agent_Pipeline.jfif)*

---

## Overview

MediAssist is a Retrieval-Augmented Generation (RAG) AI assistant that answers health-related questions by retrieving information from a curated medical knowledge
base, not from the LLM's general training memory.

Built and deployed in two phases:

| Phase | Description |
|---|---|
| **Phase 1** | Single-document RAG prototype (diabetes only) in Jupyter Notebook |
| **Phase 2** | Multi-document, six-condition knowledge base deployed as a full three-tier application (FastAPI + Streamlit), running entirely offline on CPU-only hardware |

**ADTC 2026 Submission** - Africa Deep Tech Challenge, Laptop LLM Track, Healthcare & Medical domain.

---

##  Problem Statement

Access to reliable health information remains one of the most critical unmet needs across sub-Saharan Africa. Cloud-hosted medical AI tools require:

- Internet connectivity
- API fees per query
- Stable electricity

For a patient in Nigeria, a community health worker, or a student nurse in Africa, these are not minor frictions. They are blockers.

MediAssist addresses this by:

1. Running a full RAG pipeline entirely on a local CPU-only laptop
2. Answering questions from a trusted medical knowledge base, not from hallucination
3. Requiring zero internet at inference time — works fully offline once set up
4. Costing nothing to run — no API fees, no subscriptions

---

## Phase 2 Upgrade

### What Changed — Phase 2 Upgrade

The original Phase 1 prototype covered only diabetes and ran inside a Jupyter Notebook.
Phase 2 made the following significant upgrades:

### 1. Knowledge Base Expanded from 1 to 6 Conditions

| Was (Phase 1) | Now (Phase 2) |
|---|---|
| Diabetes only | Diabetes, Malaria, Hypertension, Tuberculosis, Maternal Health, Typhoid Fever |
| 1 document | 6 documents |
| Single-condition RAG | Multi-condition African health assistant |

### 2. Index Persistence

The original in-memory index was lost every time the notebook restarted.
Phase 2 persists the index to disk:

```python
# Save index to disk — survives restarts without rebuilding
index.storage_context.persist(persist_dir="./mediassist_storage")

# Load from disk on subsequent sessions — no reprocessing needed
storage_context = StorageContext.from_defaults(persist_dir="./mediassist_storage")
index = load_index_from_storage(storage_context)
```

### 3. Model Optimised for Speed

Switched from `llama3.2` (2B) to `llama3.2:1b` (1B) after benchmarking:

| Model | Raw Inference Time | Improvement |
|---|---|---|
| llama3.2 (2B) | 92.1 seconds | Baseline |
| llama3.2:1b (1B) | 45.0 seconds | **2× faster** ✅ |

### 4. Deployed as a Three-Tier Application

| Layer | Technology | File |
|---|---|---|
| Chat Interface | Streamlit | `chat_app.py` |
| Backend API | FastAPI + Uvicorn | `main.py` |
| Knowledge Store | LlamaIndex disk persistence | `mediassist_storage/` |

### 5. Chat Interface Redesigned

- Subtitle updated to reflect all six conditions
- Sidebar added with knowledge base info, model details, sample questions
- Response time shown on every answer
- Welcome message and medical disclaimer added
- Full dark theme applied

---

## 📚 Knowledge Base

Six conditions chosen for their high burden across sub-Saharan Africa:

| # | Document | Condition | African Relevance |
|---|---|---|---|
| 1 | `diabetes_guide.txt` | Diabetes Mellitus | Fastest-growing NCD burden in Africa |
| 2 | `malaria_guide.txt` | Malaria | Africa accounts for 94% of global cases |
| 3 | `hypertension_guide.txt` | Hypertension | Affects ~130 million African adults |
| 4 | `tuberculosis_guide.txt` | Tuberculosis | Africa carries 25% of global TB burden |
| 5 | `maternal_health_guide.txt` | Maternal Health | 70% of global maternal deaths in Africa |
| 6 | `typhoid_guide.txt` | Typhoid Fever | Endemic across West Africa including Nigeria |

Each document covers: overview, types, symptoms, risk factors, diagnosis, treatment,
prevention, African-specific statistics, and when to seek care.

---

## Solution Architecture

```text
┌─────────────────────────────────────┐
│   Streamlit Chat Interface           │  ← chat_app.py
│   http://localhost:8501              │  Browser-based Q&A
└──────────────┬──────────────────────┘
               │ HTTP POST /ask
               ▼
┌─────────────────────────────────────┐
│   FastAPI Backend                    │  ← main.py
│   http://127.0.0.1:8000             │  REST API layer
└──────────────┬──────────────────────┘
               │ loads persisted index
               ▼
┌─────────────────────────────────────┐
│   LlamaIndex Vector Store            │  ← mediassist_storage/
│   Disk-persisted embeddings          │  Survives restarts
└──────────────┬──────────────────────┘
               │ retrieves top-2 chunks
               ▼
┌─────────────────────────────────────┐
│   Ollama Local Runtime               │
│   LLaMA 3.2:1b + nomic-embed-text   │  100% offline
└─────────────────────────────────────┘
```

---

## 🔁 RAG Pipeline

### 1. Document Loading

All six health documents are loaded in a single call:

```python
documents = SimpleDirectoryReader(
    input_files=[
        "diabetes_guide.txt",
        "malaria_guide.txt",
        "hypertension_guide.txt",
        "tuberculosis_guide.txt",
        "maternal_health_guide.txt",
        "typhoid_guide.txt"
    ]
).load_data()
```

### 2. Embedding Generation

Each chunk is converted to a vector embedding using `nomic-embed-text` via Ollama:

```python
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://127.0.0.1:11434"
)
```

### 3. Index Building and Persistence

Built once, saved to disk, reloaded instantly on subsequent sessions:

```python
if os.path.exists("./mediassist_storage") and os.listdir("./mediassist_storage"):
    storage_context = StorageContext.from_defaults(persist_dir="./mediassist_storage")
    index = load_index_from_storage(storage_context)
else:
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir="./mediassist_storage")
```

### 4. Query Engine

```python
# top_k=2: retrieve 2 most relevant chunks per question
# compact mode: single LLM call — faster on CPU-only hardware
query_engine = index.as_query_engine(
    similarity_top_k=2,
    response_mode="compact"
)
```

### 5. FastAPI Endpoint

```python
@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    response = query_engine.query(request.question)
    return AnswerResponse(answer=str(response))
```




##  Solution Architecture

```text
Medical Document
       │
       ▼
Document Loader
       │
       ▼
Text Chunking
       │
       ▼
Embedding Generation
(nomic-embed-text)
       │
       ▼
Vector Index
(LlamaIndex)
       │
       ▼
Retriever
       │
       ▼
LLM (Llama 3.2)
       │
       ▼
Grounded Response
```

---

##  RAG Pipeline

### 1. Document Loading

The healthcare reference document is loaded into the system using LlamaIndex.

```python
# Load the text file
# SimpleDirectoryReader handles .txt files perfectly
documents = SimpleDirectoryReader(
    input_files=["diabetes_guide.txt"]
).load_data()

print(f" Document loaded! Total chunks: {len(documents)}")

```

### 2. Embedding Generation
The project uses Ollama's `nomic-embed-text` model to convert document chunks into vector embeddings for semantic search.

```python
# Reconfigure with a longer timeout and fewer chunks per query
Settings.llm = Ollama(
    model="llama3.2",
    base_url="http://127.0.0.1:11434",
    request_timeout=900.0   # 15 minutes — safely covers slow CPU responses
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://127.0.0.1:11434"
)

print(" LLM reconfigured")

```

### 3. Vector Index Creation
Embeddings are stored in a vector index for semantic retrieval.

```python
# This single line does three things automatically:
# 1. Splits the text into chunks
# 2. Converts each chunk into an embedding using nomic-embed-text
# 3. Stores all embeddings in a local vector index for fast retrieval

index = VectorStoreIndex.from_documents(
    documents,
    show_progress=True  # Shows a progress bar while building
)

print(" Vector index built successfully!")

```

### 4. Query Retrieval

User questions are converted into embeddings and matched against relevant document chunks.

### 5. Response Generation

Retrieved context is sent to the LLM to generate grounded answers.

---

##  Tech Stack

| Component | Technology |
|------------|------------|
| Programming Language | Python |
| RAG Framework | LlamaIndex |
| LLM | Llama 3.2 |
| Embedding Model | nomic-embed-text |
| Model Serving | Ollama |
| Vector Store | LlamaIndex Vector Index |
| Frontend | Streamlit |
| Backend API | FastAPI |

---

##  Project Structure

```text
MediAssist/
│
├── diabetes_guide.txt
├── main.py
├── chat_app.py
├── mediassist_storage/
├── requirements.txt
└── README.md
```
---

##  Installation

### Clone Repository
```bash
git clone https://github.com/yourusername/MediAssist.git
cd MediAssist
```

### Install Dependencies
```bash
pip install -r requirements.txt
```
### Install Required Models
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```
---

##  Running the Project

### Start Ollama

```bash
ollama serve
```

### Start FastAPI Backend

```bash
uvicorn main:app --reload
```

### Start Streamlit Frontend

```bash
streamlit run chat_app.py
```
---

##  Core Implementation

### LLM Configuration

```python
# Reconfigure with a longer timeout and fewer chunks per query
Settings.llm = Ollama(
    model="llama3.2",
    base_url="http://127.0.0.1:11434",
    request_timeout=900.0   # 15 minutes — safely covers slow CPU responses
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://127.0.0.1:11434"
)

print("✅ LLM reconfigured with extended timeout")
```
### Embedding Configuration
```python
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://127.0.0.1:11434"
)
```
### Query Engine
```python
query_engine = index.as_query_engine(
    similarity_top_k=2,
    response_mode="compact"   # Combines chunks into a single LLM call instead of multiple refine passes
)

print(" Query engine reconfigured for faster response on CPU")
```
---

##  Application Demo

### Chat Interface

*![AppUI](https://github.com/Uzo-Hill/MediAssist-RAG-AI-Agent-/blob/main/Project_Image/App_UI.PNG)*


##  Evaluation

The system successfully retrieves information from the healthcare document and generates context-grounded answers.

### Example:


#### Sample Questions and Answers

*![QnA](https://github.com/Uzo-Hill/MediAssist-RAG-AI-Agent-/blob/main/Project_Image/QnA.PNG)*


This demonstrates that the response was generated from retrieved document content rather than generic model knowledge.

---

##  Key Learnings

- Building Retrieval-Augmented Generation systems
- Vector embeddings and semantic search
- LlamaIndex document indexing
- Local LLM deployment with Ollama
- Grounding AI responses using external knowledge

---

##  Future Improvements

- Add source citations
- Support multiple medical documents
- Introduce conversation memory
- Deploy to cloud infrastructure
- Add PDF document upload support
- Implement confidence scoring
---

## ⚠️ Disclaimer

This project is intended for educational and demonstration purposes only.

It should not be used as a substitute for professional medical advice, diagnosis, or treatment.

---

##  Author

**Hillary C. Uzoh**

- GitHub: https://github.com/Uzo-Hill
- LinkedIn: https://www.linkedin.com/in/hillaryuzoh/

---






