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

---

## 🛠️ Tech Stack

| Component | Technology | Version |
|---|---|---|
| Language | Python | 3.11+ |
| RAG Framework | LlamaIndex | 0.14.x |
| Language Model | LLaMA 3.2:1b via Ollama | 1B parameters |
| Embedding Model | nomic-embed-text via Ollama | Latest |
| Backend API | FastAPI + Uvicorn | 0.138.0 |
| Chat Interface | Streamlit | Latest |
| Index Storage | LlamaIndex disk persistence | Built-in |
| Resource Monitoring | psutil | Latest |


---

## 📂 Project Structure

```text
MediAssist_RAG/
│
├── diabetes_guide.txt           # Knowledge document 1
├── malaria_guide.txt            # Knowledge document 2
├── hypertension_guide.txt       # Knowledge document 3
├── tuberculosis_guide.txt       # Knowledge document 4
├── maternal_health_guide.txt    # Knowledge document 5
├── typhoid_guide.txt            # Knowledge document 6
│
├── mediassist_storage/          # Persisted vector index (auto-generated)
│   ├── docstore.json
│   ├── index_store.json
│   └── vector_store.json
│
├── main.py                      # FastAPI backend
├── chat_app.py                  # Streamlit chat interface
├── MediAssist_Master.ipynb      # Master notebook (clean, ordered)
├── REPORT.md                    # ADTC 2026 submission report
├── requirements.txt             # Python dependencies
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/download) installed

### 1. Clone the Repository

```bash
git clone https://github.com/Uzo-Hill/MediAssist-RAG-AI-Agent-.git
cd MediAssist-RAG-AI-Agent-
```

### 2. Pull the Required Models

```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

### 3. Install Python Dependencies

```bash
pip install llama-index \
            llama-index-llms-ollama \
            llama-index-embeddings-ollama \
            fastapi \
            uvicorn \
            streamlit \
            requests \
            nest_asyncio \
            psutil
```

> ⚠️ **Windows users:** If you have multiple Python environments (e.g. Anaconda
> and Miniconda), install dependencies into the **same environment** you will use
> to run `uvicorn` and `streamlit` from the terminal.

---

## Running the Project

MediAssist requires **three components running simultaneously**,
each in its own terminal window.

### Step 1 — Start Ollama

```bash
ollama serve
```

Confirm it is active: visit `http://localhost:11434` — should show `Ollama is running`

### Step 2 — Start the FastAPI Backend (Terminal 1)

```bash
cd MediAssist_RAG
uvicorn main:app --reload
```

Expected output:

```
Loading MediAssist index from disk...
✅ Index loaded. MediAssist API is ready.
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```



### Step 3 — Start the App Locally or with Streamlit (Terminal 2)

```bash
(base) C:\Users\DELL>cd Desktop\MediAssist_RAG

(base) C:\Users\DELL\Desktop\MediAssist_RAG>streamlit run chat_app.py

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501 
  Network URL: http://192.168.0.187:8501
```

Opens at `http://localhost:8501`

---

## 💻 Core Implementation

### LLM Configuration

```python
# llama3.2:1b — 2x faster than llama3.2 (2B) on CPU-only hardware
Settings.llm = Ollama(
    model="llama3.2:1b",
    base_url="http://127.0.0.1:11434",
    request_timeout=300.0
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://127.0.0.1:11434"
)
```

### FastAPI Backend (main.py)

```python
from fastapi import FastAPI
from pydantic import BaseModel
from llama_index.core import StorageContext, load_index_from_storage, Settings

app = FastAPI(title="MediAssist RAG API")

storage_context = StorageContext.from_defaults(persist_dir="./mediassist_storage")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine(similarity_top_k=2, response_mode="compact")

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    response = query_engine.query(request.question)
    return AnswerResponse(answer=str(response))

@app.get("/")
def health_check():
    return {"status": "MediAssist API is running"}
```

---

## 📊 Benchmarks

### Hardware Profile

```
CPU:    2 physical cores, 4 logical cores (Intel)
RAM:    7.7 GB total
GPU:    None — CPU-only inference
OS:     Windows 10/11
```

### Model Speed Comparison

| Model | Parameters | Raw Inference | Improvement |
|---|---|---|---|
| llama3.2 | 2B | 92.1 seconds | Baseline |
| llama3.2:1b | 1B | 45.0 seconds | **2× faster** ✅ |

### End-to-End RAG Response Times

| Query | Response Time |
|---|---|
| What are the symptoms of malaria? | 103.8 seconds |
| How is hypertension treated? | 91.6 seconds |
| What danger signs occur during pregnancy? | 51.0 seconds |
| How does tuberculosis spread? | 49.4 seconds |

### Full System Benchmark

| Metric | Constrained (0.9 GB free RAM) | Optimised (2.9 GB free RAM) |
|---|---|---|
| Response time | 155.5 seconds | 94.5 seconds |
| Peak RAM | 7.56 GB | 7.51 GB |
| Model RAM footprint | 0.82 GB | 2.69 GB |
| Average CPU | 79.8% | 28.5% |
| Estimated TPS | 1.34 | 1.27 |
| Performance score | 9.0 / 100 | 8.5 / 100 |
| Efficiency score | 88.3 / 100 | 61.5 / 100 |

**Key finding:** When RAM is constrained, Ollama uses a smaller footprint (0.82 GB) but runs slower. When RAM is available, it allocates more aggressively (2.69 GB) for faster inference. This adaptive behaviour is characteristic of running LLMs on constrained hardware where background processes compete for memory.

---

## 📱 Application Demo

### Chat Interface

*![streamlit_Interface](https://github.com/Uzo-Hill/MediAssist-RAG-AI-Agent-/blob/main/Project_Image/Chat_Interface.PNG)*



### Sample Question and Answer

*![Prompt_Example](https://github.com/Uzo-Hill/MediAssist-RAG-AI-Agent-/blob/main/Project_Image/chat_Examp.PNG)*


---

## Evaluation

### Domain Coverage Test

| Domain | Sample Query | Result |
|---|---|---|
| Malaria | What are the symptoms of malaria? | ✅ Correct |
| Hypertension | How is high blood pressure treated? | ✅ Correct |
| Maternal Health | What danger signs occur during pregnancy? | ✅ Correct |
| Tuberculosis | How does tuberculosis spread? | ✅ Correct |
| Diabetes | What are the risk factors for diabetes? | ✅ Correct |
| Typhoid | How can typhoid fever be prevented? | ✅ Correct |

### Hallucination / Retrieval Validation Test

To confirm MediAssist retrieves from the document — not from LLM general knowledge —
a fact-verification test was run using a specific statistic that only exists in the
custom knowledge base:

**Query:**
```
What is the exact global cost of diabetes-related healthcare annually?
```

**Expected:** `$760 billion annually` *(from diabetes_guide.txt only)*

**MediAssist answer:**

*![Prompt_Example](https://github.com/Uzo-Hill/MediAssist-RAG-AI-Agent-/blob/main/Project_Image/RAG_Confirmation.PNG)*

**Exact match confirmed** — RAG retrieval is functioning correctly.

---

## Limitations and Honest Disclosures

- **TPS is low** (1.27–1.34 tokens/sec) due to CPU-only inference with no GPU acceleration. This is an honest hardware constraint, not a software failure.
- **Response times** (45–155 seconds) are longer than cloud-hosted systems. This is expected and inherent to running a quantised LLM on a 2-core CPU.
- **Knowledge base is static.** Documents must be manually updated when medical guidelines change. A production system would require a document update pipeline.
- **Not a medical device.** MediAssist is an information assistant. It is not a diagnostic tool and should not replace professional clinical judgment.
- **English only.** The current version does not support African languages. This is identified as a future enhancement.

---

## Key Learnings

- Building a multi-document RAG system with LlamaIndex and local Ollama models
- Diagnosing real deployment challenges — cross-environment Python dependencies,
  CPU inference timeout tuning, and Windows DLL compatibility issues
- Performance trade-off between model size and response quality on constrained
  hardware (1B vs 2B parameters)
- How Ollama adaptively manages memory — allocating more aggressively when RAM
  is available, conserving when constrained
- Separating a notebook prototype into a production-style three-tier application
  (persistence layer → API → interface)
- Engineering judgment to pivot from ChromaDB to LlamaIndex disk persistence
  when a dependency fails — choosing simplicity over convention

---

## Future Improvements

- [ ] African language support — Yoruba, Igbo, Hausa, Swahili, Pidgin English
- [ ] Expand knowledge base to additional conditions (HIV/AIDS, sickle cell, cholera)
- [ ] Add source citations to every answer for clinical traceability
- [ ] Implement confidence thresholds — refuse when retrieval score is too low
- [ ] WhatsApp Business API integration for community health worker deployment
- [ ] Quantisation experiments (Q2, Q3) to improve TPS on constrained hardware
- [ ] Automatic knowledge base update pipeline from trusted medical sources
- [ ] Docker containerisation for reproducible deployment
- [ ] Add PDF document upload support


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






