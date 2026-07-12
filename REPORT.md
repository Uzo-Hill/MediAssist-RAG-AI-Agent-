# REPORT.md — MediAssist RAG AI Agent
## Africa Deep Tech Challenge 2026 — Laptop LLM Track

**Team:** Hillary C. Uzoh  
**Domain:** Healthcare & Medical  
**Submission:** MediAssist — A Local RAG-Powered Medical Q&A Assistant  
**GitHub:** https://github.com/Uzo-Hill/MediAssist-RAG-AI-Agent-  
**Date:** July 2026  

---

## 1. Problem Definition and Context

Access to reliable health information remains one of the most critical unmet needs across sub-Saharan Africa. For a patient in Nigeria, a community health worker, or a student nurse in Africa, they cannot easily query a clinical knowledge base without either internet access, a smartphone data plan, or proximity to a trained professional. Even where internet access exists, cloud-hosted medical AI tools require API fees, stable fibre connections, and sustained electricity — all of which are unreliable or unaffordable for millions of people on the continent.

The five conditions addressed by MediAssist — malaria, hypertension, tuberculosis, maternal health complications, and typhoid fever — collectively account for the majority of preventable deaths and healthcare visits across West and sub-Saharan Africa. Diabetes is also included given its rapidly growing burden on the continent. A system that can answer accurate, grounded questions about these conditions without any internet connection, cloud API, or per-query cost directly addresses a real access gap.

MediAssist is built on a core conviction: **a useful medical AI assistant does not need to be large, cloud-hosted, or expensive. It needs to be accurate, honest about what it knows, and runnable on the hardware that already exists.**

---

## 2. Identified Constraints

### Hardware Constraints
- **CPU only:** No dedicated GPU. All inference runs on 2 physical cores (4 logical cores).
- **RAM:** 7.7 GB total — matching the ADTC Standard Laptop profile (8 GB target).
- **Disk:** Standard laptop SSD. Model and index stored locally.
- **Electricity:** Designed to run on battery where necessary. No always-on server dependency.

### Connectivity Constraints
- **Zero cloud dependency:** No API calls to OpenAI, Anthropic, Google, or any external LLM provider.
- **No internet required at inference time:** Once models and documents are downloaded, MediAssist runs fully offline.
- **All data local:** The knowledge base, vector index, and model weights all reside on the user's machine.

### Model Constraints
- Smaller models (1B parameters) produce faster but sometimes less detailed responses than larger models.
- Response times on CPU-only hardware range from 45–155 seconds depending on available RAM.
- Without GPU acceleration, tokens per second (TPS) is significantly below cloud-hosted baselines.

### Data Constraints
- Knowledge base is limited to manually curated documents. It does not update automatically.
- The system answers only within the scope of its indexed documents — it will not fabricate answers from outside the knowledge base (this is a deliberate safety feature, not a limitation).

---

## 3. Design Alternatives and Final Decisions

### 3.1 LLM Selection

| Option | Considered | Decision |
|---|---|---|
| Cloud LLM (GPT-4, Claude) | Yes | ❌ Rejected — requires API fees and internet |
| LLaMA 3.2 (2B via Ollama) | Yes | ⚠️ Used in development, too slow for production |
| LLaMA 3.2:1b (1B via Ollama) | Yes | ✅ Selected — 2x faster, acceptable quality |
| Mistral 7B | Yes | ❌ Rejected — too large for 7.7 GB RAM constraint |

**Final decision:** LLaMA 3.2:1b via Ollama. Benchmarking showed a 2x speed improvement (92.1s → 45.0s raw inference) over the 2B model with acceptable answer quality on all six health domains. The model is Q4-quantised by default through Ollama, reducing memory footprint while preserving useful accuracy.

### 3.2 RAG Framework Selection

| Option | Considered | Decision |
|---|---|---|
| LlamaIndex | Yes | ✅ Selected |
| LangChain | Yes | ❌ More complex, heavier dependency tree |
| Custom retrieval | Yes | ❌ Rejected — reinventing a solved problem |

**Final decision:** LlamaIndex. Native support for local embedding models, simple document ingestion pipeline, built-in disk persistence, and clean integration with Ollama made it the most suitable choice for a constrained local deployment.

### 3.3 Vector Store Selection

| Option | Considered | Decision |
|---|---|---|
| ChromaDB | Yes | ❌ Rejected — onnxruntime DLL failure on Windows hardware |
| Pinecone | Yes | ❌ Rejected — cloud-hosted, violates offline constraint |
| LlamaIndex disk persistence | Yes | ✅ Selected |

**Final decision:** LlamaIndex's built-in `storage_context.persist()`. This saves the vector index directly to disk in a `mediassist_storage/` folder, surviving restarts without rebuilding, and requiring no external vector database dependency. For a single-machine, offline deployment this is both simpler and more reliable than a dedicated vector database.

### 3.4 Embedding Model Selection

**Selected:** `nomic-embed-text` via Ollama.  
**Reason:** Free, runs locally, produces high-quality semantic embeddings, and integrates natively with both Ollama and LlamaIndex. No API key required.

### 3.5 Application Architecture Decision

**Decision:** Three-tier architecture — persisted index → FastAPI backend → Streamlit chat interface.

This separates concerns cleanly:
- The index layer handles knowledge retrieval
- The FastAPI layer exposes the RAG pipeline as a callable API
- The Streamlit layer provides a usable chat interface

This architecture means the same FastAPI backend could serve a WhatsApp integration, a web widget, or a mobile interface without changing the core RAG logic.

---

## 4. Tools Used and Why

| Tool | Version | Purpose | Why Chosen |
|---|---|---|---|
| Python | 3.11+ | Core language | Universal, well-supported |
| Ollama | Latest | Local LLM runtime | Free, runs GGUF models locally, simple CLI |
| LLaMA 3.2:1b | 1B params | Language model | Fastest local model within RAM constraints |
| nomic-embed-text | Latest | Embedding model | Free, local, high quality semantic embeddings |
| LlamaIndex | 0.14.x | RAG framework | Native Ollama support, disk persistence, simple API |
| FastAPI | 0.138.0 | Backend API | Lightweight, fast, auto-generates /docs interface |
| Uvicorn | 0.49.0 | ASGI server | Standard FastAPI deployment server |
| Streamlit | Latest | Chat interface | Rapid UI development, Python-native |
| psutil | Latest | Resource monitoring | Measure RAM and CPU during inference |

---

## 5. Knowledge Base

MediAssist's knowledge base covers six African-relevant health conditions, manually curated into structured plain-text documents:

| Document | Condition | African Relevance |
|---|---|---|
| `diabetes_guide.txt` | Diabetes Mellitus | Fastest-growing NCD burden in Africa |
| `malaria_guide.txt` | Malaria | 94% of global cases in Africa |
| `hypertension_guide.txt` | Hypertension | Affects ~130 million African adults |
| `tuberculosis_guide.txt` | Tuberculosis | Africa carries 25% of global TB burden |
| `maternal_health_guide.txt` | Maternal Health | 70% of global maternal deaths in Africa |
| `typhoid_guide.txt` | Typhoid Fever | Endemic across West Africa including Nigeria |

Each document covers: overview, types/classification, symptoms, risk factors, diagnosis, treatment, prevention, African-specific context, global statistics, and when to seek care.

**Total documents:** 6  
**Index type:** Vector (semantic similarity)  
**Retrieval:** Top-2 most relevant chunks per query  
**Response mode:** Compact (single LLM call per query)  

---

## 6. Performance Tests and Benchmarks

### 6.1 Hardware Profile

```
CPU:          Intel processor, 2 physical cores, 4 logical cores
RAM:          7.7 GB total (matches ADTC Standard Laptop profile)
GPU:          None — CPU-only inference
Storage:      Local SSD
OS:           Windows 10/11
```

### 6.2 Model Speed Comparison

Measured as raw LLM inference time (no RAG pipeline overhead):

| Model | Parameters | Response Time | Improvement |
|---|---|---|---|
| llama3.2 | 2B | 92.1 seconds | Baseline |
| llama3.2:1b | 1B | 45.0 seconds | **2.0x faster** ✅ |

### 6.3 End-to-End RAG Pipeline Response Times

Measured through the full pipeline (retrieval + LLM + formatting):

| Query | Response Time |
|---|---|
| What are the symptoms of malaria? | 103.8 seconds |
| How is hypertension treated? | 91.6 seconds |
| What danger signs should a pregnant woman watch for? | 51.0 seconds |
| How does tuberculosis spread? | 49.4 seconds |

### 6.4 Full System Benchmark — Run 1 (Constrained Conditions)

*Represents typical real-world scenario with other applications running:*

```
Available RAM at start:   0.9 GB
Response time:            155.5 seconds
Peak RAM usage:           7.56 GB
Model RAM footprint:      0.82 GB
Average CPU usage:        79.8%
Peak CPU usage:           100.0%
Estimated tokens:         209
Estimated TPS:            1.34 tokens/sec
Performance score:        9.0 / 100
Efficiency score:         88.3 / 100
```

### 6.5 Full System Benchmark — Run 2 (Optimised Conditions)

*Measured after closing background applications to free RAM:*

```
Available RAM at start:   2.9 GB
Response time:            94.5 seconds
Peak RAM usage:           7.51 GB
Model RAM footprint:      2.69 GB
Average CPU usage:        28.5%
Peak CPU usage:           58.8%
Estimated tokens:         120
Estimated TPS:            1.27 tokens/sec
Performance score:        8.5 / 100
Efficiency score:         61.5 / 100
```

### 6.6 Key Observation on RAM and Inference Behaviour

An important hardware behaviour was observed during benchmarking: when system RAM is heavily constrained (Run 1, 0.9 GB free), Ollama operates with a minimal memory footprint (0.82 GB) but slower inference due to memory pressure. When more RAM is available (Run 2, 2.9 GB free), Ollama allocates more aggressively (2.69 GB) and inference is faster (94.5s vs 155.5s). This adaptive memory behaviour is relevant for understanding real-world deployment on constrained African hardware where background processes frequently compete for RAM.

### 6.7 ADTC Scoring Formula Estimates

Using the formula: `S = 0.50·Sacc + 0.30·Sperf + 0.20·Seff`

```
Performance contribution (0.30 × 9.0):   2.7 points
Efficiency contribution  (0.20 × 88.3):  17.7 points
Subtotal (without accuracy):              20.4 points
Accuracy score: pending official ADTC benchmark evaluation
```

The efficiency score is the system's strongest performance metric, reflecting the lightweight footprint of the 1B quantised model within the 7 GB RAM limit.

---

## 7. Retrieval Accuracy Validation

To confirm MediAssist retrieves answers from the actual document rather than the LLM's general training knowledge, a fact-verification test was performed:

**Test:** The system was asked for the exact global healthcare cost figure for diabetes — a specific statistic that exists only in the custom knowledge base document, not in common LLM training data.

**Query:** *"What is the global cost of diabetes-related healthcare?"*  
**Expected answer:** $760 billion annually (from `diabetes_guide.txt`)  
**MediAssist answer:** *"The global cost of diabetes related healthcare exceeds $760 billion annually."*  

✅ **Exact match confirmed.** This validates that the RAG retrieval pipeline is functioning correctly — answers are grounded in the indexed documents, not generated from model memory.

---

## 8. System Architecture

```
┌─────────────────────────────────────┐
│   Streamlit Chat Interface           │  chat_app.py
│   http://localhost:8501              │  Browser-based Q&A
└──────────────┬──────────────────────┘
               │ HTTP POST /ask
               ▼
┌─────────────────────────────────────┐
│   FastAPI Backend                    │  main.py
│   http://127.0.0.1:8000             │  REST API layer
└──────────────┬──────────────────────┘
               │ loads persisted index
               ▼
┌─────────────────────────────────────┐
│   LlamaIndex Vector Store            │  mediassist_storage/
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

## 9. Design Decisions Summary

| Decision | Choice Made | Reason |
|---|---|---|
| No cloud API | Fully local Ollama | Zero cost, zero connectivity dependency |
| Model size | 1B over 2B | 2x faster inference on CPU-only hardware |
| Vector store | LlamaIndex disk persistence | Avoids onnxruntime incompatibility, simpler, reliable |
| Knowledge scope | 6 African-relevant conditions | Targeted coverage of highest-burden diseases |
| Response mode | Compact (single LLM call) | Reduces inference time on constrained hardware |
| Architecture | Three-tier (index/API/UI) | Separates concerns, extensible to other frontends |

---

## 10. Limitations and Honest Disclosures

- **TPS is low** (1.27–1.34 tokens/sec) due to CPU-only inference with no GPU acceleration. This is an honest hardware constraint, not a software failure.
- **Response times** (45–155 seconds) are longer than cloud-hosted systems. This is expected and inherent to running a quantised LLM on a 2-core CPU.
- **Knowledge base is static.** Documents must be manually updated when medical guidelines change. A production system would require a document update pipeline.
- **Not a medical device.** MediAssist is an information assistant. It is not a diagnostic tool and should not replace professional clinical judgment.
- **English only.** The current version does not support African languages. This is identified as a future enhancement.

---

## 11. Future Improvements

- [ ] African language support — Yoruba, Igbo, Hausa, Swahili, Pidgin English
- [ ] Expand knowledge base to additional conditions (HIV/AIDS, sickle cell, cholera)
- [ ] Add source citations to every answer for clinical traceability
- [ ] Implement confidence thresholds — refuse to answer when retrieval score is below threshold
- [ ] WhatsApp Business API integration for community health worker deployment
- [ ] Quantisation experiments (Q2, Q3) to further reduce model size and improve TPS
- [ ] Automatic knowledge base update pipeline from trusted medical sources

---

## 12. Reproduction Instructions

### Prerequisites
- Windows, macOS, or Linux machine with 8 GB RAM
- Python 3.11+
- [Ollama](https://ollama.com) installed

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Uzo-Hill/MediAssist-RAG-AI-Agent-
cd MediAssist-RAG-AI-Agent-

# 2. Pull the required models
ollama pull llama3.2:1b
ollama pull nomic-embed-text

# 3. Install Python dependencies
pip install llama-index llama-index-llms-ollama llama-index-embeddings-ollama fastapi uvicorn streamlit requests nest_asyncio psutil

# 4. Start Ollama (if not already running)
ollama serve

# 5. Start the FastAPI backend (Terminal 1)
uvicorn main:app --reload

# 6. Start the Streamlit interface (Terminal 2)
streamlit run chat_app.py

# 7. Open your browser
# Chat interface: http://localhost:8501
# API docs:       http://127.0.0.1:8000/docs
```

---

*MediAssist is submitted to the Africa Deep Tech Challenge 2026 under the Healthcare & Medical domain.*  
*All inference runs fully offline on commodity CPU-only hardware.*  
*No paid APIs. No cloud dependencies. Built for the hardware Africa already has.*
