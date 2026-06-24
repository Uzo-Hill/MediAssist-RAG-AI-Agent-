# MediAssist-RAG-AI-Agent

A Retrieval-Augmented Generation (RAG) AI assistant that answers health-related questions using information retrieved from a trusted medical document instead of relying solely on Large Language Model (LLM) knowledge.

The project combines **LlamaIndex**, **Ollama**, and **Vector Search** to deliver grounded and context-aware medical responses.

---
## Primary Objectives
To build, evaluate, and deploy a predictive model that accurately forecasts house prices per unit area using property characteristics, and to create a user-friendly web application for real-time predictions.

---

##  Project Overview
MediAssist was built to demonstrate how Retrieval-Augmented Generation (RAG) can reduce hallucinations and improve answer reliability by retrieving information from a custom healthcare knowledge base before generating responses.

### Key Features
- Document-based question answering
- Local LLM inference using Ollama
- Vector search retrieval
- Context-grounded responses
- Fast and lightweight deployment
- No paid API required

---

##  Problem Statement

Traditional LLMs may generate inaccurate or fabricated medical information.

MediAssist addresses this challenge by:

1. Retrieving relevant information from a medical document.
2. Passing retrieved context to the LLM.
3. Generating answers grounded in verified content.

---

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






