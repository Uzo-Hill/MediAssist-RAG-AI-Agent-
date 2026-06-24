# main.py
# This file turns your RAG agent into a real web API

from fastapi import FastAPI
from pydantic import BaseModel
import nest_asyncio

nest_asyncio.apply()

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# ── Step 1: Configure the LLM and Embedding Model ──────────────────
# This must match exactly what was used to build the original index

Settings.llm = Ollama(
    model="llama3.2",
    base_url="http://127.0.0.1:11434",
    request_timeout=900.0
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://127.0.0.1:11434"
)

# ── Step 2: Load the Persisted Index from Disk ──────────────────────
# This loads the index ONCE when the server starts — not on every request

print("Loading MediAssist index from disk...")
storage_context = StorageContext.from_defaults(persist_dir="./mediassist_storage")
index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(
    similarity_top_k=2,
    response_mode="compact"
)
print("✅ Index loaded. MediAssist API is ready.")

# ── Step 3: Create the FastAPI App ───────────────────────────────────
app = FastAPI(title="MediAssist RAG API")

# Defines what a request to this API must look like
class QuestionRequest(BaseModel):
    question: str

# Defines what the API will respond with
class AnswerResponse(BaseModel):
    answer: str

# ── Step 4: Define the Endpoint ──────────────────────────────────────
# This creates a URL: POST http://127.0.0.1:8000/ask
# Any application can send a question here and get an answer back

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    response = query_engine.query(request.question)
    return AnswerResponse(answer=str(response))

# ── Step 5: A Simple Health Check Endpoint ───────────────────────────
# Useful to quickly confirm the API is running

@app.get("/")
def health_check():
    return {"status": "MediAssist API is running"}