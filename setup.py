# setup.py
# MediAssist — One-time setup script
# Run this once on any new machine before starting the application
#
# Usage:
#   1. Make sure Ollama is installed and running (ollama serve)
#   2. Run: python setup.py
#   3. Then start the app:
#        Terminal 1: uvicorn main:app --reload
#        Terminal 2: streamlit run chat_app.py

import os
import sys
import subprocess

print("=" * 55)
print("  MediAssist — Setup Script")
print("  Africa Deep Tech Challenge 2026")
print("=" * 55)

# ── Step 1: Check Ollama is running ──────────────────────────
print("\n[1/3] Checking Ollama is running...")
try:
    import requests
    r = requests.get("http://127.0.0.1:11434", timeout=5)
    print("      ✅ Ollama is running")
except Exception:
    print("      ❌ Ollama is not running.")
    print("         Please start it first with: ollama serve")
    print("         Then re-run this script.")
    sys.exit(1)

# ── Step 2: Pull required models ─────────────────────────────
print("\n[2/3] Pulling required Ollama models...")
print("      Pulling llama3.2:1b (this may take a few minutes)...")
subprocess.run(["ollama", "pull", "llama3.2:1b"], check=True)
print("      Pulling nomic-embed-text...")
subprocess.run(["ollama", "pull", "nomic-embed-text"], check=True)
print("      ✅ Models ready")

# ── Step 3: Build or load the vector index ───────────────────
print("\n[3/3] Setting up MediAssist knowledge index...")

import nest_asyncio
nest_asyncio.apply()

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    load_index_from_storage
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Configure LLM and embedding model
Settings.llm = Ollama(
    model="llama3.2:1b",
    base_url="http://127.0.0.1:11434",
    request_timeout=300.0
)
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://127.0.0.1:11434"
)

INDEX_PATH = "./mediassist_storage"

if os.path.exists(INDEX_PATH) and os.listdir(INDEX_PATH):
    # Index already exists — load and confirm it works
    print("      Existing index found. Loading to verify...")
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_PATH)
    index = load_index_from_storage(storage_context)
    print("      ✅ Index loaded successfully — no rebuild needed")
else:
    # No index found — build from all six documents
    print("      No existing index found. Building from documents...")

    required_docs = [
        "diabetes_guide.txt",
        "malaria_guide.txt",
        "hypertension_guide.txt",
        "tuberculosis_guide.txt",
        "maternal_health_guide.txt",
        "typhoid_guide.txt"
    ]

    # Confirm all documents are present
    missing = [d for d in required_docs if not os.path.exists(d)]
    if missing:
        print(f"      ❌ Missing documents: {missing}")
        print("         Ensure all .txt knowledge files are in the project folder.")
        sys.exit(1)

    documents = SimpleDirectoryReader(
        input_files=required_docs
    ).load_data()

    print(f"      Loaded {len(documents)} documents. Generating embeddings...")

    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True
    )
    index.storage_context.persist(persist_dir=INDEX_PATH)
    print("      ✅ Index built and saved to mediassist_storage/")

# ── Done ──────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  ✅ Setup complete!")
print("")
print("  Start the application:")
print("")
print("  Terminal 1:  uvicorn main:app --reload")
print("  Terminal 2:  streamlit run chat_app.py")
print("")
print("  Chat UI :  http://localhost:8501")
print("  API docs:  http://127.0.0.1:8000/docs")
print("=" * 55)
