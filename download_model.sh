#!/bin/bash
# download_model.sh
# MediAssist — Model download script for ADTC 2026 evaluation
# Downloads llama3.2:1b and nomic-embed-text via Ollama
# No credentials required — all models are publicly available

set -e

echo "=================================================="
echo "  MediAssist — Model Download Script"
echo "  Africa Deep Tech Challenge 2026"
echo "=================================================="

# Create model directory if it doesn't exist
mkdir -p model

# Check Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed."
    echo "   Please install it from https://ollama.com/download"
    exit 1
fi

echo ""
echo "[1/2] Downloading llama3.2:1b (1B parameter model)..."
ollama pull llama3.2:1b
echo "      ✅ llama3.2:1b ready"

echo ""
echo "[2/2] Downloading nomic-embed-text (embedding model)..."
ollama pull nomic-embed-text
echo "      ✅ nomic-embed-text ready"

echo ""
echo "=================================================="
echo "  ✅ All models downloaded successfully"
echo ""
echo "  Next steps:"
echo "  1. pip install -r requirements.txt"
echo "  2. python setup.py"
echo "  3. uvicorn main:app --reload"
echo "  4. streamlit run chat_app.py"
echo "  5. Open http://localhost:8501"
echo "=================================================="
