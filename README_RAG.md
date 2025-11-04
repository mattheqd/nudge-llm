# RAG API Pipeline with ZotGPT

This project provides a LangChain RAG (Retrieval Augmented Generation) API service that uses ZotGPT to generate software design suggestions based on chat history, scratchpad notes, and retrieved knowledge from document chunks.

**Note:** This is just the API pipeline. Frontend/backend integration is not yet implemented.

## Architecture

1. **Document Processing** (`scripts/extract_text.py`)
   - Extracts text from PDF textbooks
   - Chunks text into ~512 token pieces with overlap
   - Saves chunks as JSONL files

2. **RAG Service** (`scripts/rag_inference.py`)
   - Builds FAISS vector index from document chunks
   - Retrieves relevant chunks based on query similarity
   - Uses ZotGPT (Azure OpenAI) to generate suggestions
   - Incorporates chat history and scratchpad context

3. **Python API Server** (`scripts/rag_api_server.py`)
   - Flask server that exposes RAG functionality via HTTP API
   - Can be called from Node.js backend
   - Handles health checks and index rebuilding

4. **API Endpoints** (ready for Node.js integration)
   - REST API that can be called from any backend
   - Returns OpenAI-compatible format for easy integration

## Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file with:

```env
# ZotGPT API Configuration
API_KEY=your_api_key
DEPLOYMENT_ID=your_deployment_id
API_VERSION=2023-05-15
AZURE_ENDPOINT=https://azureapi.zotgpt.uci.edu

# RAG Configuration
CHUNKS_DIR=data/processed_chunks
FAISS_INDEX_DIR=data/faiss_index
RAG_API_URL=http://127.0.0.1:5000
```

### 3. Process Documents

Extract and chunk PDF textbooks:

```bash
python scripts/extract_text.py --input data/textbooks/book.pdf --output data/processed_chunks/
```

Or process a directory of PDFs:

```bash
python scripts/extract_text.py --input data/textbooks/ --output data/processed_chunks/
```

### 4. Start RAG API Server

```bash
python scripts/rag_api_server.py --port 5000
```

The server will automatically build the FAISS index on first startup if it doesn't exist.

## Usage

### API Server

The RAG API server provides REST endpoints that can be called from any backend service:

### Direct Python Usage

You can also use RAG directly from Python:

```python
from scripts.rag_inference import RAGService

rag = RAGService()
rag.build_or_load_index()
rag.initialize_llm()

result = rag.generate_suggestion(
    query="How should I implement real-time updates?",
    chat_history=[{"role": "user", "content": "I need notifications"}],
    scratchpad="Need to handle 10k concurrent users",
    k=3
)

print(result["suggestion"])
print(result["references"])
```

### Command Line

```bash
python scripts/rag_inference.py \
  --query "How should I design an authentication system?" \
  --chat-history '[{"role":"user","content":"Need secure login"}]' \
  --scratchpad "Security is top priority" \
  --k 3
```

## API Endpoints

### POST `/api/rag/suggest`

Generate design suggestion with RAG.

**Request:**
```json
{
  "query": "How should I implement caching?",
  "chat_history": [
    {"role": "user", "content": "My API is slow"},
    {"role": "assistant", "content": "What's the bottleneck?"}
  ],
  "scratchpad": "Need low latency",
  "k": 3
}
```

**Response:**
```json
{
  "suggestion": "Implement Redis caching layer...",
  "references": [
    {
      "chunk_id": 1,
      "source": "data/textbooks/book.pdf",
      "preview": "Caching strategies for distributed systems..."
    }
  ]
}
```

### POST `/api/rag/rebuild-index`

Rebuild the FAISS index from chunks (useful after adding new documents).

### GET `/health`

Health check endpoint.

## How It Works

1. **Query Processing**: The last user message is extracted as the query
2. **Document Retrieval**: FAISS vector search finds top-k relevant chunks
3. **Context Building**: Retrieved chunks + chat history + scratchpad are combined
4. **Prompt Construction**: A structured prompt includes all context sections
5. **Generation**: ZotGPT generates a design suggestion based on the enriched context
6. **Response**: Returns suggestion with source references

## Integration Example (for future implementation)

When you're ready to integrate, you can call the RAG API from your Node.js backend:

```javascript
// Example: Call RAG API from Node.js
const ragResponse = await fetch('http://127.0.0.1:5000/api/rag/suggest', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: lastUserMessage.content,
    chat_history: messages.slice(0, -1),
    scratchpad: scratchpadText,
    k: 3
  })
});

const ragResult = await ragResponse.json();
// ragResult.suggestion contains the design suggestion
// ragResult.references contains source references
```

## Customization

### Adjust Retrieval Count

Change `k` parameter (default: 3 chunks):

```javascript
// In chatService.js
body: JSON.stringify({
  // ...
  k: 5  // Retrieve 5 chunks instead of 3
})
```

### Change Embedding Model

Edit `scripts/rag_inference.py`:

```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"  # Better but slower
)
```

### Modify Chunk Size

```bash
python scripts/extract_text.py \
  --input data/textbooks/book.pdf \
  --output data/processed_chunks/ \
  --max-tokens 1024 \
  --overlap 100
```

## Troubleshooting

**Index not found**: Run `extract_text.py` first to create chunks, then restart the API server.

**RAG API connection error**: Ensure the Flask server is running on port 5000 (or update `RAG_API_URL`).

**No chunks retrieved**: Check that chunks directory exists and contains `.jsonl` files.

**ZotGPT API errors**: Verify `API_KEY`, `DEPLOYMENT_ID`, and `AZURE_ENDPOINT` in `.env`.

