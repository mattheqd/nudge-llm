# Testing the RAG Pipeline

Quick guide to test your RAG pipeline end-to-end.

## Quick Test (Recommended)

Run the automated test script:

```bash
python scripts/test_rag_pipeline.py
```

This will:
1. ✅ Check/create sample document chunks (if no PDFs found)
2. ✅ Test RAG service initialization
3. ✅ Test a direct RAG query
4. ✅ Guide you through API server testing

## Manual Testing Steps

### Step 1: Prepare Documents (if you have PDFs)

If you have PDF textbooks to process:

```bash
# Create textbooks directory
mkdir -p data/textbooks

# Place your PDF files in data/textbooks/

# Extract and chunk PDFs
python scripts/extract_text.py --input data/textbooks/ --output data/processed_chunks/
```

**OR** skip this for quick testing - the test script will create sample chunks automatically.

### Step 2: Test RAG Service Directly

Test the RAG service without the API server:

**Windows PowerShell (Recommended):**
```powershell
# Use single quotes for the JSON string
python scripts/rag_inference.py --query "How should I implement caching?" --chat-history '[{"role":"user","content":"My API is slow"}]' --scratchpad "Need low latency"
```

**Windows Command Prompt:**
```cmd
python scripts/rag_inference.py --query "How should I implement caching?" --chat-history "[{\"role\":\"user\",\"content\":\"My API is slow\"}]" --scratchpad "Need low latency"
```

**Or test with just a query (simpler):**
```powershell
python scripts/rag_inference.py --query "How should I implement caching?"
```

**Mac/Linux:**
```bash
python scripts/rag_inference.py \
  --query "How should I implement caching?" \
  --chat-history '[{"role":"user","content":"My API is slow"}]' \
  --scratchpad "Need low latency"
```

### Step 3: Start API Server

In one terminal, start the RAG API server:

```bash
python scripts/rag_api_server.py --port 5000
```

You should see:
```
Starting RAG API server on http://127.0.0.1:5000
 * Running on http://127.0.0.1:5000
```

### Step 4: Test API Endpoints

In another terminal, test the endpoints:

#### Health Check
```bash
curl http://localhost:5000/health
```

Expected response:
```json
{"status": "healthy"}
```

#### Get Design Suggestion
```bash
curl -X POST http://localhost:5000/api/rag/suggest \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How should I implement caching?",
    "chat_history": [{"role": "user", "content": "My API is slow"}],
    "scratchpad": "Need low latency, handle 10k users",
    "k": 3
  }'
```

Expected response:
```json
{
  "suggestion": "Implement Redis caching layer...",
  "references": [
    {
      "chunk_id": 1,
      "source": "sample_design_patterns.pdf",
      "preview": "To implement caching in a web application..."
    }
  ]
}
```

### Step 5: Test with Python requests

```python
import requests

response = requests.post(
    "http://localhost:5000/api/rag/suggest",
    json={
        "query": "How should I design an authentication system?",
        "chat_history": [],
        "scratchpad": "Security is top priority",
        "k": 3
    }
)

print(response.json())
```

## Troubleshooting

### "No chunks found"
- Run `extract_text.py` first, OR
- The test script will create sample chunks automatically

### "Cannot connect to API server"
- Make sure the server is running: `python scripts/rag_api_server.py --port 5000`
- Check that port 5000 is not in use: `netstat -ano | findstr :5000` (Windows)

### "API_KEY and DEPLOYMENT_ID must be set"
- Make sure your `.env` file is in the project root
- Check that `.env` has `API_KEY`, `DEPLOYMENT_ID`, `API_VERSION`, and `AZURE_ENDPOINT`

### "Index not found" or "FAISS index error"
- The index will be built automatically on first run
- If issues persist, delete `data/faiss_index/` and restart the server

## Next Steps

Once testing works:
1. Add your own PDF documents to `data/textbooks/`
2. Process them with `extract_text.py`
3. Rebuild the index (or restart server)
4. Integrate with your Node.js backend (see `README_RAG.md`)

