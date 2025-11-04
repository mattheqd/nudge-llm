# Setup Instructions

## Step 1: Create Virtual Environment

### Windows (PowerShell)
```powershell
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1
```

### Windows (Command Prompt)
```cmd
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate.bat
```

### Mac/Linux
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

## Step 2: Install Dependencies

Once your virtual environment is activated (you'll see `(venv)` in your prompt), install all requirements:

```bash
pip install -r requirements.txt
```

This will install:
- LangChain and related packages
- Transformers and ML libraries
- Flask for the API server
- PDF processing libraries
- FAISS for vector search
- And all other dependencies

## Step 3: Verify Installation

Check that key packages are installed:

```bash
python -c "import langchain; import flask; import faiss; print('✅ All packages installed')"
```

## Step 4: Create .env File

Create a `.env` file in the project root with your ZotGPT credentials:

```env
# ZotGPT API Configuration
DEPLOYMENT_ID=gpt-4o
API_VERSION=2024-02-01
API_KEY=c027e560ac7d4de1aa54247e9d2c2dd0
AZURE_ENDPOINT=https://azureapi.zotgpt.uci.edu

# RAG Configuration
CHUNKS_DIR=data/processed_chunks
FAISS_INDEX_DIR=data/faiss_index
RAG_API_URL=http://localhost:5000
```

## Step 5: You're Ready!

Now you can test the pipeline:

```bash
# Run automated test
python scripts/test_rag_pipeline.py

# Or start the API server
python scripts/rag_api_server.py --port 5000
```

## Troubleshooting

### "python is not recognized"
- Use `python3` instead of `python` on Mac/Linux
- Or add Python to your PATH

### "Activate.ps1 is not digitally signed" (Windows PowerShell)
Run this first to allow scripts:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Installation fails for specific packages
- Make sure you have Python 3.8+ installed
- On Windows, you may need Visual C++ Build Tools for some packages
- Try installing packages individually to identify the issue

### Port 5000 already in use
Use a different port:
```bash
python scripts/rag_api_server.py --port 5001
```

