# Understanding RAG: A Beginner's Guide

This guide explains how the RAG (Retrieval Augmented Generation) system works in this project, focusing on `extract_text.py` and `rag_inference.py`.

## What is RAG?

**RAG = Retrieval Augmented Generation**

RAG enhances an LLM (like ZotGPT) by giving it access to external knowledge (your documents) rather than relying only on its training data.

**Simple Analogy:**
- Without RAG: A student taking an exam with only their memory
- With RAG: A student taking an exam with access to reference books (your documents)

---

## Part 1: `extract_text.py` - Preparing Your Documents

### What It Does

`extract_text.py` takes your PDF textbooks and converts them into searchable chunks that the RAG system can use.

### Step-by-Step Breakdown

#### 1. **PDF Text Extraction** (Lines 15-21)
```python
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text
```

**What's happening:**
- Opens the PDF file
- Reads each page
- Extracts all text from each page
- Combines it into one big string

**Why:** LLMs need text, not PDFs. We extract the raw text.

#### 2. **Text Chunking** (Lines 23-49)
```python
def chunk_text(text, tokenizer, max_tokens=512, overlap=50):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    # ... splits into chunks with overlap
```

**What's happening:**
- **Tokenization:** Converts text into tokens (numeric IDs representing words/subwords)
  - Example: "How are you?" → `[1045, 527, 345]` (token IDs)
  - **Note:** These are still text-based - we use them to measure size, then decode back to text
- **Chunking:** Splits the long text into smaller pieces (~512 tokens each)
- **Overlap:** Each chunk overlaps by 50 tokens with the next one
- **Decoding:** After chunking, tokens are converted back to text for storage in chunks

**Important Distinction:**
- We use tokens to measure size (512 tokens = right size for LLMs)
- But chunks are stored as **text** (human-readable)
- Vectors (embeddings) come later in `rag_inference.py`

**Why chunking?**
- LLMs have token limits (can't process entire books at once)
- Smaller chunks = more precise search results
- Overlap prevents losing context at chunk boundaries

**Example:**
```
Original text (text): "Design patterns are reusable solutions... [500 words]"

Step 1 - Tokenize (tokens): [1045, 527, 345, 2008, ...] (count: 800 tokens)

Step 2 - Chunk by tokens:
  Chunk 1 tokens: [1045...2008] (512 tokens)
  Chunk 2 tokens: [1950...3456] (512 tokens, 50 overlap)

Step 3 - Decode back to text (chunks are stored as text):
  Chunk 1: "Design patterns are reusable solutions... [text version]"
  Chunk 2: "...solutions... architectural patterns provide... [text version]"
```

**Key Point:** Chunks in the JSONL files are **text**, not tokens or vectors. We only use tokens temporarily to measure size for chunking.

#### 3. **Saving Chunks** (Lines 51-67)
```python
chunks.append({
    "chunk_id": chunk_id,
    "text": chunk_text,
    "token_count": len(chunk_tokens),
    "source_file": source_file
})
```

**What's happening:**
- Saves each chunk as a JSON object with metadata
- Stores in JSONL format (one JSON object per line)

**Why JSONL?**
- Easy to read and process line-by-line
- Each chunk keeps track of where it came from

---

## Part 2: `rag_inference.py` - The RAG Pipeline

### Overview: The 4 Steps of RAG

1. **Embed** - Convert chunks into vectors (numbers)
2. **Store** - Save vectors in FAISS (a searchable database)
3. **Retrieve** - Find relevant chunks for a query
4. **Generate** - Use retrieved chunks + query to generate nudge

---

### Step 1: Loading Chunks (`load_chunks`, Lines 27-49)

```python
def load_chunks(self):
    chunks = []
    for jsonl_file in self.chunks_dir.glob("*.jsonl"):
        chunk = json.loads(line)
        chunks.append(Document(
            page_content=chunk["text"],
            metadata={"chunk_id": chunk.get("chunk_id", 0), ...}
        ))
```

**What's happening:**
- Reads all `.jsonl` files from `data/processed_chunks/`
- Converts each chunk into a LangChain `Document` object
- Documents contain the text + metadata (where it came from)

**Why Document objects?**
- LangChain's standard format for text data
- Keeps text and metadata together

---

### Step 2: Building/Loading the FAISS Index (`build_or_load_index`, Lines 51-101)

#### What is FAISS?

**FAISS** (Facebook AI Similarity Search) is a library for fast similarity search in high-dimensional vectors.

**Simple Explanation:**
- Imagine each document chunk as a point in 384-dimensional space
- Similar chunks are close together in this space
- FAISS finds the closest neighbors quickly

#### The Embedding Process

```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
self.vectorstore = FAISS.from_documents(chunks, embeddings)
```

**What's happening:**

1. **Input:** Chunks are **text** (from JSONL files)
   - Example chunk: `"How to implement caching? Use Redis..."` (text)

2. **Embedding Model:** `all-MiniLM-L6-v2` converts text → 384 numbers (vector)
   - Input (text): `"How to implement caching?"`
   - Output (vector): `[0.23, -0.45, 0.78, ..., 0.12]` (384 numbers)
   - **This is the transformation from text to numbers!**
   
3. **Vector Conversion:** Every chunk (text) is converted to a vector (numbers)
   - Chunk 1 (text): `"Design patterns are reusable solutions..."`
     → Vector 1 (numbers): `[0.1, 0.2, -0.3, ..., 0.5]`
   - Chunk 2 (text): `"Caching strategies for web applications..."`
     → Vector 2 (numbers): `[0.3, 0.1, 0.4, ..., -0.2]`
   - Chunk 3 (text): `"API design principles..."`
     → Vector 3 (numbers): `[0.2, 0.4, 0.1, ..., 0.3]`

4. **FAISS Index:** Stores all vectors (numbers) in a searchable format
   - Think of it like a GPS system - it knows where everything is
   - Can quickly find "nearby" vectors (similar chunks)
   - Stores: `{chunk_id: vector, ...}` mapping

**Critical Transformation Point:**
- **Before embedding:** Chunks are text (stored in JSONL)
- **After embedding:** Chunks become vectors (stored in FAISS)
- **During search:** Query text → query vector → find similar chunk vectors → return original chunk text

**Why 384 dimensions?**
- More dimensions = more nuance (but slower)
- 384 is a good balance for speed and accuracy
- The embedding model was trained to put similar meanings close together

**Example:**
```
Query: "How to cache data?"
  ↓ (embedding)
Query Vector: [0.25, -0.33, 0.41, ...]

FAISS searches for closest vectors:
  ✓ "Caching strategies for web apps" - Vector: [0.27, -0.31, 0.39, ...] ← Very close!
  ✗ "Design patterns overview" - Vector: [0.10, 0.50, -0.20, ...] ← Far away
```

---

### Step 3: Retrieval (`generate_suggestion`, Lines 137-140)

```python
relevant_docs = self.vectorstore.similarity_search(query, k=3)
```

**What's happening:**
1. Query is embedded: "How should I implement caching?" → Vector
2. FAISS searches index for 3 closest vectors (most similar chunks)
3. Returns the actual text chunks (not just vectors)

**k=3 means:** Get the top 3 most relevant chunks

**Example:**
```
Query: "How should I implement caching?"

Retrieved chunks:
1. "Use Redis for caching user sessions. Set TTL to 1 hour..."
2. "Caching strategies: Write-through, write-back, cache-aside..."
3. "Implement cache invalidation when data updates..."
```

---

### Step 4: Generation (`generate_suggestion`, Lines 150-185)

```python
# Build context from retrieved chunks
context = "\n\n".join([f"[Reference {i}]\n{doc.page_content}" 
                       for i, doc in enumerate(relevant_docs, 1)])

# Build prompt with context + chat history + scratchpad
prompt = prompt_template.format(
    context_section=f"Relevant Knowledge:\n{context}",
    chat_section=f"Chat History:\n{chat_context}",
    scratchpad_section=f"Scratchpad Notes:\n{scratchpad}",
    query=query
)

# Generate nudge using ZotGPT
response = self.llm.invoke([HumanMessage(content=prompt)]).content
```

**What's happening:**

1. **Context Building:** Combines retrieved chunks into one text block
   ```
   Relevant Knowledge:
   [Reference 1]
   Use Redis for caching...
   
   [Reference 2]
   Caching strategies...
   ```

2. **Prompt Construction:** Builds a complete prompt with:
   - **Relevant Knowledge:** The 3 retrieved chunks
   - **Chat History:** Previous conversation
   - **Scratchpad:** User's notes
   - **Query:** Current question

3. **LLM Generation:** Sends everything to ZotGPT, which:
   - Reads all the context
   - Understands the query
   - Generates a gentle nudge based on the knowledge

**The Magic:** The LLM combines:
- Its general knowledge (from training)
- Specific knowledge (from your documents)
- Current conversation context
- User's scratchpad notes

---

## How It All Works Together

### Complete Flow Example

**User asks:** "How should I implement caching?"

1. **Query Embedding:**
   ```
   "How should I implement caching?"
   → [0.25, -0.33, 0.41, ...] (384 numbers)
   ```

2. **FAISS Search:**
   ```
   Searches 865 chunks in index
   Finds 3 closest matches:
     - Chunk 234: "Redis caching strategies..."
     - Chunk 567: "Cache invalidation patterns..."
     - Chunk 123: "TTL and expiration policies..."
   ```

3. **Context Assembly:**
   ```
   Relevant Knowledge:
   [Reference 1] Redis caching strategies...
   [Reference 2] Cache invalidation patterns...
   [Reference 3] TTL and expiration policies...
   
   Chat History:
   User: "My API is slow"
   Assistant: "What's the bottleneck?"
   
   Scratchpad:
   Need low latency, handle 10k concurrent users
   ```

4. **Prompt to LLM:**
   ```
   You are a software design mentor...
   
   Relevant Knowledge:
   [all the retrieved chunks]
   
   Chat History:
   [previous conversation]
   
   Query: "How should I implement caching?"
   
   Generate a gentle nudge...
   ```

5. **LLM Response:**
   ```
   "What are the tradeoffs between write-through and 
    cache-aside strategies for your use case?"
   ```

---

## Key NLP/LangChain Concepts

### Embeddings
- **What:** Converting text → numbers (vectors)
- **Why:** Computers can calculate similarity between numbers easily
- **Example:** "cat" and "kitten" have similar vectors (close together)

### Vector Similarity
- **What:** Measuring how "close" two vectors are
- **How:** Cosine similarity (angle between vectors)
- **Example:** Vectors pointing same direction = similar meaning

### Tokenization
- **What:** Breaking text into pieces (tokens) that the model understands
- **Why:** Models process tokens, not words directly
- **Example:** "don't" → ["don", "'", "t"] (3 tokens)
- **Important:** Tokens are still text-based - you can decode them back to text. They're just a different representation of the text.

### Text vs. Tokens vs. Vectors

**Three representations of the same information:**

1. **Text (Chunks):** 
   - Human-readable: `"How to implement caching?"`
   - What we work with in chunks

2. **Tokens:**
   - Text broken into pieces: `[1045, 527, 345, 2008, 7654]`
   - Still text-based (can decode back to text)
   - Used for chunking (to measure size: ~512 tokens per chunk)

3. **Vectors/Embeddings:**
   - Numbers: `[0.25, -0.33, 0.41, ..., 0.12]` (384 numbers)
   - **This is where text becomes numbers**
   - Used for similarity search in FAISS
   - Cannot convert back to original text (one-way transformation)

### Document Chunking
- **What:** Splitting long text into smaller pieces
- **Why:** Models have token limits; smaller pieces = better search
- **Trade-off:** Too small = lose context, Too large = less precise

### FAISS Index
- **What:** Fast similarity search database
- **Why:** Searching millions of vectors would be slow without it
- **How:** Uses approximate nearest neighbor algorithms

---

## Why This Architecture?

### Without RAG:
```
User: "How to implement caching?"
LLM: [Generic answer from training data]
```

### With RAG:
```
User: "How to implement caching?"
System: [Finds relevant chunks from YOUR documents]
LLM: [Answer based on YOUR specific knowledge base]
```

**Benefits:**
- Uses your specific knowledge (textbooks, docs)
- Can cite sources (references)
- More accurate for domain-specific questions
- Updatable (add new documents, rebuild index)

---

## Summary

### The Complete Flow

**Step 1: Document Processing (`extract_text.py`)**
```
PDF → Text → Tokenize (measure size) → Decode back to Text → Store as Chunks
```

**Step 2: Indexing (`rag_inference.py` - build_or_load_index)**
```
Load Chunks (text) → Embed as Vectors (numbers) → Store in FAISS Index
```

**Step 3: Querying (`rag_inference.py` - generate_suggestion)**
```
Query (text) → Embed as Vector → Search FAISS → Retrieve Chunk Text → LLM → Nudge
```

### Key Understanding

**Two-Stage Process:**

1. **Chunking (extract_text.py):**
   - Use tokenizer to enforce size constraints (~512 tokens)
   - Convert tokens back to text for storage
   - Chunks are **stored as text** in JSONL files

2. **Indexing (rag_inference.py):**
   - Load chunks (text) from JSONL
   - Embed chunks as vectors (text → numbers)
   - Index vectors in FAISS for fast similarity search

**Why This Approach?**
- **Tokens for sizing:** Ensures chunks fit within LLM token limits
- **Text for storage:** Human-readable, easy to debug, preserves original content
- **Vectors for search:** Enables fast similarity search (can't search text directly for meaning)

### The Complete Pipeline

```
PDF → Text Extraction
  ↓
Text → Tokenization (measure: ~512 tokens)
  ↓
Tokens → Decode back to Text → Store as Chunks (TEXT in JSONL)
  ↓
[Chunks loaded in rag_inference.py]
  ↓
Text Chunks → Embedding Model → Vectors (NUMBERS)
  ↓
Vectors → FAISS Index (for fast search)
  ↓
[User Query]
  ↓
Query Text → Embedding → Query Vector → Search FAISS → Retrieve Chunk Vectors
  ↓
Retrieve Original Chunk Text → Combine with Query → LLM → Generate Nudge
```

This allows the LLM to "read" your documents and provide answers based on them, rather than just its training data!

