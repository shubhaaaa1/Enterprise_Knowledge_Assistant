# Enterprise RAG System

An intelligent document Q&A system that lets you chat with your documents, code, and knowledge bases using AI. Upload files, connect data sources, and get instant answers with citations.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-orange)
![Groq](https://img.shields.io/badge/Groq-API-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## What is This?

Enterprise RAG is a **Retrieval-Augmented Generation** system that turns your documents into an intelligent Q&A assistant. Think of it as ChatGPT for your own documents - upload files, ask questions, and get accurate answers with source citations.

### Key Features

🚀 **Ultra-Fast Responses** - Get answers in 1-2 seconds using Groq API  
📁 **Folder Upload** - Upload entire folders of documents at once  
🔍 **Smart Search** - Hybrid semantic + keyword search finds the most relevant content  
📚 **Source Citations** - Every answer includes citations with excerpts from source documents  
💬 **Conversational** - Multi-turn conversations with context memory  
🎯 **Accurate** - Grounding scores show how well answers are supported by your documents  
️ **Easy Management** - List and delete uploaded sources anytime  
🌐 **Open Access** - No authentication required - simple and straightforward

---

## Quick Demo

1. **Upload your documents** (drag & drop folders)
2. **Ask questions** in natural language
3. **Get instant answers** with citations showing exactly where the information came from

Example:
```
You: "What is the refund policy?"
AI: "The refund policy allows returns within 30 days [1]. 
     Full refunds are provided for unused items [2]."
     
     [1] refund-policy.pdf - "Customers may return items within 30 days..."
     [2] terms.txt - "Full refunds will be issued for items in original condition..."
```

---

## Getting Started

### Prerequisites

- **Python 3.11+** installed
- **Internet connection** (for Groq API)
- **Groq API key** (free at https://console.groq.com)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/enterprise-rag.git
cd enterprise-rag
```

2. **Install dependencies**
```bash
pip install -e ".[dev]"
pip install python-multipart sentence-transformers
```

3. **Get a Groq API key** (required)
   - Go to https://console.groq.com
   - Sign up for free
   - Create an API key
   - Copy the key (starts with `gsk_`)

4. **Configure environment**

Create a `.env` file or set environment variables:

```bash
# Required
GROQ_API_KEY=gsk_your_api_key_here

# Optional
GROQ_MODEL=llama-3.1-8b-instant  # Default model
CHROMA_PATH=./chroma_data        # Vector database location
LOG_DIR=logs                     # Log file directory
```

5. **Start the application**

**Windows (PowerShell):**
```powershell
$env:GROQ_API_KEY="gsk_your_api_key_here"
uvicorn enterprise_rag.api:app --host 0.0.0.0 --port 8000 --reload
```

**Linux/Mac:**
```bash
GROQ_API_KEY=gsk_your_api_key_here uvicorn enterprise_rag.api:app --host 0.0.0.0 --port 8000 --reload
```

6. **Open your browser**
   - Navigate to http://localhost:8000
   - You'll see the chat interface!

---

## How to Use

### 1. Upload Your Documents

**Option A: Drag & Drop**
- Drag files or entire folders into the upload area
- Supported formats: `.txt`, `.md`, `.rst`, `.html`, `.py`, `.js`, `.ts`, `.java`

**Option B: Click to Browse**
- Click **"📄 Files"** to select individual files
- Click **"📁 Folder"** to upload an entire folder

**Option C: Use the API**
```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@document.txt" \
  -F "files=@guide.md" \
  -F "permission_tags=public"
```

### 2. Ask Questions

Simply type your question in the chat box and press Enter:

- "What is the main topic of these documents?"
- "How do I configure the authentication system?"
- "Summarize the key points from the uploaded files"
- "What are the pricing tiers mentioned?"

### 3. Get Answers with Citations

The AI will:
- ✅ Search through your documents
- ✅ Find relevant information
- ✅ Generate a clear answer
- ✅ Show citations with source excerpts
- ✅ Display a grounding score (how well the answer is supported)

### 4. Manage Your Documents

**List all uploaded sources:**
```bash
python manage_sources.py list
```

**Delete a specific source:**
```bash
python manage_sources.py delete upload-abc123
```

**Delete all uploaded files:**
```bash
python manage_sources.py delete-uploads
```

**Clear everything:**
```bash
python manage_sources.py clear
```

---

## Advanced Features

### Available AI Models

The system uses Groq API with the following models:

- **llama-3.1-8b-instant** - Ultra-fast responses (1-2 seconds) ⚡ *[Default]*
- **llama-3.1-70b-versatile** - Higher quality, slightly slower
- **mixtral-8x7b-32768** - Large context window for longer documents

You can switch models in the UI's "LLM Settings" panel without restarting the server.

### Connect External Sources

Beyond file uploads, you can connect:

**GitHub Repositories:**
```bash
GITHUB_REPO=owner/repo \
GITHUB_TOKEN=ghp_xxxx \
GROQ_API_KEY=gsk_xxxx \
uvicorn enterprise_rag.api:app --host 0.0.0.0 --port 8000
```

**Jira Projects:**
```bash
JIRA_URL=https://yourorg.atlassian.net \
JIRA_USERNAME=you@company.com \
JIRA_TOKEN=your_api_token \
JIRA_PROJECT_KEY=ENG \
GROQ_API_KEY=gsk_xxxx \
uvicorn enterprise_rag.api:app --host 0.0.0.0 --port 8000
```

### Conversational Memory

The system remembers your conversation:
- Ask follow-up questions
- Reference previous answers
- Build on context from earlier in the chat
- Sessions auto-expire after 60 minutes of inactivity

### Grounding Scores

Every answer includes a grounding score (0-100%) showing how well the answer is supported by your documents:
- **70-100%**: High confidence - well-supported answer
- **40-69%**: Medium confidence - partially supported
- **0-39%**: Low confidence - limited support

---

## Configuration

All settings use environment variables:

### Required Configuration

```bash
# Groq API (required)
GROQ_API_KEY=gsk_your_api_key_here
```

### Optional Configuration

```bash
# LLM Settings
GROQ_MODEL=llama-3.1-8b-instant  # Default: llama-3.1-8b-instant

# Storage
CHROMA_PATH=./chroma_data  # Vector database location
LOG_DIR=logs               # Log file directory

# Chunking Configuration
CHUNK_SIZE=1024      # Chunk size in tokens (default: 1024)
CHUNK_OVERLAP=128    # Overlap between chunks in tokens (default: 128)
```

Larger chunk sizes provide more context to the LLM but use more memory and may reduce retrieval precision. Recommended values:
- Small documents (< 1000 words): 512 tokens, 64 overlap
- Medium documents: 1024 tokens, 128 overlap (default)
- Large documents: 2048 tokens, 256 overlap

### Data Sources

```bash
# GitHub
GITHUB_REPO=owner/repo
GITHUB_TOKEN=ghp_xxxx
GITHUB_PERMISSION_TAGS=engineering,public

# Jira
JIRA_URL=https://yourorg.atlassian.net
JIRA_USERNAME=you@company.com
JIRA_TOKEN=your_api_token
JIRA_PROJECT_KEY=ENG
JIRA_PERMISSION_TAGS=engineering,internal

# Local Docs
DOCS_PATH=/path/to/docs
DOCS_PERMISSION_TAGS=public
```

---

## API Reference

The system provides a REST API for programmatic access:

### Core Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `POST` | `/query` | Ask a question and get an answer |
| `POST` | `/upload` | Upload files or folders |
| `GET` | `/health` | System health check |
| `DELETE` | `/session/{id}` | Clear conversation history |

### Source Management

| Method | Path | Description |
|---|---|---|
| `GET` | `/sources/list` | List all uploaded sources |
| `DELETE` | `/sources/{source_id}` | Delete a specific source |

### Example: Ask a Question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "my-session",
    "query": "What is the refund policy?"
  }'
```

Response:
```json
{
  "answer": "The refund policy allows returns within 30 days [1]...",
  "citations": [
    {
      "number": 1,
      "source_type": "docs",
      "document_title": "refund-policy.pdf",
      "document_url": "uploads/refund-policy.pdf",
      "excerpt": "Customers may return items within 30 days..."
    }
  ],
  "grounding_score": 0.95,
  "correlation_id": "uuid"
}
```

### Example: Upload Files

```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@document.txt" \
  -F "files=@guide.md" \
  -F "permission_tags=public"
```

Response:
```json
{
  "job_id": "uuid",
  "files": [
    {
      "filename": "document.txt",
      "size_bytes": 1024,
      "source_id": "upload-abc123"
    }
  ],
  "total_files": 1,
  "total_bytes": 1024
}
```

### Example: List Sources

```bash
curl http://localhost:8000/sources/list
```

Response:
```json
{
  "sources": [
    {
      "source_id": "upload-abc123",
      "source_type": "docs",
      "count": 5
    }
  ],
  "total_documents": 5
}
```

### Example: Delete a Source

```bash
curl -X DELETE http://localhost:8000/sources/upload-abc123
```

For complete API documentation, visit http://localhost:8000/docs (Swagger UI) when the server is running.

---

## Use Cases

### 📚 Document Q&A
Upload your documentation, manuals, or knowledge base and ask questions:
- "What are the system requirements?"
- "How do I troubleshoot error X?"
- "Summarize the installation process"

### 💼 Business Intelligence
Upload reports, policies, and procedures:
- "What is our vacation policy?"
- "What were the Q3 sales figures?"
- "Summarize the compliance requirements"

### 💻 Code Understanding
Upload your codebase and ask about it:
- "How does the authentication system work?"
- "What does the process_payment function do?"
- "Find all API endpoints related to users"

### 📖 Research Assistant
Upload research papers, articles, or notes:
- "What are the main findings?"
- "Compare the methodologies used"
- "What are the limitations mentioned?"

### 🎓 Study Helper
Upload textbooks, lecture notes, or study materials:
- "Explain the concept of X"
- "What are the key differences between Y and Z?"
- "Create a summary of chapter 5"

---

## How It Works

1. **Upload**: Documents are split into chunks and converted to embeddings (vector representations)
2. **Store**: Embeddings are stored in ChromaDB (a vector database)
3. **Query**: Your question is converted to an embedding
4. **Search**: The system finds the most relevant document chunks using hybrid search (semantic + keyword)
5. **Generate**: Groq API generates an answer using only the retrieved chunks
6. **Cite**: Citations are automatically added showing which documents were used

### Why RAG?

Traditional chatbots can "hallucinate" (make up information). RAG solves this by:
- ✅ Only using your actual documents
- ✅ Providing citations for verification
- ✅ Showing confidence scores
- ✅ Allowing you to trace answers back to sources

---

## Troubleshooting

### "Groq API key not configured" Error

**Solution**: Set your Groq API key:
```bash
# Windows PowerShell
$env:GROQ_API_KEY="gsk_your_api_key_here"

# Linux/Mac
export GROQ_API_KEY=gsk_your_api_key_here
```

Or add it to your `.env` file:
```
GROQ_API_KEY=gsk_your_api_key_here
```

### "Groq API unavailable" Error

**Solutions**:
1. Check your internet connection
2. Verify your API key is valid at https://console.groq.com
3. Check Groq API status at https://status.groq.com

### "Rate limit exceeded" Error

**What it means**: Groq API has rate limits to prevent abuse. The free tier typically allows:
- ~30 requests per minute
- ~14,400 requests per day

**Solutions**:
1. **Wait a few seconds** between questions (recommended: 2-3 seconds)
2. **Upgrade your Groq plan** at https://console.groq.com for higher limits:
   - Pay-as-you-go: Higher rate limits
   - Enterprise: Custom rate limits
3. **Use a different API key** if you have multiple accounts
4. **Batch your questions** instead of asking many rapid-fire queries

**Note**: The system will automatically show a clear error message when you hit the rate limit. Simply wait a moment and try again.

### Slow Responses

Groq API typically responds in 1-2 seconds. If responses are slow:
1. Check your internet connection speed
2. Try switching to a faster model (llama-3.1-8b-instant)
3. Reduce the number of documents uploaded

### Files Not Being Found in Queries

**Solutions**:
1. Wait a few seconds after upload for indexing to complete
2. Check files were uploaded: `python manage_sources.py list`
3. Ask more specific questions mentioning file names or topics
4. Clear your session and try again

### "No valid files uploaded" Error

**Solution**: Check that your files have supported extensions:
- Supported: `.txt`, `.md`, `.rst`, `.html`, `.py`, `.js`, `.ts`, `.java`
- Not supported: `.docx`, `.xlsx`, `.pptx` (convert to `.txt` or `.md` first)

---

## Project Structure

```
enterprise-rag/
├── src/enterprise_rag/
│   ├── api.py                  # FastAPI application & endpoints
│   ├── models.py               # Data models
│   ├── vector_store.py         # ChromaDB integration
│   ├── retriever.py            # Hybrid search engine
│   ├── groq_generator.py       # Groq API wrapper
│   ├── query_rewriter.py       # Query expansion with Groq
│   ├── citation_engine.py      # Citation extraction
│   ├── conversation_manager.py # Session management
│   ├── ingestion/              # Document processing
│   │   ├── pipeline.py
│   │   ├── docs_connector.py
│   │   ├── github_connector.py
│   │   └── jira_connector.py
│   └── static/
│       └── index.html          # Web UI
├── tests/                      # Test suite
├── manage_sources.py           # Source management CLI
├── README.md                   # This file
└── .env.example               # Example environment configuration
```

---

## Technology Stack

| Component | Technology |
|---|---|
| **Backend** | FastAPI + Python 3.11+ |
| **Vector Database** | ChromaDB |
| **AI Models** | Groq API (llama-3.1, mixtral) |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **Search** | Hybrid (semantic + BM25 keyword) |
| **Frontend** | Vanilla JavaScript + HTML/CSS |
| **Testing** | pytest + Hypothesis (property-based) |

---

## Performance

With Groq API:
- **Query Response Time**: 1-2 seconds (end-to-end)
- **Upload Processing**: ~100 documents/second
- **Concurrent Users**: Supports multiple simultaneous queries
- **Context Window**: Up to 32K tokens (mixtral model)

### Rate Limits

Groq API enforces rate limits based on your plan:

**Free Tier**:
- ~30 requests per minute
- ~14,400 requests per day
- Sufficient for testing and small projects

**Paid Tiers**:
- Higher rate limits (varies by plan)
- Better for production use
- See https://console.groq.com/settings/limits for your current limits

**Best Practices**:
- Add a 2-3 second delay between questions for smooth operation
- Monitor your usage at https://console.groq.com
- Consider upgrading if you frequently hit rate limits
- The system automatically handles rate limit errors with clear messages

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## License

MIT

---

## Acknowledgments

- Powered by [Groq](https://groq.com) for ultra-fast LLM inference
- Built with [FastAPI](https://fastapi.tiangolo.com)
- Vector storage by [ChromaDB](https://www.trychroma.com)
- Embeddings by [sentence-transformers](https://www.sbert.net)
