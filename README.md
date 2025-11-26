# RAG Pipeline

A production-ready Retrieval-Augmented Generation (RAG) system with hybrid search capabilities and multiple LLM provider support.

## 🚀 Features

- **Hybrid Search**: Combines vector similarity (FAISS) and BM25 keyword search for optimal retrieval
- **Multi-format Document Support**: PDF, DOCX, TXT, and Markdown files
- **Multiple LLM Providers**: Automatic fallback from Gemini → Ollama (Phi-3) → Mock
- **RESTful API**: FastAPI backend with automatic documentation
- **Web Interface**: Simple HTML/JavaScript frontend for querying
- **Smart Chunking**: Overlapping text chunks for better context preservation

## 📋 Prerequisites

- Python 3.8+
- (Optional) Gemini API key for production use
- (Optional) Ollama with Phi-3 model for local LLM

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag_pipeline
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   # LLM Configuration
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-1.5-flash
   PHI3_LOCAL_URL=http://localhost:11434
   
   # Embedding & Index Configuration
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   INDEX_DIR=data/index
   ```

## 📁 Project Structure

```
rag_pipeline/
├── apps/
│   ├── api/
│   │   └── main.py              # FastAPI application
│   └── web/
│       └── index.html           # Web interface
├── services/
│   ├── ingest.py                # Document parsing and chunking
│   ├── build_index.py           # FAISS index creation
│   ├── retriever.py             # Hybrid search implementation
│   └── adapter.py               # LLM provider management
├── data/
│   ├── raw/                     # Uploaded documents
│   └── index/                   # FAISS index files
├── tests/
│   └── test_pipeline.py         # Unit tests
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚦 Quick Start

### 1. Build Initial Index

If you have existing documents in `data/raw/`, build the index:

```bash
python services/build_index.py data/raw/chunks_preview.json
```

### 2. Start the API Server

```bash
cd apps/api
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### 3. Access the Web Interface

Open `apps/web/index.html` in your browser, or serve it with:

```bash
cd apps/web
python -m http.server 8080
```

Then navigate to `http://localhost:8080`

### 4. API Documentation

Interactive API docs are available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📡 API Endpoints

### POST `/ingest-file`
Upload and index a new document.

**Request:**
```bash
curl -X POST "http://localhost:8000/ingest-file" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "status": "indexed",
  "path": "data/raw/abc123_document.pdf",
  "chunks": 42
}
```

### POST `/ask`
Query the RAG system.

**Request:**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main topics?"}'
```

**Response:**
```json
{
  "query": "What are the main topics?",
  "response": {
    "role": "assistant",
    "content": "Based on the documents..."
  },
  "provider": "gemini",
  "retrieved": [
    {"source": "document.pdf", "score": 0.87}
  ]
}
```

## 🔧 Configuration

### Embedding Model
Change the sentence transformer model in `.env`:
```env
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Fast, good balance
# EMBEDDING_MODEL=all-mpnet-base-v2  # Better quality, slower
```

### Chunking Parameters
Modify in `services/ingest.py`:
```python
chunk_text(text, chunk_words=300, overlap_words=50)
```

### Hybrid Search Balance
Adjust the `alpha` parameter in `retriever.py`:
```python
hybrid_search(query, k=5, alpha=0.6)  # 0.6 = 60% vector, 40% BM25
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run a specific test:
```bash
pytest tests/test_pipeline.py -v
```

## 🔄 LLM Provider Fallback

The system automatically tries providers in order:

1. **Gemini** (Cloud) - Requires API key
2. **Ollama/Phi-3** (Local) - Requires Ollama running
3. **Mock** (Fallback) - Always available for testing

To use Ollama locally:
```bash
# Install Ollama: https://ollama.ai
ollama pull phi3
ollama serve
```

## 📊 How It Works

1. **Document Ingestion**: Files are parsed and split into overlapping chunks
2. **Embedding**: Chunks are converted to vectors using Sentence Transformers
3. **Indexing**: Vectors stored in FAISS for fast similarity search
4. **Retrieval**: Query triggers both vector search (FAISS) and keyword search (BM25)
5. **Ranking**: Results merged using weighted hybrid scoring
6. **Generation**: Top chunks sent to LLM with the query for answer generation

## 🐛 Troubleshooting

**Issue**: "No module named 'services'"
- **Solution**: Run from project root or add to PYTHONPATH

**Issue**: "Gemini API key not set"
- **Solution**: Add `GEMINI_API_KEY` to `.env` or rely on Ollama/Mock fallback

**Issue**: "Index file not found"
- **Solution**: Run `build_index.py` first or upload a file via `/ingest-file`

**Issue**: CORS errors in web interface
- **Solution**: Ensure API server is running and CORS is enabled (default in `main.py`)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **FAISS** - Efficient similarity search by Meta AI
- **Sentence Transformers** - State-of-the-art embeddings
- **FastAPI** - Modern Python web framework
- **Gemini API** - Google's LLM service
- **Ollama** - Local LLM runtime

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation at `/docs` endpoint
- Review test cases for usage examples

---

Built with ❤️ using Python, FAISS, and FastAPI
