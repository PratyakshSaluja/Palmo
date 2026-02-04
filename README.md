# University RAG Chatbot API

A production-ready RAG (Retrieval-Augmented Generation) chatbot API for university applications.

## Features

- 📄 **Document Management**: Upload, list, and delete PDF, DOCX, and TXT files
- 🔍 **Semantic Search**: ChromaDB-powered vector similarity search
- 🤖 **AI-Powered Q&A**: Groq LLM with context-aware responses
- 📡 **REST API**: FastAPI with automatic OpenAPI documentation
- 🔄 **Streaming Responses**: Server-Sent Events for real-time answers

## Quick Start

### 1. Install Dependencies

```bash
cd university_rag_chatbot
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
copy .env.example .env
# Edit .env with your GROQ_API_KEY
```

### 3. Run the Server

```bash
uvicorn app.main:app --reload
```

### 4. Access the API

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/documents/upload` | Upload documents |
| GET | `/api/v1/documents` | List all documents |
| GET | `/api/v1/documents/{id}` | Get document details |
| DELETE | `/api/v1/documents/{id}` | Delete a document |
| POST | `/api/v1/query` | Query the chatbot |
| POST | `/api/v1/query/stream` | Stream query response |
| GET | `/api/v1/stats` | Get system statistics |

## Project Structure

```
university_rag_chatbot/
├── app/
│   ├── api/routers/      # API endpoints
│   ├── core/             # Exceptions, logging
│   ├── models/           # Pydantic schemas
│   ├── services/         # Business logic
│   └── utils/            # Document parsing, text splitting
├── data/                 # ChromaDB & uploads
└── tests/                # Test suite
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key (required) | - |
| `LLM_MODEL_NAME` | Groq model name | `llama3-70b-8192` |
| `EMBEDDING_MODEL_NAME` | HuggingFace model | `all-MiniLM-L6-v2` |
| `CHROMA_PERSIST_DIR` | ChromaDB storage path | `./data/chroma_db` |
| `CHUNK_SIZE` | Text chunk size | `1000` |
| `CHUNK_OVERLAP` | Chunk overlap | `200` |

## License

MIT
