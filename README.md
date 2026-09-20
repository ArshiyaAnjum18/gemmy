# Gemmy

Gemmy is a new AI chatbot for Hidden Gems Society.

## Backend foundation

The backend currently provides a minimal FastAPI application with a health endpoint. Ollama configuration is read from `OLLAMA_BASE_URL`, defaulting to `http://localhost:11434`.

Start the server from the project root with:

```powershell
python -m uvicorn backend.main:app --reload
```

Test the health endpoint at <http://127.0.0.1:8000/health>.

RAG components and knowledge-base content will be added only after the backend foundation is confirmed.
