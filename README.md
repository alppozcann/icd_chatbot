### 1️⃣ Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn faiss-cpu sentence-transformers numpy requests
```
```bash
ollama pull deepseek-r1:1.5b
ollama serve
```

to start:
```bash
uvicorn api:app --port 8001
```
