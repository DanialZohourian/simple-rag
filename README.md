# Simple RAG App

An open-source single-user, dark-themed Retrieval-Augmented Generation (RAG) application built with Flask, Tailwind CSS, ChromaDB, and OpenRouter. 

---

## Features

### Files
- Upload **TXT / DOCX / PDF** files (up to **200MB**)
- Editable filename on upload (defaults to original filename)
- Sentence-based chunking:
  - ~2000 tokens per chunk
  - ~200 tokens overlap
  - Long sentences (over 2000 tokens) are split mid-sentence by tokens
- Filename is prefixed to each embedded chunk
- Embeddings generated using:
  - `openai/text-embedding-3-large` via OpenRouter
- Stored in **ChromaDB**
- Metadata stored per chunk:
  - `file_name`
  - `chunk_number`
  - `page_number` (PDF use real pages; TXT/DOCX uses chunk number)
  - `embedded_text`
- Upload report after completion
- View list of uploaded files
- Delete files (removes all related vectors from ChromaDB)

---

### Ask
- Ask questions using RAG
- Top **6** relevant chunks retrieved
- Answer generated using:
  - `openai/gpt-4o` via OpenRouter
- Strong grounding prompt:
  - Model must answer **only** from provided context
  - No hallucination allowed
- Clean UI to inspect retrieved context and metadata

---

### History
- All questions, answers, and used context are saved
- View full past conversations
- Delete individual history entries

---

## Tech Stack

- Backend: **Flask**
- Frontend: **HTML + Tailwind CSS**
- Vector DB: **ChromaDB**
- Embeddings: **OpenRouter**
- LLM: **OpenRouter**
- Tokenization: **tiktoken**
- Parsing:
  - TXT
  - DOCX
  - PDF

---

## Setup

### Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate   # Windows
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Environment variables

```bash
OPENROUTER_API_KEY=your_key_here
OPENROUTER_HTTP_REFERER=http://localhost:5000 # not neccessary
OPENROUTER_X_TITLE=Simple RAG App # not neccessary
```

### Run

```bash
python app.py
```

## License

Apache 2.0
