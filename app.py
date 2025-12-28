import os
import uuid
import json
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, session

import config
from utils.db import DB
from utils.openrouter_client import OpenRouterClient
from utils.parsers import parse_txt, parse_docx, parse_pdf
from utils.chunking import chunk_sentences
from utils.chroma_store import ChromaStore

from werkzeug.utils import secure_filename
from dotenv import load_dotenv
load_dotenv()

ALLOWED_EXT = {".txt", ".docx", ".pdf"}

def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-this")
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

    # Ensure data dirs
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    db = DB(config.DB_PATH)
    db.init()

    if not config.OPENROUTER_API_KEY:
        # Don’t crash on import; show a clear message in UI.
        print("WARNING: OPENROUTER_API_KEY is not set. The app will not work for embeddings/chat.")

    orc = OpenRouterClient(
        api_key=config.OPENROUTER_API_KEY,
        http_referer=config.OPENROUTER_HTTP_REFERER,
        x_title=config.OPENROUTER_X_TITLE
    )

    chroma = ChromaStore(config.CHROMA_DIR)

    def allowed_file(filename: str) -> bool:
        return Path(filename).suffix.lower() in ALLOWED_EXT

    def file_type_from_ext(ext: str) -> str:
        return ext.lstrip(".").lower()

    def build_rag_prompt(question: str, contexts: list[dict]) -> list[dict]:
        # contexts: list of {"index":1-based, "file_name", "page_number", "chunk_number", "text"}
        ctx_lines = []
        for c in contexts:
            idx = c["index"]
            meta = f"file={c['file_name']} | page={c['page_number']} | chunk={c['chunk_number']}"
            ctx_lines.append(f"[{idx}] {meta}\n{c['text']}\n")

        ctx_block = "\n".join(ctx_lines).strip()

        system = (
            "You are a retrieval-grounded assistant.\n"
            "Rules:\n"
            "1) Answer ONLY using the provided context. If the answer is not in the context, say you don't know.\n"
            "2) Do NOT invent facts, numbers, diagnoses, citations, or sources.\n"
            "3) When you use a context chunk, cite it inline like [1] or [2].\n"
            "4) If multiple chunks support the same claim, cite multiple.\n"
            "Style:\n"
            "- Be direct. No filler.\n"
        )

        user = (
            f"QUESTION:\n{question}\n\n"
            f"CONTEXT:\n{ctx_block}\n\n"
            "INSTRUCTIONS:\n"
            "- Use ONLY the context.\n"
            "- If context is empty or insufficient, say you don't know.\n"
        )

        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    @app.get("/")
    def home():
        return redirect(url_for("files_page"))

    @app.get("/files")
    def files_page():
        files = db.list_files()
        last_report = session.pop("last_upload_report", None)
        return render_template("files.html", title="Files", files=files, last_report=last_report)

    @app.post("/files/upload")
    def upload_file():
        if not config.OPENROUTER_API_KEY:
            flash("Missing <b>OPENROUTER_API_KEY</b>. Set it in your environment.", "error")
            return redirect(url_for("files_page"))

        f = request.files.get("file")
        file_name = (request.form.get("file_name") or "").strip()

        if not f or f.filename == "":
            flash("No file provided.", "error")
            return redirect(url_for("files_page"))

        if not file_name:
            flash("File name is required.", "error")
            return redirect(url_for("files_page"))

        if not allowed_file(f.filename):
            flash("Unsupported file type. Only TXT, DOCX, PDF.", "error")
            return redirect(url_for("files_page"))

        file_id = str(uuid.uuid4())
        original_filename = f.filename
        ext = Path(original_filename).suffix.lower()
        safe_original = secure_filename(original_filename)
        storage_path = config.UPLOAD_DIR / f"{file_id}_{safe_original}"
        f.save(storage_path)

        size_bytes = storage_path.stat().st_size
        if size_bytes > config.MAX_CONTENT_LENGTH:
            # Flask should block earlier, but keep a hard check.
            storage_path.unlink(missing_ok=True)
            flash("File exceeds 200MB.", "error")
            return redirect(url_for("files_page"))

        # Parse
        try:
            if ext == ".txt":
                sentences = parse_txt(storage_path)
                num_pages = None
            elif ext == ".docx":
                sentences = parse_docx(storage_path)
                num_pages = None
            elif ext == ".pdf":
                sentences, num_pages = parse_pdf(storage_path)
            else:
                raise ValueError("Unsupported extension (should not happen).")
        except Exception as e:
            storage_path.unlink(missing_ok=True)
            flash(f"Failed to parse file: {e}", "error")
            return redirect(url_for("files_page"))
        
        if not sentences:
            storage_path.unlink(missing_ok=True)
            flash("Parsed file is empty or unreadable.", "error")
            return redirect(url_for("files_page"))
        print('Start Chunking')
        # Chunk
        chunks = chunk_sentences(
            sentences,
            file_name=file_name,
            target_tokens=config.TARGET_CHUNK_TOKENS,
            overlap_tokens=config.OVERLAP_TOKENS
        )

        if not chunks:
            storage_path.unlink(missing_ok=True)
            flash("Chunking produced zero chunks.", "error")
            return redirect(url_for("files_page"))
        print('Start Embedding')
        # Embed (batch)
        try:
            inputs = [c.embedded_text for c in chunks]
            embeddings = []
            batch_size = 64  # practical batch size
            print(len(inputs), "number of inputs")
            for i in range(0, len(inputs), batch_size):
                batch = inputs[i:i+batch_size]
                print(i, "embedded")
                embeddings.extend(orc.embeddings(model=config.EMBEDDING_MODEL, inputs=batch))
        except Exception as e:
            storage_path.unlink(missing_ok=True)
            flash(f"Embedding failed: {e}", "error")
            return redirect(url_for("files_page"))

        # Store in Chroma
        payload_chunks = []
        for c in chunks:
            payload_chunks.append({
                "file_name": file_name,
                "chunk_number": c.chunk_number,
                "page_number": c.page_number,
                "embedded_text": c.embedded_text,
            })

        try:
            chroma.add_chunks(
                file_id=file_id,
                file_name=file_name,
                chunks=payload_chunks,
                embeddings=embeddings
            )
        except Exception as e:
            storage_path.unlink(missing_ok=True)
            flash(f"ChromaDB insert failed: {e}", "error")
            return redirect(url_for("files_page"))

        # Store registry
        try:
            db.insert_file(
                file_id=file_id,
                file_name=file_name,
                original_filename=original_filename,
                file_type=file_type_from_ext(ext),
                storage_path=str(storage_path),
                size_bytes=size_bytes,
                num_pages=num_pages,
                num_chunks=len(chunks)
            )
        except Exception as e:
            # Try to rollback chroma chunks
            try:
                chroma.delete_file(file_id=file_id)
            except Exception:
                pass
            storage_path.unlink(missing_ok=True)
            flash(f"Failed to record file in DB: {e}", "error")
            return redirect(url_for("files_page"))

        session["last_upload_report"] = {
            "file_name": file_name,
            "file_type": file_type_from_ext(ext),
            "num_pages": num_pages,
            "num_chunks": len(chunks),
            "storage_path": str(storage_path),
        }
        flash(f"Uploaded & indexed <b>{file_name}</b> with <b>{len(chunks)}</b> chunks.", "ok")
        return redirect(url_for("files_page"))

    @app.post("/files/<file_id>/delete")
    def delete_file(file_id: str):
        row = db.get_file(file_id)
        if not row:
            flash("File not found.", "error")
            return redirect(url_for("files_page"))

        # Delete chroma chunks
        try:
            chroma.delete_file(file_id=file_id)
        except Exception as e:
            flash(f"Failed to delete from ChromaDB: {e}", "error")
            return redirect(url_for("files_page"))

        # Delete stored upload
        try:
            p = Path(row["storage_path"])
            p.unlink(missing_ok=True)
        except Exception:
            pass

        # Delete from registry
        db.delete_file(file_id)
        flash(f"Deleted <b>{row['file_name']}</b> and its indexed chunks.", "ok")
        return redirect(url_for("files_page"))

    @app.get("/ask")
    def ask_page():
        return render_template("ask.html", title="Ask", answer=None, sources=None, question=None)

    @app.post("/ask")
    def ask_submit():
        if not config.OPENROUTER_API_KEY:
            flash("Missing <b>OPENROUTER_API_KEY</b>. Set it in your environment.", "error")
            return redirect(url_for("ask_page"))

        question = (request.form.get("question") or "").strip()
        if not question:
            flash("Question is empty.", "error")
            return redirect(url_for("ask_page"))

        # Embed question for retrieval
        try:
            q_emb = orc.embeddings(model=config.EMBEDDING_MODEL, inputs=[question])[0]
        except Exception as e:
            flash(f"Query embedding failed: {e}", "error")
            return redirect(url_for("ask_page"))

        # Retrieve
        try:
            res = chroma.query(query_embedding=q_emb, n_results=config.TOP_K)
        except Exception as e:
            flash(f"Chroma query failed: {e}", "error")
            return redirect(url_for("ask_page"))

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        sources = []
        contexts_for_prompt = []

        for idx, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
            meta = meta or {}
            sources.append({
                "document": doc,
                "metadata": meta,
                "distance": dist
            })
            contexts_for_prompt.append({
                "index": idx,
                "file_name": meta.get("file_name", "unknown"),
                "page_number": meta.get("page_number", "?"),
                "chunk_number": meta.get("chunk_number", "?"),
                "text": doc
            })

        # Chat
        try:
            messages = build_rag_prompt(question, contexts_for_prompt)
            answer = orc.chat(model=config.CHAT_MODEL, messages=messages, temperature=0.2)
        except Exception as e:
            flash(f"Chat completion failed: {e}", "error")
            return redirect(url_for("ask_page"))

        # Save history (store the same source payload shown in UI, minus distances)
        history_sources = []
        for s in sources:
            history_sources.append({
                "document": s["document"],
                "metadata": s["metadata"],
            })

        db.insert_history(question=question, answer=answer, sources=history_sources)

        return render_template("ask.html", title="Ask", answer=answer, sources=sources, question=question)

    @app.get("/history")
    def history_page():
        items = db.list_history()
        return render_template("history.html", title="History", items=items)

    @app.get("/history/<int:history_id>")
    def history_detail(history_id: int):
        item = db.get_history(history_id)
        if not item:
            flash("History item not found.", "error")
            return redirect(url_for("history_page"))
        return render_template("history_detail.html", title=f"History #{history_id}", item=item)

    @app.post("/history/<int:history_id>/delete")
    def delete_history(history_id: int):
        db.delete_history(history_id)
        flash("History item deleted.", "ok")
        return redirect(url_for("history_page"))

    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5555, debug=True)
