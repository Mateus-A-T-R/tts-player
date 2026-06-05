import asyncio
import json
import re
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from pdf_utils import extract_epub_pages, extract_pages
from tts_engine import (
    KOKORO_AVAILABLE,
    NARRATOR_SILENCE,
    NARRATOR_VOICE,
    VOICES,
    generate_silence,
    get_kokoro_pipeline,
    split_sentences,
    synthesize_edge,
    synthesize_edge_narrator,
    synthesize_kokoro,
)

app = FastAPI(title="TTS Player")

import os
BOOKS_DIR = Path(os.environ.get("BOOKS_DIR", Path(__file__).parent / "books"))
BOOKS_DIR.mkdir(parents=True, exist_ok=True)


def _make_book_id(filename: str) -> str:
    stem = Path(filename).stem
    slug = re.sub(r"[^\w\-]", "_", stem).lower()
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "livro"


@app.get("/")
async def index():
    html = Path(__file__).parent / "index.html"
    return HTMLResponse(html.read_text(encoding="utf-8"))


@app.get("/api/voices")
async def voices():
    if not KOKORO_AVAILABLE:
        return {lang: {"edge": v["edge"]} for lang, v in VOICES.items()}
    return VOICES


# ── Book library ──────────────────────────────────────────────────────────────

class ProgressUpdate(BaseModel):
    page: int


@app.get("/api/books")
async def list_books():
    books = []
    for f in sorted(BOOKS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            books.append({
                "id":        data["id"],
                "filename":  data["filename"],
                "total":     data["total"],
                "last_page": data.get("last_page", 0),
            })
        except Exception:
            pass
    return books


@app.get("/api/books/{book_id}")
async def get_book(book_id: str):
    target = BOOKS_DIR / f"{book_id}.json"
    if not target.exists():
        raise HTTPException(status_code=404, detail="Livro não encontrado")
    return json.loads(target.read_text(encoding="utf-8"))


@app.patch("/api/books/{book_id}/progress")
async def save_progress(book_id: str, body: ProgressUpdate):
    target = BOOKS_DIR / f"{book_id}.json"
    if not target.exists():
        raise HTTPException(status_code=404, detail="Livro não encontrado")

    def _write():
        data = json.loads(target.read_text(encoding="utf-8"))
        data["last_page"] = body.page
        target.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    await asyncio.to_thread(_write)
    return {"ok": True}


@app.delete("/api/books/{book_id}")
async def delete_book(book_id: str):
    target = BOOKS_DIR / f"{book_id}.json"
    if not target.exists():
        raise HTTPException(status_code=404, detail="Livro não encontrado")
    target.unlink()
    return {"ok": True}


@app.post("/api/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    filename  = file.filename or "livro"
    if filename.lower().endswith(".epub"):
        pages = await asyncio.to_thread(extract_epub_pages, contents)
    else:
        pages = await asyncio.to_thread(extract_pages, contents)
    book_id   = _make_book_id(filename)
    book_data = {
        "id": book_id,
        "filename": file.filename,
        "total": len(pages),
        "pages": pages,
    }
    target = BOOKS_DIR / f"{book_id}.json"
    target.write_text(json.dumps(book_data, ensure_ascii=False), encoding="utf-8")
    return {"id": book_id, "filename": file.filename, "total": len(pages)}


# ── TTS WebSocket ─────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def tts_stream(ws: WebSocket):
    await ws.accept()

    stop_event = asyncio.Event()

    async def _listen_for_stop():
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                if msg.get("type") == "stop":
                    stop_event.set()
                    return
        except Exception:
            stop_event.set()

    try:
        data = await ws.receive_json()
    except Exception:
        return

    text: str      = data.get("text", "").strip()
    voice: str     = data.get("voice", "pt-BR-AntonioNeural")
    speed: float   = float(data.get("speed", 1.0))
    language: str  = data.get("language", "pt-BR")
    engine: str    = data.get("engine", "edge")
    pitch: str     = data.get("pitch", "+0Hz")
    narrator: bool = bool(data.get("narrator", False))

    if narrator and engine == "edge":
        if not voice or voice == data.get("voice", ""):
            voice = NARRATOR_VOICE.get(language, voice)

    if not text:
        await ws.send_json({"type": "error", "message": "Texto vazio."})
        return

    sentences = split_sentences(text)
    kokoro_pipeline = get_kokoro_pipeline(language) if engine == "kokoro" else None
    recv_task = asyncio.create_task(_listen_for_stop())

    try:
        for i, sentence in enumerate(sentences):
            if stop_event.is_set():
                break

            await ws.send_json({
                "type": "sentence_start",
                "index": i,
                "total": len(sentences),
                "words": sentence.split(),
            })

            try:
                if engine == "edge" and narrator:
                    audio_bytes = await synthesize_edge_narrator(sentence, voice)
                elif engine == "edge":
                    audio_bytes = await synthesize_edge(sentence, voice, speed, pitch)
                else:
                    audio_bytes = await asyncio.to_thread(
                        synthesize_kokoro, kokoro_pipeline, sentence, voice, speed
                    )
            except Exception as exc:
                await ws.send_json({"type": "error", "message": str(exc)})
                break

            if stop_event.is_set():
                break

            if audio_bytes:
                await ws.send_bytes(audio_bytes)

            if narrator and i < len(sentences) - 1 and not stop_event.is_set():
                await ws.send_bytes(generate_silence(NARRATOR_SILENCE))

        if not stop_event.is_set():
            await ws.send_json({"type": "done"})

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        recv_task.cancel()
        try:
            await ws.close()
        except Exception:
            pass
