import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.getenv("BHARATDOC_DATA", ROOT / "data"))
UPLOADS = DATA / "uploads"
DB = DATA / "bharatdoc.sqlite3"
MODEL = Path(os.getenv("BHARATDOC_LLM", ROOT / "models" / "qwen-1.5b"))
TRANSLATION = Path(os.getenv("BHARATDOC_TRANSLATION", ROOT / "models" / "nllb"))
VISION = ROOT / "backend" / "bin" / "vision-ocr"
TESSERACT = ROOT / ".tools" / "ocr" / "bin" / "tesseract"
TESSDATA = ROOT / "models" / "tessdata"
OCR_LANGUAGES = {"eng", "hin", "mar", "ben", "tam", "tel", "guj", "kan", "mal", "pan", "ori", "asm", "eng+hin"}
TRANSLATION_LANGUAGES = {"eng_Latn", "hin_Deva", "ben_Beng", "tam_Taml", "tel_Telu", "mar_Deva", "guj_Gujr", "kan_Knda", "mal_Mlym", "pan_Guru", "ory_Orya", "asm_Beng"}
MAX_UPLOAD = 25 * 1024 * 1024
MAX_PAGES = 20
UPLOADS.mkdir(parents=True, exist_ok=True)


@contextmanager
def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with db() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
          id TEXT PRIMARY KEY, name TEXT NOT NULL, mime TEXT NOT NULL,
          filename TEXT NOT NULL, created_at TEXT NOT NULL,
          pages INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL,
          summary TEXT NOT NULL DEFAULT '', fields_json TEXT NOT NULL DEFAULT '{}',
          elapsed_ms INTEGER NOT NULL DEFAULT 0, error TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS pages (
          document_id TEXT NOT NULL, page_number INTEGER NOT NULL,
          text TEXT NOT NULL, method TEXT NOT NULL,
          PRIMARY KEY (document_id, page_number)
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS page_search USING fts5(
          document_id UNINDEXED, page_number UNINDEXED, text
        );
        CREATE TABLE IF NOT EXISTS evaluations (
          id INTEGER PRIMARY KEY AUTOINCREMENT, document_id TEXT NOT NULL,
          created_at TEXT NOT NULL, cer REAL, wer REAL,
          field_accuracy REAL, notes TEXT NOT NULL DEFAULT ''
        );
        """)


init_db()
app = FastAPI(title="BharatDoc AI", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"], allow_methods=["*"], allow_headers=["*"])


def row_dict(row):
    result = dict(row)
    if "fields_json" in result:
        result["fields"] = json.loads(result.pop("fields_json"))
    return result


def extract_fields(text):
    patterns = {
        "dates": r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b",
        "emails": r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
        "phone_numbers": r"(?<!\d)(?:\+91[- ]?)?[6-9]\d{9}(?!\d)",
        "pin_codes": r"(?<!\d)\d{6}(?!\d)",
        "ifsc_codes": r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
        "pan_numbers": r"\b[A-Z]{5}\d{4}[A-Z]\b",
        "gst_numbers": r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]\b",
        "amounts": r"(?:₹|Rs\.?|INR)\s?[\d,]+(?:\.\d{1,2})?",
    }
    return {name: list(dict.fromkeys(re.findall(pattern, text, flags=re.I)))[:30] for name, pattern in patterns.items()}


def run(args, timeout=90):
    return subprocess.run(args, capture_output=True, text=True, check=True, timeout=timeout).stdout.strip()


def ocr_image(path, language="eng"):
    if language not in OCR_LANGUAGES:
        raise ValueError("Unsupported OCR language")
    if TESSERACT.exists():
        command = [str(TESSERACT), str(path), "stdout", "-l", language, "--tessdata-dir", str(TESSDATA)]
        return run(command, timeout=120), f"Tesseract ({language})"
    if language == "eng" and VISION.exists():
        return run([str(VISION), str(path)], timeout=120), "Apple Vision"
    raise RuntimeError("OCR engine missing. Run backend/setup-vision.sh.")


def extract_document(path, mime, language="eng"):
    if mime.startswith("image/"):
        text, method = ocr_image(path, language)
        return [(1, text, method)]
    count = int(run(["pdfinfo", str(path)]).split("Pages:", 1)[1].splitlines()[0].strip())
    if count > MAX_PAGES:
        raise ValueError(f"PDF has {count} pages; maximum is {MAX_PAGES}.")
    pages = []
    with tempfile.TemporaryDirectory(dir=DATA) as temp:
        for n in range(1, count + 1):
            text = run(["pdftotext", "-f", str(n), "-l", str(n), "-layout", str(path), "-"])
            method = "PDF text"
            if len(text.strip()) < 30:
                prefix = Path(temp) / f"page-{n}"
                run(["pdftoppm", "-f", str(n), "-l", str(n), "-scale-to", "2200", "-singlefile", "-png", str(path), str(prefix)])
                text, method = ocr_image(prefix.with_suffix(".png"), language)
            pages.append((n, text, method))
    return pages


@app.get("/api/health")
def health():
    return {"ok": True, "llm_ready": model_ready(MODEL), "translation_ready": model_ready(TRANSLATION), "ocr_ready": TESSERACT.exists() or VISION.exists(), "indic_ocr_ready": TESSERACT.exists() and (TESSDATA / "hin.traineddata").exists()}


def model_ready(path):
    return (path / "config.json").exists() and any((path / name).exists() for name in ("model.safetensors", "pytorch_model.bin", "model-00001-of-00002.safetensors"))


@app.get("/api/documents")
def documents():
    with db() as con:
        return [row_dict(r) for r in con.execute("SELECT * FROM documents ORDER BY created_at DESC")]


@app.post("/api/documents")
async def upload(file: UploadFile = File(...), ocr_language: str = Form("eng")):
    if ocr_language not in OCR_LANGUAGES:
        raise HTTPException(400, "Unsupported OCR language.")
    content = await file.read(MAX_UPLOAD + 1)
    if len(content) > MAX_UPLOAD:
        raise HTTPException(413, "Maximum file size is 25 MB.")
    if not content:
        raise HTTPException(400, "The selected file is empty.")
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        mime, suffix = "image/png", ".png"
    elif content.startswith(b"\xff\xd8\xff"):
        mime, suffix = "image/jpeg", ".jpg"
    elif b"%PDF-" in content[:1024]:
        mime, suffix = "application/pdf", ".pdf"
    else:
        raise HTTPException(400, "This file is not a PDF, PNG, or JPEG. If it is a HEIC or WebP image, export it as JPG or PNG and try again.")
    document_id = str(uuid.uuid4())
    filename = document_id + suffix
    path = UPLOADS / filename
    path.write_bytes(content)
    start = time.monotonic()
    status, error, pages = "ready", "", []
    try:
        pages = extract_document(path, mime, ocr_language)
        if not any(p[1].strip() for p in pages):
            raise ValueError("No text was found in the document.")
    except (ValueError, RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired, IndexError) as exc:
        status, error = "error", str(exc)[:500]
    text = "\n".join(p[1] for p in pages)
    fields = extract_fields(text)
    with db() as con:
        con.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?)", (document_id, Path(file.filename or filename).name, mime, filename, datetime.now(timezone.utc).isoformat(), len(pages), status, "", json.dumps(fields), int((time.monotonic()-start)*1000), error))
        for n, page_text, method in pages:
            con.execute("INSERT INTO pages VALUES (?,?,?,?)", (document_id, n, page_text, method))
            con.execute("INSERT INTO page_search VALUES (?,?,?)", (document_id, n, page_text))
        row = con.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
    return row_dict(row)


@app.get("/api/documents/{document_id}")
def document(document_id: str):
    with db() as con:
        row = con.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Document not found")
        result = row_dict(row)
        result["page_data"] = [dict(p) for p in con.execute("SELECT page_number, text, method FROM pages WHERE document_id=? ORDER BY page_number", (document_id,))]
        return result


class Question(BaseModel):
    question: str


class Translation(BaseModel):
    text: str
    source: str = "eng_Latn"
    target: str = "hin_Deva"


class Evaluation(BaseModel):
    expected_text: str = ""
    expected_fields: dict[str, list[str]] = Field(default_factory=dict)
    notes: str = ""


def llm_generate(prompt, max_tokens=300):
    if not model_ready(MODEL):
        raise HTTPException(503, "Local language model is not installed yet.")
    from .models import generate
    return generate(str(MODEL), prompt, max_tokens)


@app.post("/api/documents/{document_id}/summary")
def summarize(document_id: str):
    doc = document(document_id)
    if doc["status"] != "ready":
        raise HTTPException(400, doc["error"])
    excerpt = "\n".join(f"[Page {p['page_number']}] {p['text']}" for p in doc["page_data"])[:12000]
    exact_dates = ", ".join(doc["fields"].get("dates", [])) or "none"
    answer = llm_generate("Write 3 to 6 short bullet points using only facts explicitly stated below. Include dates, amounts, and actions only if they appear. Copy dates exactly as printed; never convert their order or spell out a month. Do not infer a purpose, significance, transaction, or missing details. Exact dates found: " + exact_dates + "\n\nDOCUMENT:\n" + excerpt, 180)
    with db() as con:
        con.execute("UPDATE documents SET summary=? WHERE id=?", (answer, document_id))
    return {"summary": answer}


def search_pages(document_id, question):
    terms = re.findall(r"[\w]+", question, flags=re.UNICODE)
    query = " OR ".join('"' + t.replace('"', '') + '"' for t in terms[:12])
    with db() as con:
        if query:
            try:
                rows = con.execute("SELECT page_number,text FROM page_search WHERE document_id=? AND page_search MATCH ? ORDER BY bm25(page_search) LIMIT 4", (document_id, query)).fetchall()
            except sqlite3.OperationalError:
                rows = []
        else:
            rows = []
        if not rows:
            rows = con.execute("SELECT page_number,text FROM pages WHERE document_id=? ORDER BY page_number LIMIT 3", (document_id,)).fetchall()
    return [dict(r) for r in rows]


@app.post("/api/documents/{document_id}/chat")
def chat(document_id: str, body: Question):
    document(document_id)
    if not body.question.strip():
        raise HTTPException(400, "Enter a question.")
    pages = search_pages(document_id, body.question)
    context = "\n\n".join(f"[Page {p['page_number']}] {p['text'][:4000]}" for p in pages)
    prompt = f"Answer only from the document excerpts below. Cite each supporting page as [Page N]. If the answer is absent, say you could not find it.\n\n{context}\n\nQuestion: {body.question}"
    answer = llm_generate(prompt, 300)
    if pages and "[Page " not in answer and "could not find" not in answer.lower():
        answer += f" [Page {pages[0]['page_number']}]"
    return {"answer": answer, "sources": [p["page_number"] for p in pages]}


@app.post("/api/translate")
def translate(body: Translation):
    if not body.text.strip():
        raise HTTPException(400, "Enter text to translate.")
    if len(body.text) > 4000:
        raise HTTPException(400, "Translate up to 4,000 characters at once.")
    if body.source not in TRANSLATION_LANGUAGES or body.target not in TRANSLATION_LANGUAGES:
        raise HTTPException(400, "Unsupported language")
    if not model_ready(TRANSLATION):
        raise HTTPException(503, "Local translation model is not installed yet.")
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "backend.app.translate_worker"],
            input=body.model_dump_json(), capture_output=True, text=True,
            cwd=ROOT, env=env, timeout=180, check=True,
        )
        return {"translation": result.stdout.strip()}
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise HTTPException(503, f"Local translation failed: {str(exc)[:250]}") from exc


def edit_distance(a, b):
    previous = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(x != y)))
        previous = current
    return previous[-1]


@app.post("/api/documents/{document_id}/evaluate")
def evaluate(document_id: str, body: Evaluation):
    doc = document(document_id)
    actual = "\n".join(p["text"] for p in doc["page_data"])
    cer = edit_distance(actual, body.expected_text) / max(1, len(body.expected_text)) if body.expected_text else None
    aw, ew = actual.split(), body.expected_text.split()
    wer = edit_distance(aw, ew) / max(1, len(ew)) if body.expected_text else None
    field_scores = []
    for key, expected in body.expected_fields.items():
        actual_values = {str(v).lower().strip() for v in doc["fields"].get(key, [])}
        expected_values = {str(v).lower().strip() for v in expected}
        field_scores.append(actual_values == expected_values)
    accuracy = sum(field_scores)/len(field_scores) if field_scores else None
    with db() as con:
        con.execute("INSERT INTO evaluations (document_id,created_at,cer,wer,field_accuracy,notes) VALUES (?,?,?,?,?,?)", (document_id, datetime.now(timezone.utc).isoformat(), cer, wer, accuracy, body.notes))
    return {"cer": cer, "wer": wer, "field_accuracy": accuracy}


@app.get("/api/evaluations")
def evaluations():
    with db() as con:
        rows = con.execute("SELECT e.*, d.name FROM evaluations e JOIN documents d ON d.id=e.document_id ORDER BY e.id DESC LIMIT 100").fetchall()
        docs = con.execute("SELECT COUNT(*) total, SUM(status='ready') ready, SUM(pages) pages, AVG(elapsed_ms) average_ms FROM documents").fetchone()
    return {"documents": dict(docs), "evaluations": [dict(r) for r in rows]}
