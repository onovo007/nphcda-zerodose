"""
Lightweight retrieval-augmented generation over an uploaded programme document (PDF or Word).

Pipeline: extract text with page numbers -> chunk -> embed chunks (OpenAI, user's key) ->
cosine retrieval -> grounded answer with page citations and a retrieval-confidence score.
No vector database needed; the index is a small numpy matrix held in session state.
"""
from __future__ import annotations

import io

import numpy as np

import llm


# --------------------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------------------
def extract(filename: str, raw: bytes) -> list[tuple[int, str]]:
    """Return [(page_number, page_text), ...]. For Word, blocks are numbered as pseudo-pages."""
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        pages = []
        for i, pg in enumerate(reader.pages):
            try:
                txt = pg.extract_text() or ""
            except Exception:
                txt = ""
            if txt.strip():
                pages.append((i + 1, txt))
        return pages
    if name.endswith((".docx", ".doc")):
        import docx
        doc = docx.Document(io.BytesIO(raw))
        paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        # Group ~12 paragraphs per pseudo-page so citations stay meaningful.
        pages, block, pageno = [], [], 1
        for p in paras:
            block.append(p)
            if len(block) >= 12:
                pages.append((pageno, "\n".join(block)))
                block, pageno = [], pageno + 1
        if block:
            pages.append((pageno, "\n".join(block)))
        return pages
    raise ValueError("Unsupported file type. Upload a PDF or Word (.docx) document.")


def chunk(pages: list[tuple[int, str]], target_words: int = 220) -> list[dict]:
    """Split each page into ~target_words chunks, keeping the page number on every chunk."""
    chunks = []
    cid = 0
    for page, text in pages:
        words = text.split()
        for i in range(0, len(words), target_words):
            piece = " ".join(words[i:i + target_words]).strip()
            if len(piece) > 40:
                chunks.append({"id": cid, "page": page, "text": piece})
                cid += 1
    return chunks


# --------------------------------------------------------------------------------------
# Index + retrieval
# --------------------------------------------------------------------------------------
def build_index(api_key: str, chunks: list[dict]):
    """Embed chunk texts. Returns (matrix, error). matrix is L2-normalized float32."""
    vecs = llm.embed(api_key, [c["text"] for c in chunks])
    if isinstance(vecs, str):  # error string
        return None, vecs
    m = np.asarray(vecs, dtype="float32")
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms, None


def retrieve(api_key: str, query: str, matrix, chunks: list[dict], k: int = 5):
    """Return (top_chunks_with_scores, confidence_pct, error)."""
    qv = llm.embed(api_key, [query])
    if isinstance(qv, str):
        return [], 0.0, qv
    q = np.asarray(qv[0], dtype="float32")
    q = q / (np.linalg.norm(q) or 1.0)
    sims = matrix @ q
    order = np.argsort(-sims)[:k]
    top = []
    for idx in order:
        c = dict(chunks[int(idx)])
        c["score"] = float(sims[int(idx)])
        top.append(c)
    # Confidence: top cosine similarity mapped to a 0-100 scale (clipped).
    confidence = round(float(np.clip(sims[order[0]], 0, 1) * 100), 1) if len(order) else 0.0
    return top, confidence, None


def answer(api_key: str, model: str, question: str, matrix, chunks: list[dict], k: int = 5,
           language: str | None = None):
    """Full RAG step. Returns dict {text, citations, confidence, snippets, error}."""
    top, confidence, err = retrieve(api_key, question, matrix, chunks, k)
    if err:
        return {"error": err}
    txt = llm.rag_answer(api_key, model, question, top, language=language)
    pages = sorted({c["page"] for c in top})
    return {"text": txt, "citations": pages, "confidence": confidence,
            "snippets": top, "error": None}
