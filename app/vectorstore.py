"""
Vector Store com SQLite + Ollama Embeddings para RAG.
Armazena chunks de texto e seus embeddings em banco SQLite local.
Usa a API do Ollama (/api/embed) para gerar embeddings vetoriais.
Compatível com Python 3.14+ (sem dependência de sentence-transformers/ChromaDB).
"""

import json
import logging
import os
import sqlite3
import struct
import threading
from pathlib import Path
from typing import Optional

import httpx
import numpy as np

logger = logging.getLogger("vectorstore")

# ── Configuração ──
DB_PATH = Path("./vector_db/vectorstore.db")
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "llama3")

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None

# ── Cache em memória para buscas rápidas ──
_cache_embeddings: Optional[np.ndarray] = None
_cache_rows: Optional[list] = None
_cache_valid = False


def configure(ollama_base: str = None, embed_model: str = None):
    """Permite configurar URL do Ollama e modelo de embedding externamente."""
    global OLLAMA_BASE, EMBED_MODEL
    if ollama_base:
        OLLAMA_BASE = ollama_base
    if embed_model:
        EMBED_MODEL = embed_model


def _get_db() -> sqlite3.Connection:
    """Retorna conexão SQLite (cria DB e tabela se necessário)."""
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                setor TEXT DEFAULT '',
                tipo_documento TEXT DEFAULT '',
                traduzido TEXT DEFAULT 'não',
                text TEXT NOT NULL,
                embedding BLOB
            )
        """)
        _conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)"
        )
        _conn.commit()
    return _conn


def _pack_embedding(emb: list[float]) -> bytes:
    """Empacota embedding como bytes float32 para armazenamento eficiente."""
    return struct.pack(f'{len(emb)}f', *emb)


def _unpack_embedding(data: bytes) -> np.ndarray:
    """Desempacota embedding de bytes float32 para numpy array."""
    n = len(data) // 4
    return np.array(struct.unpack(f'{n}f', data), dtype=np.float32)


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Gera embeddings usando a API do Ollama."""
    if not texts:
        return []

    all_embeddings: list[list[float]] = []
    batch_size = 5

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Tenta API batch /api/embed (Ollama 0.4+)
        try:
            resp = httpx.post(
                f"{OLLAMA_BASE}/api/embed",
                json={"model": EMBED_MODEL, "input": batch},
                timeout=300.0,
            )
            resp.raise_for_status()
            data = resp.json()
            embs = data.get("embeddings", [])
            if embs and len(embs) == len(batch):
                all_embeddings.extend(embs)
                continue
        except Exception as e:
            logger.debug(f"Batch /api/embed falhou, tentando individual: {e}")

        # Fallback: /api/embeddings (um por vez, API antiga)
        for text in batch:
            try:
                resp = httpx.post(
                    f"{OLLAMA_BASE}/api/embeddings",
                    json={"model": EMBED_MODEL, "prompt": text},
                    timeout=120.0,
                )
                resp.raise_for_status()
                emb = resp.json().get("embedding", [])
                if emb:
                    all_embeddings.append(emb)
                else:
                    raise ValueError("Ollama retornou embedding vazio")
            except Exception as e:
                logger.error(f"Falha ao gerar embedding: {e}")
                raise RuntimeError(
                    f"Falha ao gerar embedding via Ollama ({EMBED_MODEL}): {e}"
                )

    return all_embeddings


def _invalidate_cache():
    """Invalida o cache em memória."""
    global _cache_valid, _cache_embeddings, _cache_rows
    _cache_valid = False
    _cache_embeddings = None
    _cache_rows = None


def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Divide texto em chunks com sobreposição."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", " "]:
                idx = text.rfind(sep, start, end)
                if idx > start:
                    end = idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < len(text) else end
    return chunks


def ingest_jsonl(jsonl_path: Path, on_log=None) -> int:
    """Lê o JSONL, gera embeddings via Ollama, armazena no SQLite."""
    if not jsonl_path.exists():
        if on_log:
            on_log(f"JSONL não encontrado: {jsonl_path}")
        return 0

    entries = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if on_log:
        on_log(f"📦 JSONL: {len(entries)} entradas encontradas")

    with _lock:
        db = _get_db()
        existing = {r[0] for r in db.execute("SELECT id FROM chunks").fetchall()}

        new_data = []
        for entry in entries:
            text = entry.get("text", "")
            meta = entry.get("metadata", {})
            source = meta.get("arquivo_original", "desconhecido")

            if not text.strip():
                continue

            chunks = _chunk_text(text)
            for ci, chunk in enumerate(chunks):
                doc_id = f"{source}::chunk_{ci}"
                if doc_id in existing:
                    continue
                new_data.append({
                    "id": doc_id,
                    "source": source,
                    "chunk_index": ci,
                    "setor": meta.get("setor", "CGR_Telecom"),
                    "tipo_documento": meta.get("tipo_documento", ""),
                    "traduzido": meta.get("traduzido", "não"),
                    "text": chunk,
                })

        if not new_data:
            total = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            if on_log:
                on_log(f"✅ Nenhum chunk novo (total existente: {total})")
            return 0

        if on_log:
            on_log(f"🔄 Gerando embeddings para {len(new_data)} chunks via Ollama ({EMBED_MODEL})...")

        # Gerar embeddings em lotes
        texts = [d["text"] for d in new_data]
        try:
            all_embeddings = _embed_texts(texts)
        except RuntimeError as e:
            if on_log:
                on_log(f"❌ Erro ao gerar embeddings: {e}")
            raise

        if on_log and all_embeddings:
            on_log(
                f"📊 Embeddings gerados: {len(all_embeddings)} "
                f"({len(all_embeddings[0])} dimensões)"
            )

        # Inserir no SQLite
        for d, emb in zip(new_data, all_embeddings):
            db.execute(
                "INSERT OR IGNORE INTO chunks "
                "(id, source, chunk_index, setor, tipo_documento, traduzido, text, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    d["id"], d["source"], d["chunk_index"],
                    d["setor"], d["tipo_documento"], d["traduzido"],
                    d["text"], _pack_embedding(emb),
                ),
            )
        db.commit()
        _invalidate_cache()

        total = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if on_log:
            on_log(f"✅ SQLite Vector Store: {len(new_data)} novos chunks (total: {total})")

        return len(new_data)


def query_knowledge(question: str, n_results: int = 5) -> list[dict]:
    """Busca semântica por similaridade de cosseno com embeddings Ollama."""
    global _cache_embeddings, _cache_rows, _cache_valid

    with _lock:
        db = _get_db()

        if not _cache_valid:
            rows = db.execute(
                "SELECT id, source, chunk_index, text, embedding FROM chunks"
            ).fetchall()
            if not rows:
                return []
            _cache_rows = rows
            _cache_embeddings = np.array(
                [_unpack_embedding(r[4]) for r in rows], dtype=np.float32
            )
            _cache_valid = True
        else:
            rows = _cache_rows
            if not rows:
                return []

    if _cache_embeddings is None or len(_cache_embeddings) == 0:
        return []

    # Gerar embedding da pergunta via Ollama
    q_emb = np.array(_embed_texts([question])[0], dtype=np.float32)

    # Similaridade de cosseno vetorizada
    db_emb = _cache_embeddings
    norms_db = np.linalg.norm(db_emb, axis=1)
    norm_q = np.linalg.norm(q_emb)
    similarities = (db_emb @ q_emb) / (norms_db * norm_q + 1e-10)

    n = min(n_results, len(rows))
    top_idx = np.argsort(similarities)[::-1][:n]

    results = []
    for idx in top_idx:
        row = rows[idx]
        results.append({
            "text": row[3],
            "source": row[1],
            "distance": 1.0 - float(similarities[idx]),
        })
    return results


def get_stats() -> dict:
    """Retorna estatísticas do Vector Store SQLite."""
    with _lock:
        db = _get_db()
        total = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        sources = [
            r[0] for r in db.execute("SELECT DISTINCT source FROM chunks").fetchall()
        ]
    return {
        "total_chunks": total,
        "total_documents": len(sources),
        "sources": sorted(sources),
    }


def reset_store():
    """Remove todos os dados do Vector Store (útil para rebuild)."""
    global _conn
    with _lock:
        _invalidate_cache()
        if _conn:
            _conn.close()
            _conn = None
        if DB_PATH.exists():
            DB_PATH.unlink()
        logger.info("Vector Store SQLite resetado.")
