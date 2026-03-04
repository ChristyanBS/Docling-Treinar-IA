"""
Serviço de processamento e treinamento com documentos.
Armazena tudo no PostgreSQL (documents + knowledge_chunks) com embeddings pgvector.
"""

import logging
import os
from typing import List

from sqlalchemy import text as sa_text

from app.config import UPLOAD_DIR, CHUNKS_DIR
from app.database import SessionLocal, Document, KnowledgeChunk, TrainingStatus
from app.embedding_service import embed_texts, embed_single

logger = logging.getLogger(__name__)


def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    if not text or not text.strip():
        return []
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def extract_text_from_file(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext in [".pdf", ".docx", ".pptx"]:
            try:
                from docling.document_converter import DocumentConverter
                converter = DocumentConverter()
                result = converter.convert(filepath)
                return result.document.export_to_markdown()
            except ImportError:
                if ext == ".pdf":
                    try:
                        import PyPDF2
                        with open(filepath, "rb") as f:
                            reader = PyPDF2.PdfReader(f)
                            return "".join(p.extract_text() or "" for p in reader.pages)
                    except ImportError:
                        return "[Erro: instale docling ou PyPDF2 para ler PDFs]"
                return f"[Erro: instale docling para ler {ext}]"
        elif ext in [".txt", ".md", ".csv", ".json", ".jsonl", ".py", ".js", ".html", ".xml", ".log"]:
            for enc in ["utf-8", "latin-1", "cp1252"]:
                try:
                    with open(filepath, "r", encoding=enc) as f:
                        return f.read()
                except (UnicodeDecodeError, UnicodeError):
                    continue
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        return f"[Erro: {str(e)}]"
    return "[Erro: não foi possível ler o arquivo]"


def process_and_store_document(filepath: str, filename: str) -> dict:
    """Processa documento: extrai texto, cria chunks com embeddings e salva no PostgreSQL."""
    db = SessionLocal()
    try:
        # Verificar se já existe
        existing = db.query(Document).filter(Document.filename == filename).first()
        if existing:
            db.delete(existing)
            db.commit()

        status = TrainingStatus(filename=filename, status="processing", message="Extraindo texto...")
        db.add(status)
        db.commit()

        text = extract_text_from_file(filepath)
        if not text or text.startswith("[Erro"):
            status.status = "error"
            status.message = text or "Sem texto"
            db.commit()
            return {"success": False, "error": status.message}

        chunks = split_text_into_chunks(text)
        if not chunks:
            status.status = "error"
            status.message = "Nenhum chunk extraído"
            db.commit()
            return {"success": False, "error": status.message}

        ext = os.path.splitext(filename)[1].lower()
        doc_type_map = {
            ".pdf": "PDF", ".docx": "Word", ".pptx": "PowerPoint",
            ".txt": "Texto", ".md": "Markdown", ".csv": "CSV",
            ".json": "JSON", ".jsonl": "JSONL", ".py": "Python", ".html": "HTML"
        }
        doc_type = doc_type_map.get(ext, "Outro")

        # Salvar documento no banco
        document = Document(
            filename=filename, content=text,
            doc_type=doc_type, chunks_count=len(chunks)
        )
        db.add(document)
        db.flush()

        # Gerar embeddings para todos os chunks de uma vez (batch)
        logger.info("Gerando embeddings para %d chunks de %s", len(chunks), filename)
        embeddings = embed_texts(chunks)

        # Salvar CADA chunk no PostgreSQL com embedding
        for i, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
            chunk_obj = KnowledgeChunk(
                document_id=document.id,
                chunk_index=i,
                content=chunk_text,
                source=filename,
                embedding=emb,
            )
            db.add(chunk_obj)

        status.status = "done"
        status.message = f"{len(chunks)} chunks salvos com embeddings"
        db.commit()

        return {
            "success": True,
            "filename": filename,
            "doc_type": doc_type,
            "chunks": len(chunks),
            "text_length": len(text),
        }
    except Exception as e:
        db.rollback()
        logger.error("Erro ao processar %s: %s", filename, e)
        return {"success": False, "error": str(e)}
    finally:
        db.close()


def get_relevant_context(query: str, max_chunks: int = 8) -> str:
    """
    Busca chunks relevantes usando busca vetorial (cosine distance) no pgvector.
    Faz fallback para busca por palavras-chave se não houver embeddings.
    """
    db = SessionLocal()
    try:
        # ── 1) Busca vetorial ────────────────────────────────────
        query_emb = embed_single(query)
        has_embedding = any(v != 0.0 for v in query_emb)

        if has_embedding:
            # Busca por cosine distance (operador <=>) no pgvector
            results = db.execute(
                sa_text("""
                    SELECT content, source, embedding <=> :qvec AS dist
                    FROM knowledge_chunks
                    WHERE embedding IS NOT NULL
                    ORDER BY dist ASC
                    LIMIT :lim
                """),
                {"qvec": str(query_emb), "lim": max_chunks},
            ).fetchall()

            if results:
                parts = []
                for content, source, dist in results:
                    if dist < 0.85:  # threshold de relevância
                        parts.append(f"[Fonte: {source}]\n{content}")
                if parts:
                    return "\n\n---\n\n".join(parts)

        # ── 2) Fallback: busca por palavras-chave ────────────────
        all_chunks = db.query(KnowledgeChunk).all()
        if not all_chunks:
            return ""

        query_words = set(query.lower().split())
        scored = []
        for chunk in all_chunks:
            chunk_words = set(chunk.content.lower().split())
            score = len(query_words.intersection(chunk_words))
            if score > 0:
                scored.append((score, chunk.content, chunk.source))

        scored.sort(key=lambda x: x[0], reverse=True)

        parts = []
        for score, text_content, source in scored[:max_chunks]:
            parts.append(f"[Fonte: {source}]\n{text_content}")

        if not parts and all_chunks:
            for chunk in all_chunks[:3]:
                parts.append(f"[Fonte: {chunk.source}]\n{chunk.content}")

        return "\n\n---\n\n".join(parts)
    except Exception as e:
        logger.error("Erro em get_relevant_context: %s", e)
        return ""
    finally:
        db.close()


def get_all_documents() -> List[dict]:
    db = SessionLocal()
    try:
        docs = db.query(Document).order_by(Document.created_at.desc()).all()
        return [
            {
                "id": d.id, "filename": d.filename,
                "doc_type": d.doc_type, "chunks_count": d.chunks_count,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in docs
        ]
    except Exception:
        return []
    finally:
        db.close()


def get_training_history() -> List[dict]:
    db = SessionLocal()
    try:
        statuses = db.query(TrainingStatus).order_by(TrainingStatus.created_at.desc()).all()
        return [
            {
                "id": s.id, "filename": s.filename,
                "status": s.status, "message": s.message,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in statuses
        ]
    except Exception:
        return []
    finally:
        db.close()


def delete_document(doc_id: int) -> bool:
    """Remove documento e TODOS os seus chunks do banco."""
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc:
            upload_path = os.path.join(UPLOAD_DIR, doc.filename)
            if os.path.exists(upload_path):
                os.remove(upload_path)
            db.delete(doc)
            db.commit()
            return True
        return False
    except Exception:
        db.rollback()
        return False
    finally:
        db.close()


def backfill_embeddings(batch_size: int = 32) -> dict:
    """
    Gera embeddings para chunks que ainda não possuem (embedding IS NULL).
    Útil após migração de dados antigos do SQLite.
    """
    db = SessionLocal()
    try:
        pending = db.query(KnowledgeChunk).filter(
            KnowledgeChunk.embedding.is_(None)
        ).all()

        if not pending:
            return {"updated": 0, "message": "Todos os chunks já possuem embeddings."}

        total = len(pending)
        updated = 0

        for i in range(0, total, batch_size):
            batch = pending[i:i + batch_size]
            texts = [c.content for c in batch]
            embeddings = embed_texts(texts)
            for chunk, emb in zip(batch, embeddings):
                chunk.embedding = emb
                updated += 1
            db.commit()
            logger.info("Backfill: %d/%d chunks atualizados", updated, total)

        return {"updated": updated, "message": f"{updated} chunks atualizados com embeddings."}
    except Exception as e:
        db.rollback()
        logger.error("Erro no backfill: %s", e)
        return {"updated": 0, "error": str(e)}
    finally:
        db.close()