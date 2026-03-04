"""
Serviço de processamento e treinamento com documentos.
Armazena tudo no PostgreSQL (documents + knowledge_chunks) com embeddings pgvector.
"""

import os
import json
import logging
from typing import List
from datetime import datetime

from sqlalchemy import text as sa_text

from app.config import UPLOAD_DIR, CHUNKS_DIR
from app.database import SessionLocal, Document, KnowledgeChunk, TrainingStatus
from app.embedding_service import embed_texts, embed_single

logger = logging.getLogger(__name__)


def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Divide texto em chunks com overlap."""
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
        start += chunk_size - overlap
    return chunks


def extract_text_from_file(filepath: str) -> str:
    """Extrai texto de diversos formatos de arquivo."""
    ext = os.path.splitext(filepath)[1].lower()
    try:
        # Docling (principal)
        if ext in (".pdf", ".docx", ".doc", ".pptx", ".html", ".md"):
            try:
                from docling.document_converter import DocumentConverter
                converter = DocumentConverter()
                result = converter.convert(filepath)
                text = result.document.export_to_markdown()
                if text and text.strip():
                    return text
            except Exception as e:
                logger.warning(f"Docling falhou para {filepath}: {e}")

        # Fallback PDF
        if ext == ".pdf":
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(filepath)
                pages = [p.extract_text() or "" for p in reader.pages]
                text = "\n".join(pages)
                if text.strip():
                    return text
            except Exception as e:
                logger.warning(f"PyPDF2 falhou: {e}")

        # TXT, CSV, MD, JSON
        if ext in (".txt", ".csv", ".md", ".log"):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        if ext == ".json":
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return json.dumps(data, ensure_ascii=False, indent=2)

        # XLSX
        if ext == ".xlsx":
            try:
                import openpyxl
                wb = openpyxl.load_workbook(filepath, read_only=True)
                rows = []
                for ws in wb.worksheets:
                    for row in ws.iter_rows(values_only=True):
                        vals = [str(c) for c in row if c is not None]
                        if vals:
                            rows.append(" | ".join(vals))
                return "\n".join(rows)
            except Exception as e:
                logger.warning(f"openpyxl falhou: {e}")

        # Imagens (OCR)
        if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"):
            try:
                from PIL import Image
                import pytesseract
                img = Image.open(filepath)
                text = pytesseract.image_to_string(img, lang="por+eng")
                if text and text.strip():
                    return text
            except Exception as e:
                logger.warning(f"OCR falhou: {e}")

        # DOCX fallback
        if ext == ".docx":
            try:
                import docx
                doc = docx.Document(filepath)
                return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            except Exception as e:
                logger.warning(f"python-docx falhou: {e}")

    except Exception as e:
        logger.error(f"Erro ao extrair texto de {filepath}: {e}")

    return "[Erro: não foi possível ler o arquivo]"


def process_and_store_document(filepath: str, filename: str) -> dict:
    """Processa documento: extrai texto, cria chunks com embeddings e salva no PostgreSQL."""
    db = SessionLocal()
    try:
        ext = os.path.splitext(filename)[1].lower()
        file_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0

        # Criar registro do documento
        doc = Document(
            filename=filename,
            filepath=filepath,
            file_type=ext,
            file_size=file_size,
            status="processing",
        )
        db.add(doc)
        db.flush()

        # Extrair texto
        text = extract_text_from_file(filepath)
        if not text or text.startswith("[Erro"):
            doc.status = "error"
            doc.error_message = "Não foi possível extrair texto"
            db.commit()
            return {"success": False, "error": doc.error_message}

        # Criar chunks
        chunks = split_text_into_chunks(text, chunk_size=500, overlap=50)
        if not chunks:
            doc.status = "error"
            doc.error_message = "Nenhum chunk gerado"
            db.commit()
            return {"success": False, "error": doc.error_message}

        # Gerar embeddings em batch
        embeddings = embed_texts(chunks)
        if embeddings is None:
            logger.warning(f"Embeddings falharam para {filename}, salvando sem vetor")
            embeddings = [None] * len(chunks)

        # Salvar chunks
        for i, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
            kc = KnowledgeChunk(
                document_id=doc.id,
                chunk_index=i,
                content=chunk_text,
                source=filename,
                category=ext.replace(".", ""),
                embedding=emb,
            )
            db.add(kc)

        doc.num_chunks = len(chunks)
        doc.status = "completed"

        # Status de treinamento
        ts = TrainingStatus(
            document_id=doc.id,
            status="completed",
            message=f"{len(chunks)} chunks processados com embeddings",
        )
        db.add(ts)
        db.commit()

        emb_count = sum(1 for e in embeddings if e is not None)
        logger.info(f"Documento '{filename}': {len(chunks)} chunks, {emb_count} embeddings")

        return {
            "success": True,
            "document_id": doc.id,
            "chunks": len(chunks),
            "embeddings": emb_count,
            "filename": filename,
        }

    except Exception as e:
        db.rollback()
        logger.error(f"Erro ao processar documento {filename}: {e}")
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
        # Tentar busca vetorial
        query_emb = embed_single(query)
        if query_emb is not None:
            results = (
                db.query(KnowledgeChunk)
                .filter(KnowledgeChunk.embedding.isnot(None))
                .order_by(KnowledgeChunk.embedding.cosine_distance(query_emb))
                .limit(max_chunks)
                .all()
            )
            if results:
                parts = []
                for r in results:
                    src = r.source or "desconhecido"
                    parts.append(f"[Fonte: {src}]\n{r.content}")
                return "\n\n---\n\n".join(parts)

        # Fallback: busca por palavras-chave
        logger.info("Busca vetorial indisponível, usando fallback por palavras-chave")
        keywords = [w.lower() for w in query.split() if len(w) > 3]
        if not keywords:
            return ""

        all_chunks = db.query(KnowledgeChunk).all()
        scored = []
        for chunk in all_chunks:
            content_lower = chunk.content.lower()
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:max_chunks]

        if not top:
            return ""

        parts = []
        for _, chunk in top:
            src = chunk.source or "desconhecido"
            parts.append(f"[Fonte: {src}]\n{chunk.content}")
        return "\n\n---\n\n".join(parts)

    except Exception as e:
        logger.error(f"Erro na busca de contexto: {e}")
        return ""
    finally:
        db.close()


def get_all_documents() -> List[dict]:
    """Retorna todos os documentos."""
    db = SessionLocal()
    try:
        docs = db.query(Document).order_by(Document.created_at.desc()).all()
        return [
            {
                "id": d.id,
                "uid": d.uid,
                "filename": d.filename,
                "file_type": d.file_type,
                "file_size": d.file_size,
                "num_chunks": d.num_chunks,
                "status": d.status,
                "error_message": d.error_message,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in docs
        ]
    finally:
        db.close()


def get_training_history() -> List[dict]:
    """Retorna histórico de treinamento."""
    db = SessionLocal()
    try:
        statuses = (
            db.query(TrainingStatus)
            .order_by(TrainingStatus.created_at.desc())
            .limit(100)
            .all()
        )
        return [
            {
                "id": s.id,
                "document_id": s.document_id,
                "status": s.status,
                "message": s.message,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in statuses
        ]
    finally:
        db.close()


def delete_document(doc_id: int) -> bool:
    """Deleta documento e seus chunks."""
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            return False

        db.query(KnowledgeChunk).filter(KnowledgeChunk.document_id == doc_id).delete()
        db.query(TrainingStatus).filter(TrainingStatus.document_id == doc_id).delete()
        db.delete(doc)
        db.commit()

        # Remover arquivo físico
        if doc.filepath and os.path.exists(doc.filepath):
            try:
                os.remove(doc.filepath)
            except Exception:
                pass

        return True
    except Exception as e:
        db.rollback()
        logger.error(f"Erro ao deletar documento {doc_id}: {e}")
        return False
    finally:
        db.close()


def backfill_embeddings(batch_size: int = 32) -> dict:
    """Gera embeddings para chunks que ainda não têm."""
    db = SessionLocal()
    try:
        chunks = (
            db.query(KnowledgeChunk)
            .filter(KnowledgeChunk.embedding.is_(None))
            .limit(batch_size)
            .all()
        )
        if not chunks:
            return {"updated": 0, "message": "Nenhum chunk sem embedding"}

        texts = [c.content for c in chunks]
        embeddings = embed_texts(texts)

        if embeddings is None:
            return {"updated": 0, "error": "Não foi possível gerar embeddings"}

        updated = 0
        for chunk, emb in zip(chunks, embeddings):
            if emb is not None:
                chunk.embedding = emb
                updated += 1

        db.commit()
        remaining = (
            db.query(KnowledgeChunk)
            .filter(KnowledgeChunk.embedding.is_(None))
            .count()
        )
        return {"updated": updated, "remaining": remaining}
    except Exception as e:
        db.rollback()
        logger.error(f"Erro no backfill: {e}")
        return {"updated": 0, "error": str(e)}
    finally:
        db.close()