"""
Serviço de processamento e treinamento com documentos.
Armazena tudo no SQLite (documents + knowledge_chunks) para persistência.
"""

import os
import json
from typing import List
from app.database import SessionLocal, Document, KnowledgeChunk, TrainingStatus

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
CHUNKS_DIR = os.path.join(BASE_DIR, "data", "chunks")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)


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
    """Processa documento: extrai texto, cria chunks e salva TUDO no SQLite."""
    db = SessionLocal()
    try:
        # Verificar se já existe
        existing = db.query(Document).filter(Document.filename == filename).first()
        if existing:
            # Remover antigo para reprocessar
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
        db.flush()  # Para obter o ID

        # Salvar CADA chunk no SQLite (tabela knowledge_chunks)
        for i, chunk_text in enumerate(chunks):
            chunk_obj = KnowledgeChunk(
                document_id=document.id,
                chunk_index=i,
                content=chunk_text,
                source=filename,
            )
            db.add(chunk_obj)

        status.status = "done"
        status.message = f"{len(chunks)} chunks salvos no banco"
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
        return {"success": False, "error": str(e)}
    finally:
        db.close()


def get_relevant_context(query: str, max_chunks: int = 8) -> str:
    """Busca chunks relevantes DIRETAMENTE do banco SQLite."""
    db = SessionLocal()
    try:
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
        for score, text, source in scored[:max_chunks]:
            parts.append(f"[Fonte: {source}]\n{text}")

        # Se nenhum match por palavras-chave, pegar os primeiros chunks
        if not parts and all_chunks:
            for chunk in all_chunks[:3]:
                parts.append(f"[Fonte: {chunk.source}]\n{chunk.content}")

        return "\n\n---\n\n".join(parts)
    except Exception:
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
            # Remover upload
            upload_path = os.path.join(UPLOAD_DIR, doc.filename)
            if os.path.exists(upload_path):
                os.remove(upload_path)
            # O cascade deleta os KnowledgeChunks automaticamente
            db.delete(doc)
            db.commit()
            return True
        return False
    except Exception:
        db.rollback()
        return False
    finally:
        db.close()