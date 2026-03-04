"""
Banco de dados PostgreSQL + pgvector.
"""
import uuid
import logging
from datetime import datetime
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, Float, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector

from app.config import DATABASE_URL, EMBED_DIM

logger = logging.getLogger(__name__)

engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


def new_id() -> str:
    return uuid.uuid4().hex[:16]


# ── Modelos ──────────────────────────────────────────────

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, autoincrement=True)
    uid = Column(String(32), default=new_id, unique=True, index=True)
    filename = Column(String(512), nullable=False)
    filepath = Column(String(1024))
    file_type = Column(String(32))
    file_size = Column(Integer, default=0)
    num_chunks = Column(Integer, default=0)
    status = Column(String(32), default="pending")
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, index=True)
    chunk_index = Column(Integer, default=0)
    content = Column(Text, nullable=False)
    source = Column(String(512))
    category = Column(String(128))
    embedding = Column(Vector(EMBED_DIM))
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_kc_source", "source"),
        Index("ix_kc_category", "category"),
    )


class ChatMemory(Base):
    __tablename__ = "chat_memories"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), index=True)
    role = Column(String(16))
    content = Column(Text, nullable=False)
    fact = Column(Text)
    memory_type = Column(String(32), default="conversation")
    relevance_score = Column(Float, default=0.5)
    embedding = Column(Vector(EMBED_DIM))
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_cm_session", "session_id"),
        Index("ix_cm_created", "created_at"),
    )


class TrainingStatus(Base):
    __tablename__ = "training_status"
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, index=True)
    status = Column(String(32), default="pending")
    message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), unique=True, index=True)
    title = Column(String(256), default="Nova Conversa")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ── Context manager ──────────────────────────────────────

@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ── Init ─────────────────────────────────────────────────

def init_db():
    """Cria extensão pgvector e todas as tabelas."""
    try:
        with engine.connect() as conn:
            conn.execute(__import__("sqlalchemy").text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        Base.metadata.create_all(bind=engine)
        logger.info("Banco PostgreSQL + pgvector inicializado com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao inicializar banco: {e}")
        raise