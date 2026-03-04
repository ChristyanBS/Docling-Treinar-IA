"""
Banco de dados PostgreSQL + pgvector — sessões de chat, documentos,
chunks de conhecimento com embeddings vetoriais.
"""

import logging
import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column, DateTime, ForeignKey, Index, Integer, String, Text,
    create_engine, event, text,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.pool import QueuePool

from app.config import DATABASE_URL, EMBED_DIM

logger = logging.getLogger(__name__)

# ── Engine (PostgreSQL com pool de conexões) ─────────────────────
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def new_id() -> str:
    return uuid.uuid4().hex[:12]


# ── Sessões de Chat ──────────────────────────────────────────────
class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(String(24), primary_key=True, default=new_id)
    preview = Column(String(120), default="Nova conversa")
    created_at = Column(DateTime, default=datetime.utcnow)
    messages = relationship(
        "ChatMessage", back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(24), ForeignKey("chat_sessions.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("ChatSession", back_populates="messages")


# ── Documentos processados ───────────────────────────────────────
class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False, unique=True)
    content = Column(Text, nullable=False)
    doc_type = Column(String(50), default="unknown")
    chunks_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    chunks = relationship(
        "KnowledgeChunk", back_populates="document",
        cascade="all, delete-orphan",
    )


# ── Chunks de Conhecimento (banco permanente para RAG) ───────────
class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    chunk_index = Column(Integer, default=0)
    content = Column(Text, nullable=False)
    source = Column(String(255), default="")
    embedding = Column(Vector(EMBED_DIM), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    document = relationship("Document", back_populates="chunks")


# ── Memória de conversas (fatos aprendidos via chat) ─────────────
class ChatMemory(Base):
    __tablename__ = "chat_memories"
    id = Column(Integer, primary_key=True, autoincrement=True)
    fact = Column(Text, nullable=False)
    source_session_id = Column(String(24), default="")
    category = Column(String(50), default="general")
    embedding = Column(Vector(EMBED_DIM), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class TrainingStatus(Base):
    __tablename__ = "training_status"
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    status = Column(String(50), default="pending")
    message = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)


# ── Índices para busca vetorial (ivfflat) ────────────────────────
# São criados após create_all para que as tabelas já existam.
_VECTOR_INDEXES = [
    Index(
        "ix_knowledge_chunks_embedding",
        KnowledgeChunk.embedding,
        postgresql_using="ivfflat",
        postgresql_with={"lists": 100},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    ),
    Index(
        "ix_chat_memories_embedding",
        ChatMemory.embedding,
        postgresql_using="ivfflat",
        postgresql_with={"lists": 100},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    ),
]


def init_db():
    """Cria a extensão pgvector e todas as tabelas/índices."""
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        Base.metadata.create_all(bind=engine)
        # Criar índices vetoriais (ignora se já existem)
        for idx in _VECTOR_INDEXES:
            try:
                idx.create(bind=engine)
            except Exception:
                pass  # Índice já existe
        logger.info("Banco PostgreSQL + pgvector inicializado com sucesso.")
    except Exception as e:
        logger.warning(
            "Não foi possível conectar ao PostgreSQL: %s. "
            "Verifique se o container Docker está rodando (docker-compose up -d).",
            e,
        )


def get_db():
    """Context manager / generator para obter sessão do banco."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()