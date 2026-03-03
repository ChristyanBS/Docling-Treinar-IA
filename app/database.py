"""
Banco de dados SQLite — sessões de chat, documentos, chunks de conhecimento.
"""

import os
import uuid
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DATABASE_URL = f"sqlite:///{os.path.join(DATA_DIR, 'knowledge.db')}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, echo=False)
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
    messages = relationship("ChatMessage", back_populates="session",
                            cascade="all, delete-orphan",
                            order_by="ChatMessage.created_at")


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(24), ForeignKey("chat_sessions.id"), nullable=False)
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
    chunks = relationship("KnowledgeChunk", back_populates="document",
                          cascade="all, delete-orphan")


# ── Chunks de Conhecimento (banco permanente para RAG) ───────────
class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, default=0)
    content = Column(Text, nullable=False)
    source = Column(String(255), default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    document = relationship("Document", back_populates="chunks")


# ── Memória de conversas (fatos aprendidos via chat) ─────────────
class ChatMemory(Base):
    __tablename__ = "chat_memories"
    id = Column(Integer, primary_key=True, autoincrement=True)
    fact = Column(Text, nullable=False)
    source_session_id = Column(String(24), default="")
    category = Column(String(50), default="general")
    created_at = Column(DateTime, default=datetime.utcnow)


class TrainingStatus(Base):
    __tablename__ = "training_status"
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    status = Column(String(50), default="pending")
    message = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()