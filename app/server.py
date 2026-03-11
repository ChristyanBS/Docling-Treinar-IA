"""
Servidor FastAPI — Assistente CGR Redes & Telecom v4.0
PostgreSQL + pgvector + Ollama + Docling
"""
import os
import shutil
import logging
from pathlib import Path
from datetime import datetime

import json
import sqlalchemy

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import UPLOAD_DIR, CHAT_MODEL
from app.database import (
    init_db,
    SessionLocal,
    get_db,
    ChatMemory,
    ChatSession,
    KnowledgeChunk,
    Document,
)
from app.ollama_service import check_ollama, chat_stream
from app.training_service import (
    process_and_store_document,
    get_relevant_context,
    get_all_documents,
    get_training_history,
    delete_document,
    backfill_embeddings,
    split_text_into_chunks,
)
from app.memory_service import (
    store_facts,
    store_interaction_memory,
    get_relevant_memories,
    get_past_conversations_context,
    get_recent_memories_summary,
    backfill_memory_embeddings,
    register_user_feedback,
)
from app.embedding_service import embed_texts, embed_single

import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_KB_CONTEXT_CHARS = 5000
MAX_MEMORY_CONTEXT_CHARS = 3000

# ── App ──────────────────────────────────────────────────

app = FastAPI(title="Assistente CGR — Redes & Telecom", version="5.0.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"
IMG_DIR = STATIC_DIR / "img"

# Criar diretório img se não existir
IMG_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/img", StaticFiles(directory=str(IMG_DIR)), name="img")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── System prompt ────────────────────────────────────────

SYSTEM_PROMPT = (
    "Você é o Especialista de Redes do CGR (Centro de Gerência de Redes).\n"
    "Sua missão é apoiar a equipe técnica com precisão, segurança e agilidade.\n"
    "\n"
    "DIRETRIZES DE COMPORTAMENTO (OBRIGATÓRIAS):\n"
    "1. BASE ESTRITA: Suas respostas devem ser baseadas ESTRITAMENTE na 'Base de Conhecimento' (Documentos) e na 'Memória' fornecida.\n"
    "2. PADRÕES DE MERCADO: Utilize padrões técnicos reconhecidos (Cisco, Juniper, MikroTik, Huawei, Datacom, RFCs) quando o documento for genérico.\n"
    "3. SEM ALUCINAÇÕES: Se a informação não estiver no contexto, ADMITA: 'Não encontrei essa informação na minha base atual. Por favor, faça o upload do PDF técnico ou cole o procedimento para que eu possa aprender.'\n"
    "4. SEGURANÇA OPERACIONAL: NUNCA exiba senhas em texto claro, mesmo que estejam nos documentos. Responda: 'A credencial está referenciada no documento X (consulte o cofre de senhas padrão)'.\n"
    "\n"
    "CASOS DE USO ESPECÍFICOS:\n"
    "- Troubleshooting: Se o usuário relatar 'link caiu' ou 'latência', sugira passos de diagnóstico baseados nos manuais de contingência do CGR.\n"
    "- Scripts: Ao gerar configurações (VLAN, OSPF), use a sintaxe exata do vendor solicitado e comente as linhas críticas.\n"
    "- Inventário: Se perguntado 'onde está o servidor X', cite o documento e a data da informação para garantir que não é obsoleta.\n"
    "\n"
    "=== MEMÓRIA PERSISTENTE (PRIORIDADE MÁXIMA) ===\n"
    "Você possui memória que persiste entre conversas.\n"
    "Abaixo você pode receber 'FATOS APRENDIDOS' — são informações que o usuário te ensinou.\n"
    "REGRAS DE MEMÓRIA:\n"
    "1. Fatos aprendidos superam manuais. Ex: Se o manual diz IP 192.168.1.1, mas a memória diz que no CGR usamos 10.0.0.1, responda 10.0.0.1.\n"
    "2. Cite a fonte da memória. Ex: 'Conforme você me ensinou anteriormente...'.\n"
    "REGRAS GERAIS:\n"
    "- Se o usuário perguntar sobre algo que está nos fatos → RESPONDA usando os fatos.\n"
    "- Se perguntar 'quem é X?' e X está nos fatos → responda com base nos fatos, NÃO invente.\n"
    "- Se perguntar sobre lugares, equipes, supervisores → olhe nos fatos PRIMEIRO.\n"
    "- Quando o usuário ensinar algo novo, confirme: 'Anotado! Vou lembrar que [fato].' e repita o fato EXATAMENTE como ele disse.\n"
    "- Priorize informações recentes se houver conflito de datas.\n"
    "\n"
    "=== BASE DE CONHECIMENTO (DOCUMENTOS) ===\n"
    "Você também pode receber conteúdo de documentos que foram enviados (PDFs, textos, etc).\n"
    "REGRAS sobre documentos:\n"
    "- Se receber conteúdo da 'Base de conhecimento técnico', USE esse conteúdo para responder.\n"
    "- PRESTE ATENÇÃO À DATA DO DOCUMENTO (Metadata) se fornecida no contexto.\n"
    "- Cite a fonte: 'Segundo o manual [Nome do Arquivo]...' .\n"
)


# ── Startup ──────────────────────────────────────────────

@app.on_event("startup")
def startup():
    try:
        init_db()
        logger.info("Banco PostgreSQL + pgvector inicializado.")
    except Exception as e:
        logger.error("Falha ao inicializar banco: %s", e)


# ── Pages ────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ── Health ───────────────────────────────────────────────

@app.get("/api/health")
def health():
    ollama = check_ollama()
    db_ok = False
    try:
        db = SessionLocal()
        db.execute(sqlalchemy.text("SELECT 1"))
        db_ok = True
        db.close()
    except Exception:
        pass
    return {"ollama": ollama, "database": db_ok}


# ── Suggestions ──────────────────────────────────────────

@app.get("/api/suggestions")
def suggestions():
    defaults = [
        "Como configurar uma OLT Nokia?",
        "Explicar protocolo OSPF",
        "VLANs e segmentação de rede",
        "Comandos Datacom DM4100",
    ]
    try:
        db = SessionLocal()
        docs = (
            db.query(Document.filename)
            .order_by(Document.created_at.desc())
            .limit(10)
            .all()
        )
        db.close()
        if docs:
            items = []
            for d in docs:
                name = os.path.splitext(d.filename)[0]
                items.append("Resumo de " + name)
            return {"suggestions": items[:6]}
    except Exception:
        pass
    return {"suggestions": defaults}


# ── Chat ─────────────────────────────────────────────────

def _save_assistant_response(session_id: str, user_msg: str, ai_msg: str):
    """Salva resposta do assistente e extrai fatos em background thread."""
    try:
        store_interaction_memory(session_id, "assistant", ai_msg, skip_embedding=True)
        store_facts(session_id, user_msg, ai_msg)
    except Exception as e:
        logger.error("Erro ao salvar resposta em background: %s", e)


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    user_msg = body.get("message", "").strip()
    session_id = body.get("session_id") or "default"

    if not user_msg:
        return JSONResponse({"error": "Mensagem vazia"}, status_code=400)

    # Salvar mensagem do usuário (sem embedding para não bloquear)
    store_interaction_memory(session_id, "user", user_msg, skip_embedding=True)

    # Gerar embedding UMA vez e reutilizar em todas as buscas
    query_emb = embed_single(user_msg)

    # Buscar contexto usando o embedding pré-computado
    memory_context = get_relevant_memories(user_msg, max_results=10, query_embedding=query_emb)
    kb_context = get_relevant_context(user_msg, max_chunks=6, query_embedding=query_emb)
    conv_history = get_past_conversations_context(session_id, limit=8)

    # Se não encontrou memórias relevantes, buscar fatos recentes como fallback
    if not memory_context:
        recent_summary = get_recent_memories_summary(limit=15)
        if recent_summary:
            memory_context = recent_summary

    # Montar mensagens — MEMÓRIA tem prioridade sobre KB
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if memory_context:
        mem_trimmed = memory_context[:MAX_MEMORY_CONTEXT_CHARS]
        messages.append({
            "role": "system",
            "content": "=== FATOS APRENDIDOS (PRIORIDADE MÁXIMA — use estes para responder) ===\n\n" + mem_trimmed,
        })

    if kb_context:
        kb_trimmed = kb_context[:MAX_KB_CONTEXT_CHARS]
        messages.append({
            "role": "system",
            "content": "=== BASE DE CONHECIMENTO (documentos enviados — use para responder) ===\n\n" + kb_trimmed,
        })

    # Adicionar histórico como mensagens separadas (melhor para formato chat)
    if conv_history:
        for msg in conv_history:
            role = msg["role"] if msg["role"] in ("user", "assistant") else "user"
            messages.append({"role": role, "content": msg["content"]})

    messages.append({"role": "user", "content": user_msg})

    total_chars = sum(len(m["content"]) for m in messages)
    logger.info("Chat session=%s msgs=%d total_chars=%d kb=%s mem=%s",
                session_id, len(messages), total_chars,
                "sim" if kb_context else "não",
                "sim" if memory_context else "não")

    # Streaming
    full_response_parts = []

    def generate():
        for token in chat_stream(messages):
            full_response_parts.append(token)
            yield "event: token\ndata: " + json.dumps({"token": token}) + "\n\n"
        # Enviar done ANTES de salvar (não bloquear o cliente)
        yield "event: done\ndata: " + json.dumps({"session_id": session_id}) + "\n\n"
        # Salvar resposta em background thread
        ai_msg = "".join(full_response_parts)
        t = threading.Thread(
            target=_save_assistant_response,
            args=(session_id, user_msg, ai_msg),
            daemon=True,
        )
        t.start()

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Sessions ─────────────────────────────────────────────

@app.get("/api/sessions")
def list_sessions():
    db = SessionLocal()
    try:
        sessions = db.query(ChatSession).order_by(ChatSession.updated_at.desc()).all()
        result = []
        for s in sessions:
            created = s.created_at.isoformat() if s.created_at else None
            updated = s.updated_at.isoformat() if s.updated_at else None
            result.append({
                "session_id": s.session_id,
                "title": s.title,
                "created_at": created,
                "updated_at": updated,
            })
        return result
    finally:
        db.close()


@app.post("/api/sessions")
async def create_session(request: Request):
    body = await request.json()
    session_id = body.get("session_id", "")
    title = body.get("title", "Nova Conversa")

    db = SessionLocal()
    try:
        existing = (
            db.query(ChatSession)
            .filter(ChatSession.session_id == session_id)
            .first()
        )
        if not existing:
            s = ChatSession(session_id=session_id, title=title)
            db.add(s)
            db.commit()
        return {"ok": True}
    except Exception as e:
        db.rollback()
        return {"ok": False, "error": str(e)}
    finally:
        db.close()


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    db = SessionLocal()
    try:
        db.query(ChatMemory).filter(ChatMemory.session_id == session_id).delete()
        db.query(ChatSession).filter(ChatSession.session_id == session_id).delete()
        db.commit()
        return {"ok": True}
    except Exception as e:
        db.rollback()
        return {"ok": False, "error": str(e)}
    finally:
        db.close()


# ── Chat history ─────────────────────────────────────────

@app.get("/api/chat/history")
def chat_history(session_id: str = "default"):
    db = SessionLocal()
    try:
        msgs = (
            db.query(ChatMemory)
            .filter(ChatMemory.session_id == session_id)
            .filter(ChatMemory.memory_type == "conversation")
            .order_by(ChatMemory.created_at.asc())
            .all()
        )
        result = []
        for m in msgs:
            result.append({
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat(),
            })
        return result
    finally:
        db.close()


@app.delete("/api/chat/history")
def clear_history(session_id: str = "default", clear_memories: bool = False):
    db = SessionLocal()
    try:
        if clear_memories:
            db.query(ChatMemory).filter(
                ChatMemory.session_id == session_id
            ).delete()
        else:
            db.query(ChatMemory).filter(
                ChatMemory.session_id == session_id,
                ChatMemory.memory_type == "conversation",
            ).delete()
        db.commit()
        return {"ok": True}
    except Exception as e:
        db.rollback()
        return {"ok": False, "error": str(e)}
    finally:
        db.close()


# ── Document upload ──────────────────────────────────────

@app.post("/api/learn")
async def learn(file: UploadFile = File(None), text: str = Form(None)):
    # Upload de arquivo
    if file and file.filename:
        dest = UPLOAD_DIR / file.filename
        with open(str(dest), "wb") as f:
            shutil.copyfileobj(file.file, f)
        result = process_and_store_document(str(dest), file.filename)
        return result

    # Texto direto
    if text and text.strip():
        chunks = split_text_into_chunks(text, chunk_size=400, overlap=50)
        if not chunks:
            return {"success": False, "error": "Texto muito curto"}

        embeddings = embed_texts(chunks)
        if embeddings is None:
            embeddings = [None] * len(chunks)

        db = SessionLocal()
        try:
            for i in range(len(chunks)):
                kc = KnowledgeChunk(
                    document_id=None,
                    chunk_index=i,
                    content=chunks[i],
                    source="texto_direto",
                    category="manual",
                    embedding=embeddings[i],
                )
                db.add(kc)
            db.commit()
            emb_count = 0
            for e in embeddings:
                if e is not None:
                    emb_count += 1
            return {"success": True, "chunks": len(chunks), "embeddings": emb_count}
        except Exception as e:
            db.rollback()
            return {"success": False, "error": str(e)}
        finally:
            db.close()

    return {"success": False, "error": "Nenhum arquivo ou texto enviado"}


# ── Documents management ─────────────────────────────────

@app.get("/api/documents")
def documents():
    return get_all_documents()


@app.delete("/api/documents/{doc_id}")
def remove_document(doc_id: int):
    ok = delete_document(doc_id)
    return {"ok": ok}


@app.get("/api/training/history")
def training_history():
    return get_training_history()


# ── Backfill embeddings ──────────────────────────────────

@app.post("/api/backfill")
def backfill():
    chunks_result = backfill_embeddings()
    memory_result = backfill_memory_embeddings()
    return {"chunks": chunks_result, "memories": memory_result}


# ── Knowledge stats ──────────────────────────────────────

@app.get("/api/stats")
def stats():
    db = SessionLocal()
    try:
        total_docs = db.query(Document).count()
        total_chunks = db.query(KnowledgeChunk).count()
        chunks_with_emb = (
            db.query(KnowledgeChunk)
            .filter(KnowledgeChunk.embedding.isnot(None))
            .count()
        )
        total_memories = db.query(ChatMemory).count()
        total_facts = (
            db.query(ChatMemory)
            .filter(ChatMemory.memory_type == "learned_fact")
            .count()
        )
        return {
            "documents": total_docs,
            "chunks": total_chunks,
            "chunks_with_embeddings": chunks_with_emb,
            "memories": total_memories,
            "facts": total_facts,
        }
    finally:
        db.close()


# ── Rotas de compatibilidade com frontend ────────────────

@app.get("/api/ollama/status")
def ollama_status():
    """Retorna status do Ollama no formato esperado pelo frontend."""
    result = check_ollama()
    if result.get("online"):
        return {"status": "connected", "models": result.get("models", [])}
    return {"status": "offline", "models": []}


@app.get("/api/knowledge/stats")
def knowledge_stats():
    """Alias de /api/stats no formato esperado pelo frontend."""
    db = SessionLocal()
    try:
        total_docs = db.query(Document).count()
        total_chunks = db.query(KnowledgeChunk).count()
        sources = [
            d.filename
            for d in db.query(Document.filename)
            .order_by(Document.created_at.desc())
            .limit(20)
            .all()
        ]
        return {
            "total_chunks": total_chunks,
            "total_documents": total_docs,
            "sources": sources,
        }
    finally:
        db.close()


@app.get("/api/chat/sessions")
def chat_sessions_list():
    """Lista sessões no formato esperado pelo frontend."""
    db = SessionLocal()
    try:
        sessions = db.query(ChatSession).order_by(ChatSession.updated_at.desc()).all()
        result = []
        for s in sessions:
            last_msg = (
                db.query(ChatMemory.content)
                .filter(
                    ChatMemory.session_id == s.session_id,
                    ChatMemory.memory_type == "conversation",
                    ChatMemory.role == "user",
                )
                .order_by(ChatMemory.created_at.desc())
                .first()
            )
            preview = last_msg[0][:60] if last_msg else s.title or "Nova conversa"
            result.append({"id": s.session_id, "preview": preview})
        return {"sessions": result}
    finally:
        db.close()


@app.post("/api/chat/sessions")
async def chat_sessions_create(request: Request):
    """Cria sessão via /api/chat/sessions."""
    return await create_session(request)


@app.delete("/api/chat/sessions/{session_id}")
def chat_sessions_delete(session_id: str):
    """Deleta sessão via /api/chat/sessions/{id}."""
    return delete_session(session_id)


@app.get("/api/chat/history/{session_id}")
def chat_history_by_path(session_id: str):
    """Retorna histórico de chat no formato esperado pelo frontend."""
    db = SessionLocal()
    try:
        msgs = (
            db.query(ChatMemory)
            .filter(ChatMemory.session_id == session_id)
            .filter(ChatMemory.memory_type == "conversation")
            .order_by(ChatMemory.created_at.asc())
            .all()
        )
        messages = []
        for m in msgs:
            messages.append({"role": m.role, "content": m.content})
        return {"messages": messages}
    finally:
        db.close()


@app.post("/api/memory/save")
async def memory_save(request: Request):
    """Salva fato na memória (ex: nome do usuário)."""
    body = await request.json()
    fact = body.get("fact", "").strip()
    session_id = body.get("session_id", "default")

    if not fact:
        return {"ok": False, "error": "Nenhum fato fornecido"}

    db = SessionLocal()
    try:
        mem = ChatMemory(
            session_id=session_id,
            role="system",
            content=fact,
            fact=fact,
            memory_type="learned_fact",
        )
        emb = embed_texts([fact])
        if emb and emb[0]:
            mem.embedding = emb[0]
        db.add(mem)
        db.commit()
        return {"ok": True}
    except Exception as e:
        db.rollback()
        return {"ok": False, "error": str(e)}
    finally:
        db.close()


@app.post("/api/process")
async def process_pipeline(request: Request):
    """Stub para pipeline de processamento."""
    return {"ok": True, "message": "Pipeline não implementado. Use /api/learn para enviar documentos."}


@app.post("/api/process/cancel")
def cancel_pipeline():
    """Stub para cancelar pipeline."""
    return {"ok": True}


@app.post("/api/chat/feedback")
async def chat_feedback(request: Request):
    """Recebe feedback (like/dislike) de uma mensagem."""
    body = await request.json()
    session_id = body.get("session_id")
    user_msg = body.get("user_msg")
    ai_msg = body.get("ai_msg")
    feedback = body.get("feedback")  # "like" ou "dislike"

    if not all([session_id, user_msg, ai_msg, feedback]):
        return {"ok": False, "error": "Dados incompletos"}

    success = register_user_feedback(session_id, user_msg, ai_msg, feedback)
    return {"ok": success}
