"""
Servidor FastAPI — implementa TODAS as rotas que o index.html espera,
com SSE streaming para chat e learn, sessões de chat, CORS para Live Server.
"""

import asyncio
import json
import os
import shutil
import traceback
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.database import (
    Base, ChatMemory, ChatMessage, ChatSession, Document, KnowledgeChunk,
    SessionLocal, TrainingStatus, init_db, new_id,
)
from app.ollama_service import check_ollama_status, chat_stream, list_models
from app.training_service import (
    CHUNKS_DIR, UPLOAD_DIR,
    delete_document,
    extract_text_from_file,
    get_all_documents,
    get_relevant_context,
    get_training_history,
    process_and_store_document,
    split_text_into_chunks,
)
from app.memory_service import (
    extract_facts_from_exchange,
    get_past_conversations_context,
    get_recent_memories_summary,
    get_relevant_memories,
    store_facts,
)

# ── Inicialização ────────────────────────────────────────────────
init_db()
app = FastAPI(title="Docling IA Trainer", version="3.1.0")

# CORS — permite Live Server (5500) e qualquer origem em dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Estado do pipeline (simplificado) ────────────────────────────
_pipeline_running = False
_pipeline_cancel = False
_pipeline_logs: List[str] = []


# ── Helper SSE ───────────────────────────────────────────────────
def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse_response(generator):
    """StreamingResponse com headers corretos para SSE via CORS."""
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ══════════════════════════════════════════════════════════════════
#  SERVE INDEX.HTML
# ══════════════════════════════════════════════════════════════════
@app.get("/")
async def serve_index():
    idx = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(idx):
        return FileResponse(idx)
    return JSONResponse({"message": "Servidor rodando", "docs": "/docs"})


# ══════════════════════════════════════════════════════════════════
#  OLLAMA STATUS
# ══════════════════════════════════════════════════════════════════
@app.get("/api/ollama/status")
async def ollama_status():
    return await check_ollama_status()


@app.get("/api/ollama/models")
async def ollama_models():
    return {"models": await list_models()}


# ══════════════════════════════════════════════════════════════════
#  SUGGESTIONS
# ══════════════════════════════════════════════════════════════════
@app.get("/api/suggestions")
async def suggestions():
    db = SessionLocal()
    try:
        docs = db.query(Document).all()
        names = [d.filename for d in docs]
    except Exception:
        names = []
    finally:
        db.close()

    base = ["Explicar protocolo OSPF", "O que é RIP?", "Resumo Nokia OLT"]
    for n in names[:3]:
        base.append(f"Resumo de {n}")
    return {"suggestions": base[:6]}


# ══════════════════════════════════════════════════════════════════
#  CHAT — SSE streaming (token-a-token)
# ══════════════════════════════════════════════════════════════════
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Chat com SSE streaming, memória de conversas e aprendizado via chat."""

    # Preparar dados ANTES do streaming
    try:
        db = SessionLocal()

        # Obter ou criar sessão
        session_id = req.session_id
        sess = None
        if session_id:
            sess = db.query(ChatSession).filter(ChatSession.id == session_id).first()

        if not sess:
            session_id = new_id()
            sess = ChatSession(id=session_id, preview=req.message[:80])
            db.add(sess)
            db.commit()
            
        db.close()
    except Exception as e:
        try:
            db.close()
        except Exception:
            pass
        async def error_gen():
            yield _sse("error", {"error": f"Erro ao preparar chat: {str(e)}"})
        return _sse_response(error_gen())

    # ── 1) Contexto de documentos treinados (RAG) ────────────────
    doc_context = get_relevant_context(req.message)

    # ── 2) Memórias de conversas anteriores (fatos aprendidos) ───
    memories = get_relevant_memories(req.message, max_results=8)
    # Também incluir memórias recentes gerais (pode não bater por keyword)
    recent_mem = get_recent_memories_summary(limit=10)
    all_memories_text = ""
    if memories:
        all_memories_text = "\n".join(f"- {m}" for m in memories)
    # Adicionar memórias recentes que não estão nas relevantes
    if recent_mem:
        for line in recent_mem.split("\n"):
            if line.strip() and line.strip() not in (all_memories_text or ""):
                all_memories_text += "\n" + line.strip()

    # ── 3) Conversas passadas relevantes ─────────────────────────
    past_convs = get_past_conversations_context(
        query=req.message,
        current_session_id=session_id,
        max_sessions=3,
        max_messages_per_session=6,
    )

    # ── Montar system prompt com tudo ────────────────────────────
    sys_prompt = (
        "Você é um assistente inteligente da CGR Telecom. "
        "Responda em português brasileiro de forma clara e completa.\n\n"
        "REGRA IMPORTANTE: Você tem memória de longo prazo. Você DEVE usar as "
        "informações das memórias e conversas anteriores ao responder. "
        "Se o usuário perguntar algo que já foi discutido ou informado antes, "
        "responda com base nessas memórias. Priorize as memórias sobre contexto "
        "de documentos quando forem sobre informações pessoais do usuário."
    )

    if all_memories_text.strip():
        sys_prompt += (
            "\n\n🧠 MEMÓRIAS SALVAS (informações que o usuário já compartilhou — "
            "USE estas informações nas suas respostas):\n"
            + all_memories_text.strip()
        )

    if past_convs:
        sys_prompt += (
            "\n\n💬 CONVERSAS ANTERIORES RELEVANTES:\n"
            + past_convs
        )

    if doc_context:
        sys_prompt += (
            "\n\n📚 CONTEXTO DOS DOCUMENTOS TREINADOS:\n"
            + doc_context
        )

    messages = [{"role": "system", "content": sys_prompt}]

    # Adicionar a pergunta atual do usuário
    messages.append({"role": "user", "content": req.message})

    # Identificar fontes de documentos
    sources = []
    if doc_context:
        for line in doc_context.split("\n"):
            if line.startswith("[Fonte: ") and line.endswith("]"):
                src = line[8:-1]
                if src not in sources:
                    sources.append(src)
    if memories:
        sources.append("💾 Memória de conversas")

    # ── Gerar resposta com streaming ─────────────────────────────
    user_message = req.message  # capturar para usar no generator

    async def generate():
        full_response = ""
        try:
            if sources:
                yield _sse("sources", {"sources": sources})

            # Passa o session_id para o chat_stream para que ele possa buscar o histórico
            async for token in chat_stream(messages, session_id=session_id):
                full_response += token
                yield _sse("token", {"token": token})

            # Salvar MENSAGEM DO USUÁRIO, resposta e extrair fatos para memória
            db2 = SessionLocal()
            try:
                # Salva a mensagem do usuário que iniciou esta troca
                db2.add(ChatMessage(
                    session_id=session_id, role="user", content=user_message
                ))
                # Salva a resposta do assistente
                db2.add(ChatMessage(
                    session_id=session_id, role="assistant", content=full_response
                ))
                sess2 = db2.query(ChatSession).filter(ChatSession.id == session_id).first()
                if sess2:
                    sess2.preview = user_message[:80]
                db2.commit()
            except Exception:
                db2.rollback()
            finally:
                db2.close()

            # ── APRENDER COM O CHAT: extrair fatos e salvar ──────
            try:
                facts = extract_facts_from_exchange(
                    user_msg=user_message,
                    assistant_msg=full_response,
                    session_id=session_id,
                )
                if facts:
                    store_facts(facts, session_id=session_id, category="chat")
            except Exception:
                pass  # Não interromper o chat por falha na memória

            yield _sse("done", {"session_id": session_id})

        except Exception as e:
            yield _sse("error", {"error": str(e)})

    return _sse_response(generate())


# ══════════════════════════════════════════════════════════════════
#  CHAT SESSIONS
# ══════════════════════════════════════════════════════════════════
@app.get("/api/chat/sessions")
async def list_sessions():
    db = SessionLocal()
    try:
        sessions = db.query(ChatSession).order_by(ChatSession.created_at.desc()).limit(30).all()
        return {
            "sessions": [
                {"id": s.id, "preview": s.preview or "Nova conversa"}
                for s in sessions
            ]
        }
    except Exception:
        return {"sessions": []}
    finally:
        db.close()


@app.get("/api/chat/history/{session_id}")
async def get_session_history(session_id: str):
    db = SessionLocal()
    try:
        msgs = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
            .all()
        )
        return {
            "messages": [
                {"role": m.role, "content": m.content}
                for m in msgs
            ]
        }
    except Exception:
        return {"messages": []}
    finally:
        db.close()


@app.delete("/api/chat/sessions/{session_id}")
async def delete_session(session_id: str):
    db = SessionLocal()
    try:
        sess = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if sess:
            db.delete(sess)
            db.commit()
            return {"ok": True}
        raise HTTPException(404, "Sessão não encontrada")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(500, str(e))
    finally:
        db.close()


# ══════════════════════════════════════════════════════════════════
#  LEARN — SSE streaming (upload + processamento)
# ══════════════════════════════════════════════════════════════════
@app.post("/api/learn")
async def learn(files: List[UploadFile] = File(...)):
    """Recebe arquivos, processa e retorna progresso via SSE."""

    async def generate():
        processed_names = []

        for i, file in enumerate(files):
            filename = file.filename or f"arquivo_{i}"
            try:
                yield _sse("progress", {"step": 1, "total": 3, "label": f"Recebendo {filename}..."})
                yield _sse("log", {"msg": f"📥 Recebendo: {filename}"})
                await asyncio.sleep(0.2)

                filepath = os.path.join(UPLOAD_DIR, filename)
                content = await file.read()
                with open(filepath, "wb") as f:
                    f.write(content)

                yield _sse("log", {"msg": f"💾 Salvo: {filename} ({len(content)} bytes)"})

                yield _sse("progress", {"step": 2, "total": 3, "label": f"Processando {filename}..."})
                yield _sse("log", {"msg": f"⚙️ Extraindo texto e criando chunks..."})
                await asyncio.sleep(0.2)

                result = process_and_store_document(filepath, filename)

                if result["success"]:
                    yield _sse("log", {"msg": f"✅ {filename}: {result['chunks']} chunks salvos no banco SQLite"})
                    processed_names.append(filename)
                else:
                    yield _sse("log", {"msg": f"❌ {filename}: {result.get('error', 'erro')}"})
                    yield _sse("error", {"msg": result.get("error", "Erro ao processar")})
                    continue

                yield _sse("progress", {"step": 3, "total": 3, "label": "Base atualizada!"})
                yield _sse("log", {"msg": "📚 Base de conhecimento atualizada!"})
                await asyncio.sleep(0.2)

            except Exception as e:
                yield _sse("error", {"msg": f"Erro com {filename}: {str(e)}"})
                yield _sse("log", {"msg": f"❌ Exceção: {str(e)}"})

        yield _sse("done", {"files": processed_names})

    return _sse_response(generate())


# ══════════════════════════════════════════════════════════════════
#  KNOWLEDGE STATS
# ══════════════════════════════════════════════════════════════════
@app.get("/api/knowledge/stats")
async def knowledge_stats():
    db = SessionLocal()
    try:
        docs = db.query(Document).all()
        total_chunks = db.query(KnowledgeChunk).count()
        total_memories = db.query(ChatMemory).count()
        total_sessions = db.query(ChatSession).count()
        sources = [d.filename for d in docs]
        return {
            "total_chunks": total_chunks,
            "total_documents": len(docs),
            "total_memories": total_memories,
            "total_sessions": total_sessions,
            "sources": sources,
        }
    except Exception:
        return {"total_chunks": 0, "total_documents": 0, "total_memories": 0, "total_sessions": 0, "sources": []}
    finally:
        db.close()


@app.get("/api/memories")
async def list_memories():
    """Lista todas as memórias (fatos) aprendidos via chat."""
    db = SessionLocal()
    try:
        mems = db.query(ChatMemory).order_by(ChatMemory.created_at.desc()).limit(50).all()
        return {
            "memories": [
                {
                    "id": m.id,
                    "fact": m.fact,
                    "category": m.category,
                    "session_id": m.source_session_id,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                }
                for m in mems
            ],
            "total": db.query(ChatMemory).count(),
        }
    except Exception:
        return {"memories": [], "total": 0}
    finally:
        db.close()


@app.delete("/api/memories/{mem_id}")
async def delete_memory(mem_id: int):
    db = SessionLocal()
    try:
        mem = db.query(ChatMemory).filter(ChatMemory.id == mem_id).first()
        if mem:
            db.delete(mem)
            db.commit()
            return {"ok": True}
        raise HTTPException(404, "Memória não encontrada")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(500, str(e))
    finally:
        db.close()


@app.post("/api/knowledge/ingest")
async def knowledge_ingest():
    """Força ingestão de arquivos da pasta saida_ia."""
    logs = []
    saida_dir = os.path.join(BASE_DIR, "saida_ia")
    if not os.path.exists(saida_dir):
        return {"logs": ["📁 Pasta saida_ia não encontrada"]}

    for fname in os.listdir(saida_dir):
        fpath = os.path.join(saida_dir, fname)
        if not os.path.isfile(fpath):
            continue

        db = SessionLocal()
        try:
            existing = db.query(Document).filter(Document.filename == fname).first()
            if existing:
                logs.append(f"⏭️ {fname} já existe")
                continue

            text = extract_text_from_file(fpath)
            if not text or text.startswith("[Erro"):
                logs.append(f"❌ {fname}: {text}")
                continue

            chunks = split_text_into_chunks(text)
            ext = os.path.splitext(fname)[1].lower()
            doc_type = {".md": "Markdown", ".jsonl": "JSONL", ".txt": "Texto"}.get(ext, "Outro")

            doc = Document(filename=fname, content=text, doc_type=doc_type, chunks_count=len(chunks))
            db.add(doc)
            db.flush()

            for i, chunk_text in enumerate(chunks):
                db.add(KnowledgeChunk(
                    document_id=doc.id, chunk_index=i,
                    content=chunk_text, source=fname,
                ))

            db.commit()
            logs.append(f"✅ {fname}: {len(chunks)} chunks ingeridos no SQLite")
        except Exception as e:
            db.rollback()
            logs.append(f"❌ {fname}: {str(e)}")
        finally:
            db.close()

    return {"logs": logs}


# ══════════════════════════════════════════════════════════════════
#  PIPELINE (advanced panel)
# ══════════════════════════════════════════════════════════════════
class PipelineRequest(BaseModel):
    profile: str = "safe"
    force_overwrite: bool = False


@app.post("/api/process")
async def start_process(req: PipelineRequest):
    global _pipeline_running, _pipeline_cancel, _pipeline_logs
    if _pipeline_running:
        raise HTTPException(409, "Pipeline já está rodando")

    _pipeline_running = True
    _pipeline_cancel = False
    _pipeline_logs = []

    input_dir = os.path.join(BASE_DIR, "entrada_cgr")
    os.makedirs(input_dir, exist_ok=True)

    async def run_pipeline():
        global _pipeline_running, _pipeline_cancel
        try:
            files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
            _pipeline_logs.append(f"📂 {len(files)} arquivos em entrada_cgr/")

            for i, fname in enumerate(files):
                if _pipeline_cancel:
                    _pipeline_logs.append("⛔ Cancelado")
                    break
                fpath = os.path.join(input_dir, fname)
                _pipeline_logs.append(f"⚙️ [{i+1}/{len(files)}] {fname}")
                result = process_and_store_document(fpath, fname)
                if result["success"]:
                    _pipeline_logs.append(f"✅ {fname}: {result['chunks']} chunks")
                else:
                    _pipeline_logs.append(f"❌ {fname}: {result.get('error', 'erro')}")

            _pipeline_logs.append("🏁 Pipeline concluído!")
        except Exception as e:
            _pipeline_logs.append(f"❌ Erro: {str(e)}")
        finally:
            _pipeline_running = False

    asyncio.create_task(run_pipeline())
    return {"status": "started"}


@app.get("/api/process/logs")
async def process_logs():
    async def generate():
        sent = 0
        while _pipeline_running or sent < len(_pipeline_logs):
            while sent < len(_pipeline_logs):
                yield _sse("log", {"msg": _pipeline_logs[sent]})
                sent += 1
            if not _pipeline_running:
                break
            await asyncio.sleep(0.5)
        yield _sse("done", {})

    return _sse_response(generate())


@app.post("/api/process/cancel")
async def cancel_process():
    global _pipeline_cancel
    _pipeline_cancel = True
    return {"status": "cancelling"}


# ══════════════════════════════════════════════════════════════════
#  DOCUMENTS (CRUD)
# ══════════════════════════════════════════════════════════════════
@app.get("/api/documents")
async def list_documents():
    docs = get_all_documents()
    return {"documents": docs, "total": len(docs)}


@app.delete("/api/documents/{doc_id}")
async def remove_document(doc_id: int):
    if delete_document(doc_id):
        return {"message": "Removido"}
    raise HTTPException(404, "Não encontrado")


# ══════════════════════════════════════════════════════════════════
#  HEALTH
# ══════════════════════════════════════════════════════════════════
@app.get("/api/health")
async def health():
    return {"status": "ok"}
