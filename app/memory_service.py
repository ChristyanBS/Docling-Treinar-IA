"""
Serviço de memória conversacional.
- Extrai fatos/informações importantes de cada conversa e salva no PostgreSQL.
- Busca memórias relevantes por similaridade vetorial (pgvector).
- Busca histórico de conversas anteriores por similaridade.
"""

import logging
from datetime import datetime
from typing import List, Tuple

from sqlalchemy import text as sa_text

from app.database import (
    ChatMemory, ChatMessage, ChatSession, SessionLocal,
)
from app.embedding_service import embed_single, embed_texts

logger = logging.getLogger(__name__)


# ── Palavras indicadoras de informação importante ────────────────
_FACT_INDICATORS = [
    "meu nome é", "me chamo", "eu sou", "trabalho com", "trabalho na",
    "meu ip é", "o ip é", "o endereço é", "a senha é", "o servidor é",
    "o modelo é", "a porta é", "estou usando", "nós usamos", "a empresa",
    "o cliente", "o contrato", "o número", "sempre que", "lembre",
    "anota", "guarda isso", "não esqueça", "importante:",
    "meu email", "meu telefone", "o problema é", "a solução é",
    "a rede", "a vlan", "o switch", "o roteador", "o olt",
    "configuração", "topologia", "endereço", "equipamento",
]

# Frases que indicam pedido de memória ("lembre", "guarde", etc.)
_EXPLICIT_MEMORY = [
    "lembre", "lembra", "anota", "guarda", "guarde", "memorize",
    "não esqueça", "nao esqueca", "salva isso", "salve isso",
    "registra isso", "importante:", "nota:",
]


def extract_facts_from_exchange(
    user_msg: str,
    assistant_msg: str,
    session_id: str = "",
) -> List[str]:
    """
    Analisa uma troca user↔assistant e extrai fatos relevantes
    para salvar na memória de longo prazo.
    """
    facts: List[str] = []
    user_lower = user_msg.lower()

    # 1) O usuário pediu explicitamente para lembrar algo?
    for trigger in _EXPLICIT_MEMORY:
        if trigger in user_lower:
            facts.append(f"O usuário pediu para lembrar: {user_msg.strip()}")
            return facts  # Não precisa analisar mais

    # 2) A mensagem contém indicadores de informação útil?
    matched_indicators = []
    for indicator in _FACT_INDICATORS:
        if indicator in user_lower:
            matched_indicators.append(indicator)

    if matched_indicators:
        # Extrair frases que contêm os indicadores
        sentences = [s.strip() for s in user_msg.replace("\n", ". ").split(".") if s.strip()]
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for indicator in matched_indicators:
                if indicator in sentence_lower and len(sentence.strip()) > 10:
                    facts.append(sentence.strip())
                    break  # Evitar duplicar mesma frase

    # 3) Se a mensagem for longa o suficiente e informativa, salvar resumo
    #    (mensagens acima de 50 chars que não são perguntas simples)
    if not facts and len(user_msg) > 80:
        is_question = user_lower.strip().endswith("?")
        has_info_words = any(w in user_lower for w in [
            "é", "são", "está", "funciona", "usa", "preciso",
            "configurei", "instalei", "mudei", "criei", "fiz",
        ])
        if not is_question and has_info_words:
            # Salvar as primeiras 2 frases como contexto
            sentences = [s.strip() for s in user_msg.split(".") if len(s.strip()) > 15]
            for s in sentences[:2]:
                facts.append(s.strip())

    return facts[:5]  # Máximo de 5 fatos por troca


def store_facts(facts: List[str], session_id: str = "", category: str = "general"):
    """Salva fatos extraídos na tabela chat_memories com embeddings."""
    if not facts:
        return

    db = SessionLocal()
    try:
        # Filtrar fatos válidos e não duplicados
        new_facts = []
        for fact in facts:
            if len(fact.strip()) < 5:
                continue
            existing = db.query(ChatMemory).filter(ChatMemory.fact == fact.strip()).first()
            if existing:
                continue
            new_facts.append(fact.strip())

        if not new_facts:
            return

        # Gerar embeddings em batch
        embeddings = embed_texts(new_facts)

        for fact_text, emb in zip(new_facts, embeddings):
            db.add(ChatMemory(
                fact=fact_text,
                source_session_id=session_id,
                category=category,
                embedding=emb,
            ))
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error("Erro ao salvar fatos: %s", e)
    finally:
        db.close()


def get_relevant_memories(query: str, max_results: int = 5) -> List[str]:
    """
    Busca memórias (fatos de conversas anteriores) relevantes para a query.
    Usa busca vetorial (cosine distance) com fallback para palavras-chave.
    """
    db = SessionLocal()
    try:
        # ── 1) Busca vetorial ────────────────────────────────────
        query_emb = embed_single(query)
        has_embedding = any(v != 0.0 for v in query_emb)

        if has_embedding:
            results = db.execute(
                sa_text("""
                    SELECT fact, embedding <=> :qvec AS dist
                    FROM chat_memories
                    WHERE embedding IS NOT NULL
                    ORDER BY dist ASC
                    LIMIT :lim
                """),
                {"qvec": str(query_emb), "lim": max_results},
            ).fetchall()

            if results:
                relevant = [fact for fact, dist in results if dist < 0.80]
                if relevant:
                    return relevant

        # ── 2) Fallback: palavras-chave ──────────────────────────
        all_memories = db.query(ChatMemory).order_by(ChatMemory.created_at.desc()).all()
        if not all_memories:
            return []

        query_words = set(query.lower().split())
        scored: List[Tuple[int, str]] = []
        for mem in all_memories:
            mem_words = set(mem.fact.lower().split())
            score = len(query_words.intersection(mem_words))
            if score > 0:
                scored.append((score, mem.fact))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [fact for _, fact in scored[:max_results]]
    except Exception as e:
        logger.error("Erro em get_relevant_memories: %s", e)
        return []
    finally:
        db.close()


def get_past_conversations_context(
    query: str,
    current_session_id: str = "",
    max_sessions: int = 3,
    max_messages_per_session: int = 4,
) -> str:
    """
    Busca conversas ANTERIORES relevantes para a query atual.
    Retorna um resumo formatado das conversas passadas relacionadas.
    """
    db = SessionLocal()
    try:
        # Buscar todas as sessões que NÃO são a sessão atual
        sessions = (
            db.query(ChatSession)
            .filter(ChatSession.id != current_session_id)
            .order_by(ChatSession.created_at.desc())
            .limit(50)  # Buscar nas últimas 50 sessões
            .all()
        )

        if not sessions:
            return ""

        query_words = set(query.lower().split())
        scored_sessions: List[Tuple[int, str, list]] = []

        for sess in sessions:
            # Buscar mensagens da sessão
            msgs = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_id == sess.id)
                .order_by(ChatMessage.created_at)
                .all()
            )
            if not msgs:
                continue

            # Pontuar pela relevância das mensagens
            session_text = " ".join(m.content.lower() for m in msgs)
            session_words = set(session_text.split())
            score = len(query_words.intersection(session_words))

            if score > 0:
                msg_data = [(m.role, m.content) for m in msgs]
                scored_sessions.append((score, sess.id, msg_data))

        scored_sessions.sort(key=lambda x: x[0], reverse=True)

        # Formatar contexto das melhores sessões
        parts = []
        for _, sess_id, msgs in scored_sessions[:max_sessions]:
            conv_lines = []
            for role, content in msgs[-max_messages_per_session:]:
                prefix = "Usuário" if role == "user" else "Assistente"
                # Limitar tamanho do conteúdo
                short = content[:300] + "..." if len(content) > 300 else content
                conv_lines.append(f"  {prefix}: {short}")

            if conv_lines:
                parts.append("Conversa anterior:\n" + "\n".join(conv_lines))

        return "\n\n".join(parts)
    except Exception:
        return ""
    finally:
        db.close()


def get_recent_memories_summary(limit: int = 10) -> str:
    """Retorna um resumo das memórias mais recentes."""
    db = SessionLocal()
    try:
        memories = (
            db.query(ChatMemory)
            .order_by(ChatMemory.created_at.desc())
            .limit(limit)
            .all()
        )
        if not memories:
            return ""
        return "\n".join(f"- {m.fact}" for m in memories)
    except Exception:
        return ""
    finally:
        db.close()


def backfill_memory_embeddings(batch_size: int = 32) -> dict:
    """
    Gera embeddings para memórias que ainda não possuem (embedding IS NULL).
    Útil após migração de dados antigos do SQLite.
    """
    db = SessionLocal()
    try:
        pending = db.query(ChatMemory).filter(
            ChatMemory.embedding.is_(None)
        ).all()

        if not pending:
            return {"updated": 0, "message": "Todas as memórias já possuem embeddings."}

        total = len(pending)
        updated = 0

        for i in range(0, total, batch_size):
            batch = pending[i:i + batch_size]
            texts = [m.fact for m in batch]
            embeddings = embed_texts(texts)
            for mem, emb in zip(batch, embeddings):
                mem.embedding = emb
                updated += 1
            db.commit()
            logger.info("Backfill memórias: %d/%d atualizadas", updated, total)

        return {"updated": updated, "message": f"{updated} memórias atualizadas com embeddings."}
    except Exception as e:
        db.rollback()
        logger.error("Erro no backfill de memórias: %s", e)
        return {"updated": 0, "error": str(e)}
    finally:
        db.close()