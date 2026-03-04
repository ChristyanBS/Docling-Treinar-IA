"""
Serviço de memória contínua da IA.
Armazena fatos, conversas e aprendizados no PostgreSQL com embeddings pgvector.
"""
import re
import logging
from typing import List, Optional
from datetime import datetime

from app.database import SessionLocal, ChatMemory
from app.embedding_service import embed_texts, embed_single

logger = logging.getLogger(__name__)


def extract_facts_from_exchange(user_msg: str, ai_msg: str) -> List[str]:
    """Extrai fatos explícitos da troca de mensagens."""
    facts = []

    teaching_patterns = [
        r"(?:lembre|anote|memorize|grave|saiba)\s*(?:que|:)\s*(.+)",
        r"(?:a|o)\s+(\w+)\s+(?:é|são|fica|tem|usa|funciona)\s+(.+)",
        r"(?:ip|endereço|porta|vlan|olt|ont|onu|switch|roteador)\s+(.+?)(?:\.|$)",
        r"(?:para|pra)\s+(?:configurar|acessar|conectar)\s+(.+?)(?:\.|$)",
        r"(?:o modelo|a marca|o tipo)\s+(.+?)(?:\.|$)",
    ]

    for pattern in teaching_patterns:
        matches = re.findall(pattern, user_msg, re.IGNORECASE)
        for m in matches:
            if isinstance(m, tuple):
                fact = " ".join(m).strip()
            else:
                fact = m.strip()
            if len(fact) > 10:
                facts.append(fact)

    confirm_patterns = [
        r"(?:isso|correto|exato|sim|isso mesmo)",
        r"(?:não|errado|incorreto|na verdade)\s*[,.]?\s*(.+)",
    ]
    for pattern in confirm_patterns:
        matches = re.findall(pattern, user_msg, re.IGNORECASE)
        if matches:
            for m in matches:
                if isinstance(m, str) and len(m.strip()) > 10:
                    facts.append("Correção: " + m.strip())

    if not facts and len(user_msg) > 50:
        tech_keywords = [
            "ip", "vlan", "olt", "ont", "onu", "switch", "roteador",
            "porta", "config", "rede", "gpon", "epon", "nokia", "huawei",
            "datacom", "intelbras", "ospf", "bgp", "rack", "servidor",
        ]
        msg_lower = user_msg.lower()
        if any(kw in msg_lower for kw in tech_keywords):
            facts.append(user_msg[:500])

    return facts


def store_facts(session_id: str, user_msg: str, ai_msg: str) -> int:
    """Extrai fatos de uma troca e salva no banco com embeddings."""
    facts = extract_facts_from_exchange(user_msg, ai_msg)
    if not facts:
        return 0

    db = SessionLocal()
    try:
        embeddings = embed_texts(facts)
        if embeddings is None:
            embeddings = [None] * len(facts)

        stored = 0
        for fact, emb in zip(facts, embeddings):
            mem = ChatMemory(
                session_id=session_id,
                role="fact",
                content=fact,
                fact=fact,
                memory_type="learned_fact",
                relevance_score=0.8,
                embedding=emb,
            )
            db.add(mem)
            stored += 1

        db.commit()
        logger.info("Sessão %s: %d fatos salvos com embeddings", session_id, stored)
        return stored
    except Exception as e:
        db.rollback()
        logger.error("Erro ao salvar fatos: %s", e)
        return 0
    finally:
        db.close()


def store_interaction_memory(session_id: str, role: str, content: str) -> None:
    """Salva uma mensagem de conversa no banco com embedding."""
    db = SessionLocal()
    try:
        emb = embed_single(content[:1000])
        mem = ChatMemory(
            session_id=session_id,
            role=role,
            content=content,
            memory_type="conversation",
            relevance_score=0.5,
            embedding=emb,
        )
        db.add(mem)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error("Erro ao salvar interação: %s", e)
    finally:
        db.close()


def get_relevant_memories(query: str, max_results: int = 5) -> str:
    """Busca memórias relevantes usando busca vetorial."""
    db = SessionLocal()
    try:
        query_emb = embed_single(query)
        if query_emb is not None:
            results = (
                db.query(ChatMemory)
                .filter(ChatMemory.embedding.isnot(None))
                .filter(ChatMemory.memory_type == "learned_fact")
                .order_by(ChatMemory.embedding.cosine_distance(query_emb))
                .limit(max_results)
                .all()
            )
            if results:
                parts = []
                for r in results:
                    parts.append("- " + (r.fact or r.content))
                return "\n".join(parts)

        keywords = [w.lower() for w in query.split() if len(w) > 3]
        if not keywords:
            return ""

        all_mems = (
            db.query(ChatMemory)
            .filter(ChatMemory.memory_type == "learned_fact")
            .order_by(ChatMemory.created_at.desc())
            .limit(200)
            .all()
        )

        scored = []
        for mem in all_mems:
            content_lower = (mem.fact or mem.content or "").lower()
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > 0:
                scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:max_results]

        if top:
            parts = []
            for _, m in top:
                parts.append("- " + (m.fact or m.content))
            return "\n".join(parts)

        return ""
    except Exception as e:
        logger.error("Erro ao buscar memórias: %s", e)
        return ""
    finally:
        db.close()


def get_past_conversations_context(session_id: str, limit: int = 20) -> str:
    """Retorna as últimas mensagens da sessão atual."""
    db = SessionLocal()
    try:
        msgs = (
            db.query(ChatMemory)
            .filter(ChatMemory.session_id == session_id)
            .filter(ChatMemory.memory_type == "conversation")
            .order_by(ChatMemory.created_at.desc())
            .limit(limit)
            .all()
        )
        msgs.reverse()
        if not msgs:
            return ""
        parts = []
        for m in msgs:
            parts.append(m.role + ": " + m.content)
        return "\n".join(parts)
    except Exception as e:
        logger.error("Erro ao buscar conversas: %s", e)
        return ""
    finally:
        db.close()


def get_recent_memories_summary(limit: int = 10) -> str:
    """Retorna resumo dos fatos recentes aprendidos."""
    db = SessionLocal()
    try:
        mems = (
            db.query(ChatMemory)
            .filter(ChatMemory.memory_type == "learned_fact")
            .order_by(ChatMemory.created_at.desc())
            .limit(limit)
            .all()
        )
        if not mems:
            return ""
        parts = []
        for m in mems:
            parts.append("- " + (m.fact or m.content))
        return "Fatos recentes aprendidos:\n" + "\n".join(parts)
    except Exception as e:
        logger.error("Erro ao buscar resumo: %s", e)
        return ""
    finally:
        db.close()


def backfill_memory_embeddings(batch_size: int = 32) -> dict:
    """Gera embeddings para memórias que ainda não têm."""
    db = SessionLocal()
    try:
        mems = (
            db.query(ChatMemory)
            .filter(ChatMemory.embedding.is_(None))
            .limit(batch_size)
            .all()
        )
        if not mems:
            return {"updated": 0, "message": "Nenhuma memória sem embedding"}

        texts = []
        for m in mems:
            texts.append((m.fact or m.content)[:1000])
        embeddings = embed_texts(texts)

        if embeddings is None:
            return {"updated": 0, "error": "Não foi possível gerar embeddings"}

        updated = 0
        for mem, emb in zip(mems, embeddings):
            if emb is not None:
                mem.embedding = emb
                updated += 1

        db.commit()
        remaining = (
            db.query(ChatMemory)
            .filter(ChatMemory.embedding.is_(None))
            .count()
        )
        return {"updated": updated, "remaining": remaining}
    except Exception as e:
        db.rollback()
        logger.error("Erro no backfill de memórias: %s", e)
        return {"updated": 0, "error": str(e)}
    finally:
        db.close()