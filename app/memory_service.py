"""
Serviço de memória contínua da IA.
Armazena fatos, conversas e aprendizados no PostgreSQL com embeddings pgvector.
"""
import re
import json
import logging
from typing import List, Optional
from datetime import datetime

import httpx

from app.config import OLLAMA_BASE_URL, CHAT_MODEL
from app.database import SessionLocal, ChatMemory
from app.embedding_service import embed_texts, embed_single

logger = logging.getLogger(__name__)

_LLM_EXTRACT_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)


def _llm_extract_facts(user_msg: str, ai_msg: str) -> List[str]:
    """Usa o LLM para extrair fatos importantes da conversa (roda em background)."""
    prompt = (
        "Extraia SOMENTE os fatos novos e importantes desta troca de mensagens.\n"
        "Fatos são informações sobre: pessoas, nomes, cargos, IPs, equipamentos, "
        "senhas, configurações, procedimentos, locais, telefones, equipe.\n"
        "Se não houver fatos novos, responda apenas: NENHUM\n"
        "Formato: um fato por linha, sem numeração.\n\n"
        f"Usuário: {user_msg[:500]}\n"
        f"Assistente: {ai_msg[:300]}\n\n"
        "Fatos extraídos:"
    )
    try:
        with httpx.Client(timeout=_LLM_EXTRACT_TIMEOUT) as client:
            resp = client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": CHAT_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_ctx": 1024, "temperature": 0.1, "num_predict": 200},
                },
            )
            if resp.status_code == 200:
                text = resp.json().get("response", "").strip()
                if "NENHUM" in text.upper() or not text:
                    return []
                lines = [l.strip().lstrip("•-*123456789. ") for l in text.split("\n")]
                return [l for l in lines if len(l) > 5 and "nenhum" not in l.lower()]
    except Exception as e:
        logger.debug("LLM fact extraction falhou (normal em background): %s", e)
    return []


def extract_facts_from_exchange(user_msg: str, ai_msg: str) -> List[str]:
    """Extrai fatos explícitos da troca de mensagens.
    Analisa tanto a mensagem do usuário quanto a confirmação da IA."""
    facts = []
    msg_lower = user_msg.lower().strip()

    # ── 1. Comandos explícitos de memorização ──
    explicit_patterns = [
        r"(?:lembre|anote|memorize|grave|saiba|salve|guarde|registre)\s*(?:que|isso|:|-)\s*(.+)",
        r"(?:lembre|anote|memorize|grave|saiba|salve|guarde|registre)\s+(.+)",
        r"(?:não\s+esqueça|preciso\s+que\s+(?:lembre|saiba))\s*(?:que|:)?\s*(.+)",
    ]
    for pattern in explicit_patterns:
        matches = re.findall(pattern, user_msg, re.IGNORECASE)
        for m in matches:
            fact = m.strip() if isinstance(m, str) else " ".join(m).strip()
            if len(fact) > 5:
                facts.append(fact)

    # ── 2. Definições e identificações (X é Y) ──
    identity_patterns = [
        # "Wallace é o coordenador", "Maria é a técnica"
        r"(\b[A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+)*)\s+(?:é|são|era|foi|será)\s+(.+?)(?:\.|!|\?|$)",
        # "o Wallace é ...", "a Maria é ..."
        r"(?:o|a|os|as)\s+(\w+)\s+(?:é|são|era|foi|será|fica|tem|usa|funciona|trabalha|mora)\s+(.+?)(?:\.|!|\?|$)",
        # "meu nome é ...", "eu sou ..."
        r"(?:meu\s+nome\s+(?:é|e)|eu\s+(?:sou|me\s+chamo))\s+(.+?)(?:\.|!|\?|,|$)",
        # "eu trabalho ...", "eu moro ..."
        r"eu\s+(?:trabalho|moro|fico|estou)\s+(.+?)(?:\.|!|\?|$)",
    ]
    for pattern in identity_patterns:
        matches = re.findall(pattern, user_msg, re.IGNORECASE)
        for m in matches:
            if isinstance(m, tuple):
                fact = " é ".join(p.strip() for p in m if p.strip())
            else:
                fact = m.strip()
            if len(fact) > 5 and fact not in facts:
                facts.append(fact)

    # ── 3. Informações pessoais e de equipe ──
    personal_patterns = [
        r"(?:meu|minha|nosso|nossa)\s+(\w+)\s+(?:é|são|fica|tem|se\s+chama)\s+(.+?)(?:\.|!|\?|$)",
        r"(?:aqui|na\s+empresa|no\s+trabalho|na\s+equipe).*?(?:é|são|fica|tem|usa|usamos)\s+(.+?)(?:\.|!|\?|$)",
    ]
    for pattern in personal_patterns:
        matches = re.findall(pattern, user_msg, re.IGNORECASE)
        for m in matches:
            if isinstance(m, tuple):
                fact = " ".join(p.strip() for p in m if p.strip())
            else:
                fact = m.strip()
            if len(fact) > 5 and fact not in facts:
                facts.append(fact)

    # ── 4. Informações técnicas de rede ──
    tech_patterns = [
        r"(?:ip|endereço|porta|vlan|olt|ont|onu|switch|roteador|gateway|dns|servidor)\s+(.+?)(?:\.|!|\?|$)",
        r"(?:para|pra)\s+(?:configurar|acessar|conectar|entrar)\s+(.+?)(?:\.|!|\?|$)",
        r"(?:o\s+modelo|a\s+marca|o\s+tipo|a\s+versão|o\s+firmware)\s+(.+?)(?:\.|!|\?|$)",
        r"(?:senha|login|usuário|acesso)\s+(?:é|do|da|para)\s+(.+?)(?:\.|!|\?|$)",
    ]
    for pattern in tech_patterns:
        matches = re.findall(pattern, user_msg, re.IGNORECASE)
        for m in matches:
            fact = m.strip() if isinstance(m, str) else " ".join(m).strip()
            if len(fact) > 5 and fact not in facts:
                facts.append(fact)

    # ── 5. Correções ──
    correction_patterns = [
        r"(?:não|errado|incorreto|na\s+verdade|corrigindo|correto\s+é)\s*[,.:;\-]?\s*(.+)",
    ]
    for pattern in correction_patterns:
        matches = re.findall(pattern, user_msg, re.IGNORECASE)
        for m in matches:
            if isinstance(m, str) and len(m.strip()) > 5:
                facts.append("Correção: " + m.strip())

    # ── 6. IA confirmou que aprendeu → salvar mensagem original como fato ──
    if not facts:
        ai_lower = ai_msg.lower() if ai_msg else ""
        ai_confirms = any(kw in ai_lower for kw in [
            "vou lembrar", "anotado", "memorizado", "salvei", "guardei",
            "registrado", "vou guardar", "aprendi", "entendido",
            "lembrar disso", "vou anotar", "vou salvar",
        ])
        if ai_confirms and len(user_msg) > 10:
            facts.append(user_msg[:500])

    # ── 7. Fallback: mensagem informativa longa com palavras-chave ──
    if not facts and len(user_msg) > 30:
        info_keywords = [
            "ip", "vlan", "olt", "ont", "onu", "switch", "roteador",
            "porta", "config", "rede", "gpon", "epon", "nokia", "huawei",
            "datacom", "intelbras", "ospf", "bgp", "rack", "servidor",
            "coordenador", "gerente", "técnico", "equipe", "responsável",
            "telefone", "email", "endereço", "localização", "setor",
        ]
        if any(kw in msg_lower for kw in info_keywords):
            facts.append(user_msg[:500])

    # Deduplicar e limpar
    seen = set()
    unique_facts = []
    for f in facts:
        f_clean = f.strip()
        if f_clean and f_clean.lower() not in seen:
            seen.add(f_clean.lower())
            unique_facts.append(f_clean)

    return unique_facts


def store_facts(session_id: str, user_msg: str, ai_msg: str) -> int:
    """Extrai fatos de uma troca (regex + LLM) e salva no banco com embeddings."""
    # Extração por regex (rápida)
    facts = extract_facts_from_exchange(user_msg, ai_msg)

    # Extração por LLM (mais inteligente, roda em background mesmo)
    try:
        llm_facts = _llm_extract_facts(user_msg, ai_msg)
        if llm_facts:
            existing_lower = {f.lower() for f in facts}
            for lf in llm_facts:
                if lf.lower() not in existing_lower:
                    facts.append(lf)
                    existing_lower.add(lf.lower())
    except Exception as e:
        logger.debug("LLM fact extraction error: %s", e)

    if not facts:
        return 0

    db = SessionLocal()
    try:
        # Verificar fatos duplicados já existentes no banco
        existing_facts = set()
        try:
            recent = (
                db.query(ChatMemory.fact)
                .filter(ChatMemory.memory_type == "learned_fact")
                .order_by(ChatMemory.created_at.desc())
                .limit(100)
                .all()
            )
            existing_facts = {r.fact.lower().strip() for r in recent if r.fact}
        except Exception:
            pass

        new_facts = [f for f in facts if f.lower().strip() not in existing_facts]
        if not new_facts:
            return 0

        embeddings = embed_texts(new_facts)
        if embeddings is None:
            embeddings = [None] * len(new_facts)

        stored = 0
        for fact, emb in zip(new_facts, embeddings):
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


def store_interaction_memory(session_id: str, role: str, content: str, skip_embedding: bool = False) -> None:
    """Salva uma mensagem de conversa no banco. skip_embedding=True para não bloquear."""
    db = SessionLocal()
    try:
        emb = None
        if not skip_embedding:
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


def get_relevant_memories(query: str, max_results: int = 5, query_embedding=None) -> str:
    """Busca memórias relevantes usando busca vetorial + keyword combinados."""
    db = SessionLocal()
    try:
        found = []
        seen_ids = set()

        # Busca vetorial
        query_emb = query_embedding if query_embedding is not None else embed_single(query)
        if query_emb is not None:
            results = (
                db.query(ChatMemory)
                .filter(ChatMemory.embedding.isnot(None))
                .filter(ChatMemory.memory_type == "learned_fact")
                .order_by(ChatMemory.embedding.cosine_distance(query_emb))
                .limit(max_results)
                .all()
            )
            for r in results:
                found.append(r)
                seen_ids.add(r.id)

        # Busca por keywords (complementa a vetorial)
        keywords = [w.lower() for w in query.split() if len(w) > 2]
        if keywords:
            all_mems = (
                db.query(ChatMemory)
                .filter(ChatMemory.memory_type == "learned_fact")
                .order_by(ChatMemory.created_at.desc())
                .limit(200)
                .all()
            )
            scored = []
            for mem in all_mems:
                if mem.id in seen_ids:
                    continue
                content_lower = (mem.fact or mem.content or "").lower()
                score = sum(1 for kw in keywords if kw in content_lower)
                if score > 0:
                    scored.append((score, mem))
            scored.sort(key=lambda x: x[0], reverse=True)
            for _, mem in scored[:max_results]:
                if mem.id not in seen_ids:
                    found.append(mem)
                    seen_ids.add(mem.id)

        if not found:
            return ""

        # Limitar ao max_results total
        found = found[:max_results * 2]  # incluir ambas as fontes
        parts = []
        for r in found:
            parts.append("- " + (r.fact or r.content))
        return "\n".join(parts)
    except Exception as e:
        logger.error("Erro ao buscar memórias: %s", e)
        return ""
    finally:
        db.close()


def get_past_conversations_context(session_id: str, limit: int = 10) -> list:
    """Retorna as últimas mensagens da sessão como lista de dicts {role, content}.
    Cada mensagem é truncada a 300 chars e o total é limitado a ~3000 chars."""
    MAX_MSG_CHARS = 300
    MAX_TOTAL_CHARS = 3000
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
            return []
        result = []
        total = 0
        for m in msgs:
            content = m.content[:MAX_MSG_CHARS]
            if len(m.content) > MAX_MSG_CHARS:
                content += "..."
            total += len(content)
            if total > MAX_TOTAL_CHARS:
                break
            result.append({"role": m.role, "content": content})
        return result
    except Exception as e:
        logger.error("Erro ao buscar conversas: %s", e)
        return []
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