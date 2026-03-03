"""
Serviço de memória conversacional.
- Extrai fatos/informações importantes de cada conversa e salva no SQLite.
- Busca memórias relevantes para enriquecer o contexto do chat.
- Busca histórico de conversas anteriores por similaridade.
"""

from datetime import datetime
from typing import List, Tuple

from app.database import (
    ChatMemory, ChatMessage, ChatSession, SessionLocal,
)


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
    """Salva fatos extraídos na tabela chat_memories."""
    if not facts:
        return

    db = SessionLocal()
    try:
        for fact in facts:
            if len(fact.strip()) < 5:
                continue
            # Evitar duplicatas exatas
            existing = db.query(ChatMemory).filter(ChatMemory.fact == fact.strip()).first()
            if existing:
                continue
            db.add(ChatMemory(
                fact=fact.strip(),
                source_session_id=session_id,
                category=category,
            ))
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def get_relevant_memories(query: str, max_results: int = 5) -> List[str]:
    """
    Busca memórias (fatos de conversas anteriores) relevantes para a query.
    Usa correspondência de palavras-chave.
    """
    db = SessionLocal()
    try:
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
    except Exception:
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
